import hashlib
import importlib
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file
from diffwofost.physical_models.config import ComputeConfig


def _default_model_filename(model):
    """Build a stable default filename from the model structure.

    The filename depends on the model class name and serialized constructor
    kwargs, so repeated saves of the same model structure reuse the same path
    unless a custom name is provided explicitly.

    Args:
        model (torch.nn.Module): Model instance to name.

    Returns:
        str: Default safetensors filename for this model structure.
    """
    init_kwargs = json.dumps(dict(getattr(model, "init_kwargs", {})), sort_keys=True)
    structure_digest = hashlib.sha256(init_kwargs.encode("utf-8")).hexdigest()[:12]
    return f"{model.__class__.__name__.lower()}-{structure_digest}.safetensors"


def _load_model_metadata(path):
    """Load metadata stored in a safetensors file.

    Args:
        path (str | Path): Path to the safetensors file.

    Returns:
        dict: Metadata dictionary stored alongside the tensors.

    Raises:
        ValueError: If the file does not contain metadata.
    """
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise ValueError(f"No metadata found in safetensors file: {path}")
    return metadata


def _build_safetensors_metadata(model):
    """Build the metadata needed to reconstruct a saved model.

    Args:
        model (torch.nn.Module): Model instance to describe.

    Returns:
        dict: Metadata with module, class, and constructor kwargs.
    """
    return {
        "diffwofost.model_module": model.__class__.__module__,
        "diffwofost.model_class": model.__class__.__qualname__,
        "diffwofost.init_kwargs": json.dumps(
            dict(getattr(model, "init_kwargs", {})),
            sort_keys=True,
        ),
    }


def save_model(model, path=None, filename=None, directory=None):
    """Persist a torch model with safetensors and constructor metadata.

    If no explicit path is provided, the model is saved under a stable default
    location in a hidden repository-local directory. The default filename depends on
    the model class name and stored constructor kwargs so repeated saves of the
    same model structure reuse the same file.

    Args:
        model (torch.nn.Module): Model instance to persist.
        path (str | Path | None): Full target path. When provided, `filename`
            and `directory` must be omitted.
        filename (str | None): Optional custom filename used with `directory`.
        directory (str | Path | None): Optional custom directory used with
            `filename` or the default filename.

    Returns:
        Path: Path of the saved safetensors file.

    Raises:
        ValueError: If `path` is combined with `filename` or `directory`.
    """
    if path is not None and (filename is not None or directory is not None):
        raise ValueError("Pass either path or filename/directory, not both.")

    if path is None:
        target_directory = (
            Path(__file__).resolve().parents[3] / ".diffwofost-ml-models"
            if directory is None
            else Path(directory)
        )
        target_filename = _default_model_filename(model) if filename is None else filename
        path = Path(target_directory) / target_filename

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(tensors, str(path), metadata=_build_safetensors_metadata(model))
    return path


def load_model(path, model_class=None, device=None, dtype=None):
    """Load a diffWOFOST model from a safetensors file.

    The model class is discovered from metadata by default. A caller can also
    provide `model_class` explicitly to validate that the stored class matches
    the expected one.

    Args:
        path (str | Path): Path to the saved safetensors file.
        model_class (type[torch.nn.Module] | None): Expected model class. When
            omitted, the class is resolved from the stored metadata.
        device (str | torch.device | None): Target device for the restored
            model. Defaults to the active `ComputeConfig` device.
        dtype (torch.dtype | None): Target dtype for the restored model.
            Defaults to the active `ComputeConfig` dtype.

    Returns:
        torch.nn.Module: Restored model instance with loaded parameters.

    Raises:
        ValueError: If the stored class does not match the provided
            `model_class`.
    """
    path = Path(path).expanduser().resolve()
    metadata = _load_model_metadata(path)
    stored_module_name = metadata.get("diffwofost.model_module")
    stored_class_name = metadata.get("diffwofost.model_class")

    if model_class is None:
        module = importlib.import_module(stored_module_name)
        model_class = getattr(module, stored_class_name)
    elif (
        stored_module_name != model_class.__module__
        or stored_class_name != model_class.__qualname__
    ):
        raise ValueError(
            f"Safetensors file {path} stores {stored_module_name}.{stored_class_name}, "
            f"not {model_class.__module__}.{model_class.__qualname__}."
        )

    init_kwargs = json.loads(metadata["diffwofost.init_kwargs"])
    model = model_class(**init_kwargs)
    state_dict = load_file(str(path), device="cpu")
    model.load_state_dict(state_dict)
    target_device = ComputeConfig.get_device() if device is None else device
    target_dtype = ComputeConfig.get_dtype() if dtype is None else dtype
    model.to(device=target_device, dtype=target_dtype)
    return model
