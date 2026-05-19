import hashlib
import importlib
import json
import tempfile
from pathlib import Path
from typing import cast
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file
from diffwofost.physical_models.config import ComputeConfig

_MODEL_CLASS_MODULE_KEY = "diffwofost.model_module"
_MODEL_CLASS_NAME_KEY = "diffwofost.model_class"
_MODEL_INIT_KWARGS_KEY = "diffwofost.init_kwargs"


def _normalize_path(path):
    return Path(path).expanduser().resolve()


def _serialize_init_kwargs(init_kwargs):
    return json.dumps(init_kwargs, sort_keys=True)


def _deserialize_init_kwargs(serialized_kwargs):
    return json.loads(serialized_kwargs)


def _default_model_directory():
    return Path(tempfile.gettempdir()) / "diffwofost-ml-models"


def _default_model_filename(model):
    init_kwargs = _serialize_init_kwargs(model.get_init_kwargs())
    structure_digest = hashlib.sha256(init_kwargs.encode("utf-8")).hexdigest()[:12]
    return f"{model.__class__.__name__.lower()}-{structure_digest}.safetensors"


def _load_model_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise ValueError(f"No metadata found in safetensors file: {path}")
    return metadata


class SafeTensorModelMixin:
    """Reusable safetensors-based persistence for diffWOFOST torch modules."""

    def get_init_kwargs(self):
        """Return constructor arguments needed to rebuild this model."""
        return {}

    def _build_safetensors_metadata(self):
        return {
            _MODEL_CLASS_MODULE_KEY: self.__class__.__module__,
            _MODEL_CLASS_NAME_KEY: self.__class__.__qualname__,
            _MODEL_INIT_KWARGS_KEY: _serialize_init_kwargs(self.get_init_kwargs()),
        }

    def save_model(self, path=None, filename=None, directory=None):
        """Persist the model weights and minimal constructor metadata."""
        torch_model = cast(torch.nn.Module, self)
        if path is not None and (filename is not None or directory is not None):
            raise ValueError("Pass either path or filename/directory, not both.")

        if path is None:
            target_directory = _default_model_directory() if directory is None else Path(directory)
            target_filename = _default_model_filename(self) if filename is None else filename
            path = Path(target_directory) / target_filename

        path = _normalize_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in torch_model.state_dict().items()
        }
        save_file(tensors, str(path), metadata=self._build_safetensors_metadata())
        return path

    @classmethod
    def load_model(cls, path, device=None, dtype=None):
        """Reconstruct a model instance from a safetensors file."""
        path = _normalize_path(path)
        metadata = _load_model_metadata(path)
        module_name = metadata.get(_MODEL_CLASS_MODULE_KEY)
        class_name = metadata.get(_MODEL_CLASS_NAME_KEY)
        if module_name != cls.__module__ or class_name != cls.__qualname__:
            raise ValueError(
                f"Safetensors file {path} stores {module_name}.{class_name}, "
                f"not {cls.__module__}.{cls.__qualname__}."
            )

        init_kwargs = _deserialize_init_kwargs(metadata[_MODEL_INIT_KWARGS_KEY])
        model = cls(**init_kwargs)
        torch_model = cast(torch.nn.Module, model)
        state_dict = load_file(str(path), device="cpu")
        torch_model.load_state_dict(state_dict)
        target_device = ComputeConfig.get_device() if device is None else device
        target_dtype = ComputeConfig.get_dtype() if dtype is None else dtype
        torch_model.to(device=target_device, dtype=target_dtype)
        return model


def load_model(path, device=None, dtype=None):
    """Load a diffWOFOST model from safetensors metadata without pickle."""
    path = _normalize_path(path)
    metadata = _load_model_metadata(path)
    module = importlib.import_module(metadata[_MODEL_CLASS_MODULE_KEY])
    model_class = getattr(module, metadata[_MODEL_CLASS_NAME_KEY])
    if not issubclass(model_class, SafeTensorModelMixin):
        raise TypeError(
            f"Model class {model_class.__module__}.{model_class.__qualname__} "
            "does not support safetensors persistence."
        )
    return model_class.load_model(path, device=device, dtype=dtype)
