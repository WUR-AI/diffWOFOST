"""Save and load diffWOFOST models via safetensors.

Provides a single pair of functions — :func:`save_model` and :func:`load_model` —
that handle both **ML models** (``torch.nn.Module``) and **physical models**
(``Configuration`` + ``ParameterProvider``):

* ML models are persisted with ``state_dict()`` and constructor metadata in a
  single ``.safetensors`` file.
* Physical models are persisted as a directory containing ``config.json``
  (class references + output variables), ``parameters.safetensors`` (tensor
  parameters), ``parameters.json`` (scalar/list parameters), and any embedded
  ML models (``CROP_NN_MODEL`` / ``CROP_COMPONENTS``).

The format uses `SafeTensors <https://github.com/safetensors/safetensors>`_
rather than pickle, avoiding the security and portability concerns of
``torch.save`` / ``torch.load``.
"""

import hashlib
import importlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration


def _class_ref(cls):
    """``{module, qualname}`` for a Python class."""
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def _load_class(ref):
    """Import and return a class from a ``_class_ref`` dict."""
    module = importlib.import_module(ref["module"])
    return getattr(module, ref["qualname"])


def _default_model_filename(model):
    """Stable filename from class name + init_kwargs hash."""
    init_kwargs = json.dumps(dict(getattr(model, "init_kwargs", {})), sort_keys=True)
    digest = hashlib.sha256(init_kwargs.encode("utf-8")).hexdigest()[:12]
    return f"{model.__class__.__name__.lower()}-{digest}.safetensors"


def _save_torch_model(model, path):
    """Persist a torch.nn.Module to a safetensors file."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {n: t.detach().cpu().contiguous() for n, t in model.state_dict().items()}
    metadata = {
        "diffwofost.model_module": model.__class__.__module__,
        "diffwofost.model_class": model.__class__.__qualname__,
        "diffwofost.init_kwargs": json.dumps(
            dict(getattr(model, "init_kwargs", {})),
            sort_keys=True,
        ),
    }
    save_file(tensors, str(path), metadata=metadata)
    return path


def _load_torch_model(path, model_class=None, device=None, dtype=None):
    """Restore a torch.nn.Module from a safetensors file."""
    path = Path(path).expanduser().resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise ValueError(f"No metadata found in safetensors file: {path}")

    stored_mod = metadata.get("diffwofost.model_module")
    stored_cls = metadata.get("diffwofost.model_class")

    if model_class is None:
        model_class = getattr(importlib.import_module(stored_mod), stored_cls)
    elif stored_mod != model_class.__module__ or stored_cls != model_class.__qualname__:
        raise ValueError(
            f"Safetensors file {path} stores {stored_mod}.{stored_cls}, "
            f"not {model_class.__module__}.{model_class.__qualname__}."
        )

    model = model_class(**json.loads(metadata["diffwofost.init_kwargs"]))
    model.load_state_dict(load_file(str(path), device="cpu"))
    target_device = ComputeConfig.get_device() if device is None else device
    target_dtype = ComputeConfig.get_dtype() if dtype is None else dtype
    return model.to(device=target_device, dtype=target_dtype)


def save_model(
    path=None, *, model=None, config=None, parameterprovider=None, filename=None, directory=None
):
    """Save a diffWOFOST model (ML and/or physical) to disk.

    **ML model only**::

        save_model(path="model.safetensors", model=my_ml_model)
        save_model(model=my_ml_model)  # auto-names under .diffwofost-ml-models/

    **Physical model (Configuration + ParameterProvider)**::

        save_model(path="my_model_dir", config=my_config, parameterprovider=my_provider)

    Embedded ML models (``CROP_NN_MODEL`` / ``CROP_COMPONENTS`` entries) are
    saved alongside automatically.

    Args:
        path: Target file (ML model) or directory (physical model). When
            ``model`` is given without ``path``, a default location is used.
        model: ``torch.nn.Module`` instance to persist.
        config: ``Configuration`` for a physical model.
        parameterprovider: PCSE ``ParameterProvider`` with parameter values.
        filename: Custom filename (ML model, used with ``directory``).
        directory: Custom directory (ML model, used with ``filename``).

    Returns:
        Path: Path to the saved file or directory.
    """
    # --- ML model path ---
    if model is not None:
        if config is not None or parameterprovider is not None:
            raise ValueError("Pass either 'model' or 'config'+'parameterprovider', not both.")
        if path is not None and (filename is not None or directory is not None):
            raise ValueError("Pass either 'path' or 'filename'/'directory', not both.")
        if path is None:
            target_dir = (
                Path.cwd().resolve() / ".diffwofost-ml-models"
                if directory is None
                else Path(directory)
            )
            path = target_dir / (_default_model_filename(model) if filename is None else filename)
        return _save_torch_model(model, path)

    # --- Physical model path ---
    if config is not None:
        if path is None:
            raise ValueError("'path' is required when saving a physical model.")
        root = Path(path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        # ---- config.json ----
        cfg = {
            "CROP": _class_ref(config.CROP),
            "OUTPUT_VARS": config.OUTPUT_VARS,
            "SUMMARY_OUTPUT_VARS": config.SUMMARY_OUTPUT_VARS,
            "TERMINAL_OUTPUT_VARS": config.TERMINAL_OUTPUT_VARS,
            "OUTPUT_INTERVAL": config.OUTPUT_INTERVAL,
            "OUTPUT_INTERVAL_DAYS": config.OUTPUT_INTERVAL_DAYS,
            "OUTPUT_WEEKDAY": config.OUTPUT_WEEKDAY,
            "model_config_file": str(config.model_config_file)
            if config.model_config_file
            else None,
            "description": config.description,
            "AGROMANAGEMENT": _class_ref(config.AGROMANAGEMENT),
            "SOIL": _class_ref(config.SOIL) if config.SOIL is not None else None,
        }
        if config.CROP_NN_MODEL is not None:
            cfg["CROP_NN_MODEL"] = _class_ref(
                config.CROP_NN_MODEL.__class__
                if isinstance(config.CROP_NN_MODEL, torch.nn.Module)
                else config.CROP_NN_MODEL
            )
            cfg["CROP_NN_MODEL_is_instance"] = isinstance(config.CROP_NN_MODEL, torch.nn.Module)
        else:
            cfg["CROP_NN_MODEL"] = None

        if config.CROP_COMPONENTS:
            components = {}
            for name, override in config.CROP_COMPONENTS.items():
                entry: dict = {"class": _class_ref(override["class"])}
                m = override.get("model")
                if m is not None:
                    entry["model"] = _class_ref(
                        m.__class__ if isinstance(m, torch.nn.Module) else m
                    )
                    entry["model_is_instance"] = isinstance(m, torch.nn.Module)
                for k, v in override.items():
                    if k not in ("class", "model") and k not in entry:
                        entry[k] = v
                components[name] = entry
            cfg["CROP_COMPONENTS"] = components
        else:
            cfg["CROP_COMPONENTS"] = None

        with open(root / "config.json", "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True, default=str)

        # ---- parameters (tensors → safetensors, rest → JSON) ----
        all_tensors = {}
        nontensor_data = {}
        for group_name in ("_sitedata", "_timerdata", "_soildata", "_cropdata", "_override"):
            group = getattr(parameterprovider, group_name, {})
            if not group:
                continue
            nt = {}
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    all_tensors[f"{group_name}__{key}"] = value.detach().cpu().contiguous()
                else:
                    nt[key] = value
            if nt:
                nontensor_data[group_name] = nt

        if all_tensors:
            save_file(all_tensors, str(root / "parameters.safetensors"))
        if nontensor_data:
            with open(root / "parameters.json", "w", encoding="utf-8") as fh:
                json.dump(nontensor_data, fh, indent=2, sort_keys=True, default=str)

        # ---- embedded ML models ----
        if config.CROP_NN_MODEL is not None and isinstance(config.CROP_NN_MODEL, torch.nn.Module):
            _save_torch_model(config.CROP_NN_MODEL, root / "crop_nn_model.safetensors")
        if config.CROP_COMPONENTS:
            comp_dir = root / "crop_components"
            for comp_name, override in config.CROP_COMPONENTS.items():
                m = override.get("model") if isinstance(override, dict) else None
                if isinstance(m, torch.nn.Module):
                    comp_dir.mkdir(parents=True, exist_ok=True)
                    _save_torch_model(m, comp_dir / f"{comp_name}.safetensors")
        return root

    raise ValueError("Provide either 'model' or 'config'+'parameterprovider'.")


def load_model(path, *, model_class=None, device=None, dtype=None):
    """Load a diffWOFOST model from disk.

    Automatically detects the save format:

    * If *path* is a ``.safetensors`` file → returns a ``torch.nn.Module``.
    * If *path* is a directory → returns ``(Configuration, ParameterProvider)``.

    Args:
        path: Path to a safetensors file or a directory saved by
            :func:`save_model`.
        model_class: Expected ML model class (optional validation).
        device: Target device. Defaults to ``ComputeConfig`` device.
        dtype: Target dtype. Defaults to ``ComputeConfig`` dtype.

    Returns:
        ``torch.nn.Module`` or ``(Configuration, ParameterProvider)``.
    """
    path = Path(path).expanduser().resolve()

    if dtype is None:
        dtype = ComputeConfig.get_dtype()
    if device is None:
        device = ComputeConfig.get_device()

    # --- Directory → physical model ---
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Not a model directory (no config.json): {path}")
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)

        # Rebuild Configuration
        soil_cls = _load_class(cfg["SOIL"]) if cfg.get("SOIL") else None
        crop_nn_model_cls = _load_class(cfg["CROP_NN_MODEL"]) if cfg.get("CROP_NN_MODEL") else None

        crop_components = None
        if cfg.get("CROP_COMPONENTS"):
            crop_components = {}
            for name, override in cfg["CROP_COMPONENTS"].items():
                resolved = {"class": _load_class(override["class"])}
                if "model" in override:
                    resolved["model"] = _load_class(override["model"])
                for k, v in override.items():
                    if k not in ("class", "model", "model_is_instance") and k not in resolved:
                        resolved[k] = v
                crop_components[name] = resolved

        mcf = cfg.get("model_config_file")
        config = Configuration(
            CROP=_load_class(cfg["CROP"]),
            CROP_COMPONENTS=crop_components,
            CROP_NN_MODEL=crop_nn_model_cls,
            SOIL=soil_cls,
            AGROMANAGEMENT=_load_class(cfg["AGROMANAGEMENT"]),
            OUTPUT_VARS=cfg["OUTPUT_VARS"],
            SUMMARY_OUTPUT_VARS=cfg.get("SUMMARY_OUTPUT_VARS", []),
            TERMINAL_OUTPUT_VARS=cfg.get("TERMINAL_OUTPUT_VARS", []),
            OUTPUT_INTERVAL=cfg.get("OUTPUT_INTERVAL", "daily"),
            OUTPUT_INTERVAL_DAYS=cfg.get("OUTPUT_INTERVAL_DAYS", 1),
            OUTPUT_WEEKDAY=cfg.get("OUTPUT_WEEKDAY", 0),
            model_config_file=Path(mcf) if mcf else None,
            description=cfg.get("description"),
        )

        # Rebuild param data from safetensors + JSON
        param_data = {}
        sf_path = path / "parameters.safetensors"
        if sf_path.is_file():
            for safe_name, tensor in load_file(str(sf_path), device="cpu").items():
                group_name, pname = safe_name.split("__", 1)
                param_data.setdefault(group_name, {"tensor_params": {}, "nontensor_params": {}})
                param_data[group_name]["tensor_params"][pname] = tensor
        json_path = path / "parameters.json"
        if json_path.is_file():
            with open(json_path, encoding="utf-8") as fh:
                for group_name, params in json.load(fh).items():
                    param_data.setdefault(group_name, {"tensor_params": {}, "nontensor_params": {}})
                    param_data[group_name]["nontensor_params"] = params

        # Reconstruct ParameterProvider
        from pcse.base.parameter_providers import ParameterProvider

        init_kwargs = {}
        for group_name in ("sitedata", "timerdata", "soildata", "cropdata"):
            key = f"_{group_name}"
            if key in param_data:
                group_data = dict(param_data[key].get("nontensor_params", {}))
                for pname, pvalue in param_data[key].get("tensor_params", {}).items():
                    group_data[pname] = pvalue.to(dtype=dtype, device=device)
                init_kwargs[group_name] = group_data
        provider = ParameterProvider(**init_kwargs)
        override_data = param_data.get("_override", {})
        for pname, pvalue in override_data.get("nontensor_params", {}).items():
            provider.set_override(pname, pvalue, check=False)
        for pname, pvalue in override_data.get("tensor_params", {}).items():
            provider.set_override(pname, pvalue.to(dtype=dtype, device=device), check=False)

        # Restore embedded ML models
        nn_path = path / "crop_nn_model.safetensors"
        if nn_path.is_file():
            config.CROP_NN_MODEL = _load_torch_model(nn_path, device=device, dtype=dtype)
        comp_dir = path / "crop_components"
        if comp_dir.is_dir() and config.CROP_COMPONENTS:
            for comp_name in config.CROP_COMPONENTS:
                cmp = comp_dir / f"{comp_name}.safetensors"
                if cmp.is_file():
                    config.CROP_COMPONENTS[comp_name]["model"] = _load_torch_model(
                        cmp, device=device, dtype=dtype
                    )

        return config, provider

    # --- File → ML model ---
    return _load_torch_model(path, model_class=model_class, device=device, dtype=dtype)
