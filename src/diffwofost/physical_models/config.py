import importlib
import inspect
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Self
import pcse
import torch
from pcse.agromanager import AgroManager
from pcse.base import AncillaryObject
from pcse.base import SimulationObject


def class_ref(cls):
    """``{module, qualname}`` for a Python class."""
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def load_class(ref):
    """Import and return a class from a ``class_ref`` dict."""
    module = importlib.import_module(ref["module"])
    return getattr(module, ref["qualname"])


class ComputeConfig:
    """Central configuration for device and dtype settings.

    This class acts as a factory for default configuration settings that are
    captured by simulation objects upon initialization. This enables precise
    control over where (device) and how (dtype) each model computation occurs,
    allowing for multiple models with different configurations to coexist.

    **Key Concept: Configuration Capture**

    When a simulation object (e.g., `WOFOST_Leaf_Dynamics`) is initialized, it
    queries `ComputeConfig` for the current device and dtype. The model *captures*
    and stores these settings for its lifetime. Subsequent changes to
    `ComputeConfig` will only affect *newly created* objects, leaving existing
    ones unchanged.

    **Default Behavior:**

    - **Device**: Defaults to torch.get_default_device()
    - **Dtype**: Defaults to torch.get_default_dtype()

    **Basic Usage:**

        >>> from diffwofost.physical_models.config import ComputeConfig
        >>> import torch
        >>>
        >>> # Configure defaults for new models
        >>> ComputeConfig.set_device('cuda')
        >>> ComputeConfig.set_dtype(torch.float32)
        >>>
        >>> # Get current defaults
        >>> device = ComputeConfig.get_device()
        >>> dtype = ComputeConfig.get_dtype()

    **Creating Models with Different Settings:**

    Because models capture the configuration at initialization, you can create
    instances with different settings in the same process:

        >>> from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
        >>>
        >>> # Create a model on GPU (float32)
        >>> ComputeConfig.set_device('cuda')
        >>> ComputeConfig.set_dtype(torch.float32)
        >>> model_gpu = WOFOST_Leaf_Dynamics(...)
        >>>
        >>> # Create a model on CPU (float64)
        >>> ComputeConfig.set_device('cpu')
        >>> ComputeConfig.set_dtype(torch.float64)
        >>> model_cpu = WOFOST_Leaf_Dynamics(...)
        >>>
        >>> # model_gpu remains on cuda, model_cpu stays on cpu.

    Setting the model properties like model.device = torch.device("cpu") or
    model.dtype = torch.float64 returns `AttributeError`. Always use
    ComputeConfig.set_device(...) and ComputeConfig.set_dtype(...).

    **Resetting to Defaults:**

        >>> ComputeConfig.reset_to_defaults()

    """

    _device: torch.device = None
    _dtype: torch.dtype = None

    @classmethod
    def _initialize_defaults(cls):
        """Initialize default device and dtype if not already set."""
        if cls._device is None:
            cls._device = torch.get_default_device()
        if cls._dtype is None:
            cls._dtype = torch.get_default_dtype()

    @classmethod
    def get_device(cls) -> torch.device:
        """Get the current device setting.

        Returns:
            torch.device: The current device (cuda or cpu)
        """
        cls._initialize_defaults()
        return cls._device

    @classmethod
    def set_device(cls, device: str | torch.device) -> None:
        """Set the device to use for tensor operations.

        Args:
            device (str | torch.device): Device to use ('cuda', 'cpu', or torch.device object)

        Example:
            >>> ComputeConfig.set_device('cuda')
            >>> ComputeConfig.set_device(torch.device('cpu'))
        """
        if isinstance(device, str):
            cls._device = torch.device(device)
        else:
            cls._device = device

    @classmethod
    def get_dtype(cls) -> torch.dtype:
        """Get the current dtype setting.

        Returns:
            torch.dtype: The current dtype (e.g., torch.float32, torch.float64)
        """
        cls._initialize_defaults()
        return cls._dtype

    @classmethod
    def set_dtype(cls, dtype: torch.dtype) -> None:
        """Set the dtype to use for tensor creation.

        Args:
            dtype (torch.dtype): PyTorch dtype (torch.float32, torch.float64, etc.)

        Example:
            >>> ComputeConfig.set_dtype(torch.float32)
        """
        cls._dtype = dtype

    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset device and dtype to their default values."""
        cls._device = None
        cls._dtype = None
        cls._initialize_defaults()


@dataclass
class Configuration:
    """Class to store model configuration from a PCSE configuration files."""

    CROP: type[SimulationObject]
    CROP_COMPONENTS: dict | None = None
    CROP_NN_MODEL: type[torch.nn.Module] | None = None
    SOIL: type[SimulationObject] | None = None
    AGROMANAGEMENT: type[AncillaryObject] = AgroManager
    OUTPUT_VARS: list = field(default_factory=list)
    SUMMARY_OUTPUT_VARS: list = field(default_factory=list)
    TERMINAL_OUTPUT_VARS: list = field(default_factory=list)
    OUTPUT_INTERVAL: str = "daily"  # "daily"|"dekadal"|"monthly"
    OUTPUT_INTERVAL_DAYS: int = 1
    OUTPUT_WEEKDAY: int = 0
    model_config_file: str | Path | None = None
    description: str | None = None

    def __post_init__(self):
        """Validate config data based on CROP.initialize signature."""
        sig_arguments = inspect.signature(self.CROP.initialize).parameters

        # Nullify CROP_NN_MODEL and CROP_COMPONENTS, if not compatible with CROP.initialize
        for field_value, sig_key, attr_name in [
            (self.CROP_NN_MODEL, "nn_model", "CROP_NN_MODEL"),
            (self.CROP_COMPONENTS, "component_overrides", "CROP_COMPONENTS"),
        ]:
            if field_value is not None and sig_key not in sig_arguments:
                setattr(self, attr_name, None)

        # Validate component overrides have "class" key with non-None value
        for component_name, override in (self.CROP_COMPONENTS or {}).items():
            self._validate_component_override(component_name, override)

    @staticmethod
    def _validate_component_override(component_name: str, override) -> None:
        if not isinstance(override, dict) or not override:
            raise ValueError(f"Component override for '{component_name}' must be a non-empty dict")
        if "class" not in override:
            raise ValueError(f"Component override '{component_name}' must have a 'class' key")
        if override["class"] is None:
            raise ValueError(f"Component override '{component_name}' 'class' cannot be None")

    @classmethod
    def to_dict(cls, config: Self) -> dict:
        """Serialize a Configuration instance to a JSON-compatible dict.

        Args:
            config: Configuration instance to serialize.

        Returns:
            dict: Serializable dict with class references and configuration values.
        """
        cfg = {
            "CROP": class_ref(config.CROP),
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
            "AGROMANAGEMENT": class_ref(config.AGROMANAGEMENT),
            "SOIL": class_ref(config.SOIL) if config.SOIL is not None else None,
        }
        if config.CROP_NN_MODEL is not None:
            cfg["CROP_NN_MODEL"] = class_ref(
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
                entry: dict = {"class": class_ref(override["class"])}
                m = override.get("model")
                if m is not None:
                    entry["model"] = class_ref(m.__class__ if isinstance(m, torch.nn.Module) else m)
                    entry["model_is_instance"] = isinstance(m, torch.nn.Module)
                for k, v in override.items():
                    if k not in ("class", "model") and k not in entry:
                        entry[k] = v
                components[name] = entry
            cfg["CROP_COMPONENTS"] = components
        else:
            cfg["CROP_COMPONENTS"] = None

        return cfg

    @classmethod
    def from_dict(cls, cfg: dict) -> Self:
        """Reconstruct a Configuration instance from a dict created by :meth:`to_dict`.

        Args:
            cfg: Dict produced by :meth:`to_dict` (or loaded from a saved config.json).

        Returns:
            Configuration: A new Configuration instance with classes resolved.
        """
        soil_cls = load_class(cfg["SOIL"]) if cfg.get("SOIL") else None
        crop_nn_model_cls = load_class(cfg["CROP_NN_MODEL"]) if cfg.get("CROP_NN_MODEL") else None

        crop_components = None
        if cfg.get("CROP_COMPONENTS"):
            crop_components = {}
            for name, override in cfg["CROP_COMPONENTS"].items():
                resolved = {"class": load_class(override["class"])}
                if "model" in override:
                    resolved["model"] = load_class(override["model"])
                for k, v in override.items():
                    if k not in ("class", "model", "model_is_instance") and k not in resolved:
                        resolved[k] = v
                crop_components[name] = resolved

        mcf = cfg.get("model_config_file")
        config = Configuration(
            CROP=load_class(cfg["CROP"]),
            CROP_COMPONENTS=crop_components,
            CROP_NN_MODEL=crop_nn_model_cls,
            SOIL=soil_cls,
            AGROMANAGEMENT=load_class(cfg["AGROMANAGEMENT"]),
            OUTPUT_VARS=cfg["OUTPUT_VARS"],
            SUMMARY_OUTPUT_VARS=cfg.get("SUMMARY_OUTPUT_VARS", []),
            TERMINAL_OUTPUT_VARS=cfg.get("TERMINAL_OUTPUT_VARS", []),
            OUTPUT_INTERVAL=cfg.get("OUTPUT_INTERVAL", "daily"),
            OUTPUT_INTERVAL_DAYS=cfg.get("OUTPUT_INTERVAL_DAYS", 1),
            OUTPUT_WEEKDAY=cfg.get("OUTPUT_WEEKDAY", 0),
            model_config_file=Path(mcf) if mcf else None,
            description=cfg.get("description"),
        )
        return config

    @classmethod
    def from_pcse_config_file(cls, filename: str | Path) -> Self:
        """Load the model configuration from a PCSE configuration file.

        Args:
            filename (str | pathlib.Path): Path to the configuraiton file. The path is first
                interpreted with respect to the current working directory and, if not found, it will
                then be interpreted with respect to the `conf` folder in the PCSE package.

        Returns:
            Configuration: Model configuration instance

        Raises:
            FileNotFoundError: if the configuraiton file does not exist
            RuntimeError: if parsing the configuration file fails
        """
        config = {}

        path = Path(filename)
        if path.is_absolute() or path.is_file():
            model_config_file = path
        else:
            pcse_dir = Path(pcse.__path__[0])
            model_config_file = pcse_dir / "conf" / path
        model_config_file = model_config_file.resolve()

        # check that configuration file exists
        if not model_config_file.exists():
            msg = f"PCSE model configuration file does not exist: {model_config_file.name}"
            raise FileNotFoundError(msg)
        # store for later use
        config["model_config_file"] = model_config_file

        # Load file using execfile
        try:
            loc = {}
            bytecode = compile(open(model_config_file).read(), model_config_file, "exec")
            exec(bytecode, {}, loc)
        except Exception as e:
            msg = f"Failed to load configuration from file {model_config_file}"
            raise RuntimeError(msg) from e

        # Add the descriptive header for later use
        if "__doc__" in loc:
            desc = loc.pop("__doc__")
            if len(desc) > 0:
                description = desc
                if description[-1] != "\n":
                    description += "\n"
            config["description"] = description

        # Loop through the attributes in the configuration file
        for key, value in loc.items():
            if key.isupper():
                config[key] = value
        return cls(**config)

    def update_output_variable_lists(
        self,
        output_vars: str | list | tuple | set | None = None,
        summary_vars: str | list | tuple | set | None = None,
        terminal_vars: str | list | tuple | set | None = None,
    ):
        """Updates the lists of output variables that are defined in the configuration file.

        This is useful because sometimes you want the flexibility to get access to an additional
        model variable which is not in the standard list of variables defined in the model
        configuration file. The more elegant way is to define your own configuration file, but this
        adds some flexibility particularly for use in jupyter notebooks and exploratory analysis.

        Note that there is a different behaviour given the type of the variable provided. List and
        string inputs will extend the list of variables, while set/tuple inputs will replace the
        current list.

        Args:
            output_vars: the variable names to add/replace for the OUTPUT_VARS configuration
                variable
            summary_vars: the variable names to add/replace for the SUMMARY_OUTPUT_VARS
                configuration variable
            terminal_vars: the variable names to add/replace for the TERMINAL_OUTPUT_VARS
                configuration variable

        Raises:
            TypeError: if the type of the input arguments is not recognized
        """
        config_varnames = ["OUTPUT_VARS", "SUMMARY_OUTPUT_VARS", "TERMINAL_OUTPUT_VARS"]
        for varitems, config_varname in zip(
            [output_vars, summary_vars, terminal_vars], config_varnames, strict=True
        ):
            if varitems is None:
                continue
            else:
                if isinstance(varitems, str):  # A string: we extend the current list
                    getattr(self, config_varname).extend(varitems.split())
                elif isinstance(varitems, list):  # a list: we extend the current list
                    getattr(self, config_varname).extend(varitems)
                elif isinstance(varitems, tuple | set):  # tuple/set we replace the current list
                    attr = getattr(self, config_varname)
                    attr.clear()
                    attr.extend(list(varitems))
                else:
                    msg = f"Unrecognized input for `output_vars` to engine(): {output_vars}"
                    raise TypeError(msg)
