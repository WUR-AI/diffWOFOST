from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Self
import pcse
from pcse.agromanager import AgroManager
from pcse.base import AncillaryObject
from pcse.base import SimulationObject


@dataclass(frozen=True)
class Configuration:
    """Class to store model configuration from a PCSE configuration files."""

    CROP: type[SimulationObject]
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
