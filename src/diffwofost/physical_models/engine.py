import gc
from pathlib import Path
import torch
from pcse import signals
from pcse.base import BaseEngine
from pcse.engine import Engine as PcseEngine
from pcse.timer import Timer
from pcse.traitlets import Instance
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.variablekiosk import VariableKiosk


class Engine(PcseEngine):
    mconf = Instance(Configuration)

    def __init__(
        self,
        config: str | Path | Configuration | None = None,
    ):
        BaseEngine.__init__(self)

        if config is None:
            msg = "A model configuration must be provided when initializing the engine."
            raise TypeError(msg)

        # If a path is given, load the model configuration from a PCSE config file
        if isinstance(config, str | Path):
            self.mconf = Configuration.from_pcse_config_file(config)
        else:
            self.mconf = config


    def _reset_runtime_state(self):
        for component_name in ("crop", "soil"):
            component = getattr(self, component_name, None)
            if component is not None:
                component._delete()
                setattr(self, component_name, None)

        gc.collect()

        self.flag_terminate = False
        self.flag_crop_finish = False
        self.flag_crop_start = False
        self.flag_crop_delete = False
        self.flag_output = False
        self.flag_summary_output = False

        self._saved_output = []
        self._saved_summary_output = []
        self._saved_terminal_output = {}

    def setup(
        self,
        parameterprovider,
        weatherdataprovider,
        agromanagement,
        external_states=None,
    ):
        """Set up the engine for a new simulation run."""
        if external_states is None:
            external_states = self._default_external_states
        else:
            self._default_external_states = external_states

        self._reset_runtime_state()

        self.parameterprovider = parameterprovider
        self._shape = _get_params_shape(self.parameterprovider)

        # Variable kiosk for registering and publishing variables
        self.kiosk = VariableKiosk(external_states)

        # Register handlers for starting/finishing the crop simulation, for
        # handling output and terminating the system
        self._connect_signal(self._on_CROP_START, signal=signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)
        self._connect_signal(self._on_OUTPUT, signal=signals.output)
        self._connect_signal(self._on_TERMINATE, signal=signals.terminate)

        # Component for agromanagement
        self.agromanager = self.mconf.AGROMANAGEMENT(self.kiosk, agromanagement)
        start_date = self.agromanager.start_date
        end_date = self.agromanager.end_date

        # Timer: starting day, final day and model output
        self.timer = Timer(self.kiosk, start_date, end_date, self.mconf)
        self.day, _ = self.timer()
        self.kiosk(self.day)

        # Driving variables
        self.weatherdataprovider = weatherdataprovider
        self.drv = self._get_driving_variables(self.day)

        # Component for simulation of soil processes
        if self.mconf.SOIL is not None:
            self.soil = self.mconf.SOIL(self.day, self.kiosk, parameterprovider)

        # Call AgroManagement module for management actions at initialization
        self.agromanager(self.day, self.drv)

        # Calculate initial rates
        self.calc_rates(self.day, self.drv)
        return self

    def _on_CROP_START(
        self, day, crop_name=None, variety_name=None, crop_start_type=None, crop_end_type=None
    ):
        """Starts the crop."""
        self.logger.debug(f"Received signal 'CROP_START' on day {day}")

        if self.crop is not None:
            raise RuntimeError(
                "A CROP_START signal was received while self.cropsimulation still holds a valid "
                "cropsimulation object. It looks like you forgot to send a CROP_FINISH signal with "
                "option crop_delete=True"
            )

        self.parameterprovider.set_active_crop(
            crop_name, variety_name, crop_start_type, crop_end_type
        )
        self.crop = self.mconf.CROP(day, self.kiosk, self.parameterprovider, shape=self._shape)

    def _finish_cropsimulation(self, day):
        self.flag_crop_finish = False

        self.crop.finalize(day)
        self._save_summary_output()

        if self.flag_crop_delete:
            self.flag_crop_delete = False
            self.crop._delete()
            self.crop = None
            gc.collect()


def _get_params_shape(parameterprovider):
    shape = ()
    for paramname in parameterprovider._unique_parameters:
        param = parameterprovider[paramname]
        if isinstance(param, torch.Tensor):
            # We need to drop the last dimension from the Afgen table parameters
            param_shape = param.shape[:-1] if paramname.endswith("TB") else param.shape
            if not param_shape or shape == param_shape:
                continue
            elif param_shape and not shape:
                shape = tuple(param_shape)
            else:
                raise ValueError("Non-matching shapes found in parameter provider!")
    return shape
