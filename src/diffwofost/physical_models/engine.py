from pathlib import Path
from pcse import signals
from pcse.base import BaseEngine
from pcse.base.variablekiosk import VariableKiosk
from pcse.engine import Engine
from pcse.timer import Timer
from pcse.traitlets import Instance
from .config import Configuration


class Engine(Engine):
    mconf = Instance(Configuration)

    def __init__(
        self,
        parameterprovider,
        weatherdataprovider,
        agromanagement,
        config: str | Path | Configuration,
    ):
        BaseEngine.__init__(self)

        # If a path is given, load the model configuration from a PCSE config file
        if isinstance(config, str | Path):
            self.mconf = Configuration.from_pcse_config_file(config)
        else:
            self.mconf = config

        self.parameterprovider = parameterprovider

        # Variable kiosk for registering and publishing variables
        self.kiosk = VariableKiosk()

        # Placeholder for variables to be saved during a model run
        self._saved_output = []
        self._saved_summary_output = []
        self._saved_terminal_output = {}

        # register handlers for starting/finishing the crop simulation, for
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
