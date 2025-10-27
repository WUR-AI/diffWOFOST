"""This file contains code that is required to run the YAML unit tests.

It contains:
    - SimulationObjectTestHelper: a simobj that wraps the simulation object to be tested.
    - VariableKioskTestHelper: A subclass of the VariableKiosk that can use externally
      forced states/rates
    - ConfigurationLoaderTestHelper: An subclass of ConfigurationLoader that allows to
      specify the simbojects to be test dynamically
    - EngineTestHelper: engine specifically for running the YAML tests.
    - WeatherDataProviderTestHelper: a weatherdata provides that takes the weather
      inputs from the YAML file.

Note that the code here is *not* python2 compatible.
"""

import logging
import os
import torch
import yaml
from pcse import signals
from pcse.agromanager import AgroManager
from pcse.base import ConfigurationLoader
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.simulationobject import SimulationObject
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.base.weather import WeatherDataProvider
from pcse.engine import BaseEngine
from pcse.engine import Engine
from pcse.timer import Timer
from pcse.traitlets import Instance

logging.disable(logging.CRITICAL)

this_dir = os.path.dirname(__file__)


def nothing(*args, **kwargs):
    """A function that does nothing."""
    pass


class SimulationObjectTestHelper(SimulationObject):
    """This wraps the SimulationObject for testing.

    This ensuree that the computations are not carried out before crop emergence
    (e.g. DVS >= 0). The latter does not apply for the phenology simobject
    itself which simulates emergence. The phenology simobject is recognized
    because the variable DVS is not an external variable.
    """

    test_class = None
    subsimobject = Instance(SimulationObject)

    def initialize(self, day, kiosk, parvalues):
        """Initialize the subsimobject."""
        self.subsimobject = self.test_class(day, kiosk, parvalues)

    def calc_rates(self, day, drv):
        """Calculate the rates of the subsimobject."""
        # some simobject do not provide a `calc_rates()` function but are directly callable
        # here we check for those cases.
        func = self.subsimobject if callable(self.subsimobject) else self.subsimobject.calc_rates
        if not self.kiosk.is_external_state("DVS"):
            func(day, drv)
        else:
            if self.kiosk.DVS >= 0:
                func(day, drv)
            else:
                self.subsimobject.zerofy()

    def integrate(self, day, delt=1.0):
        """Integrate the states of the subsimobject."""
        # If the simobject is callable, we do not need integration so we use the
        # `nothing()` function.
        func = nothing if callable(self.subsimobject) else self.subsimobject.integrate
        if not self.kiosk.is_external_state("DVS"):
            func(day, delt)
        else:
            if self.kiosk.DVS >= 0:
                func(day, delt)
            else:
                self.subsimobject.touch()


class VariableKioskTestHelper(VariableKiosk):
    """Variable Kiosk for testing purposes which allows to use external states."""

    external_state_list = None

    def __init__(self, external_state_list):
        super().__init__()
        self.current_externals = {}
        if external_state_list is not None:
            self.external_state_list = external_state_list

    def __call__(self, day):
        """Sets the external state/rate variables for the current day.

        Returns True if the list of external state/rate variables is exhausted,
        otherwise False.
        """
        if self.external_state_list is not None:
            current_externals = self.external_state_list.pop(0)
            forcing_day = current_externals.pop("DAY")
            msg = "Failure updating VariableKiosk with external states: days are not matching!"
            assert forcing_day == day, msg
            self.current_externals.clear()
            self.current_externals.update(current_externals)
            if len(self.external_state_list) == 0:
                return True

        return False

    def is_external_state(self, item):
        """Returns True if the item is an external state."""
        return item in self.current_externals

    def __getattr__(self, item):
        """Allow use of attribute notation.

        eg "kiosk.LAI" on published rates or states.
        """
        if item in self.current_externals:
            return self.current_externals[item]
        else:
            return dict.__getitem__(self, item)

    def __getitem__(self, item):
        """Override __getitem__ to first look in external states."""
        if item in self.current_externals:
            return self.current_externals[item]
        else:
            return dict.__getitem__(self, item)

    def __contains__(self, key):
        """Override __contains__ to first look in external states."""
        return key in self.current_externals or dict.__contains__(self, key)


class ConfigurationLoaderTestHelper(ConfigurationLoader):
    def __init__(self, YAML_test_inputs, simobject, waterbalance=None):
        self.model_config_file = "Test config"
        self.description = "Configuration loader for running YAML tests"
        self.CROP = simobject
        self.SOIL = waterbalance
        self.AGROMANAGEMENT = AgroManager
        self.OUTPUT_INTERVAL = "daily"
        self.OUTPUT_INTERVAL_DAYS = 1
        self.OUTPUT_WEEKDAY = 0
        self.OUTPUT_VARS = list(YAML_test_inputs["Precision"].keys())
        self.SUMMARY_OUTPUT_VARS = []
        self.TERMINAL_OUTPUT_VARS = []


class EngineTestHelper(Engine):
    """An engine which is purely for running the YAML unit tests."""

    def __init__(
        self,
        parameterprovider,
        weatherdataprovider,
        agromanagement,
        test_config,
        external_states=None,
    ):
        BaseEngine.__init__(self)

        # Load the model configuration
        self.mconf = ConfigurationLoader(test_config)
        self.parameterprovider = parameterprovider

        # Variable kiosk for registering and publishing variables
        self.kiosk = VariableKioskTestHelper(external_states)

        # Placeholder for variables to be saved during a model run
        self._saved_output = list()
        self._saved_summary_output = list()
        self._saved_terminal_output = dict()

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
        self.day, delt = self.timer()
        # Update external states in the kiosk
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

    def _run(self):
        """Make one time step of the simulation."""
        # Update timer
        self.day, delt = self.timer()

        # When the list of external states is exhausted the VariableKioskTestHelper will
        # return True signalling the end of the test
        stop_test = self.kiosk(self.day)
        if stop_test:
            self._send_signal(
                signal=signals.crop_finish, day=self.day, finish_type="maturity", crop_delete=False
            )

        # State integration and update to forced variables
        self.integrate(self.day, delt)

        # Driving variables
        self.drv = self._get_driving_variables(self.day)

        # Agromanagement decisions
        self.agromanager(self.day, self.drv)

        # Rate calculation
        self.calc_rates(self.day, self.drv)

        if self.flag_terminate is True:
            self._terminate_simulation(self.day)


class WeatherDataProviderTestHelper(WeatherDataProvider):
    """It stores the weatherdata contained within the YAML tests."""

    def __init__(self, yaml_weather):
        super().__init__()
        for weather in yaml_weather:
            if "SNOWDEPTH" in weather:
                weather.pop("SNOWDEPTH")
            wdc = WeatherDataContainer(**weather)
            self._store_WeatherDataContainer(wdc, wdc.DAY)


def prepare_engine_input(file_path, crop_model_params):
    """Prepare the inputs for the engine from the YAML file."""
    inputs = yaml.safe_load(open(file_path))
    agro_management_inputs = inputs["AgroManagement"]
    cropd = inputs["ModelParameters"]

    weather_data_provider = WeatherDataProviderTestHelper(inputs["WeatherVariables"])
    crop_model_params_provider = ParameterProvider(cropdata=cropd)
    external_states = inputs["ExternalStates"]

    # convert parameters to tensors
    crop_model_params_provider.clear_override()
    for name in crop_model_params:
        value = torch.tensor(crop_model_params_provider[name], dtype=torch.float32)
        crop_model_params_provider.set_override(name, value, check=False)

    # convert external states to tensors
    tensor_external_states = [
        {k: v if k == "DAY" else torch.tensor(v, dtype=torch.float32) for k, v in item.items()}
        for item in external_states
    ]
    return (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        tensor_external_states,
    )


def get_test_data(file_path):
    """Get the test data from the YAML file."""
    inputs = yaml.safe_load(open(file_path))
    return inputs["ModelResults"], inputs["Precision"]


def calculate_numerical_grad(get_model_fn, param_name, param_value, out_name):
    """Calculate the numerical gradient of output with respect to a parameter."""
    delta = 1e-6
    p_plus = param_value + delta
    p_minus = param_value - delta

    model = get_model_fn()
    output = model({param_name: torch.nn.Parameter(p_plus)})
    loss_plus = output[out_name].sum(dim=0)

    model = get_model_fn()
    output = model({param_name: torch.nn.Parameter(p_minus)})
    loss_minus = output[out_name].sum(dim=0)

    return (loss_plus.data - loss_minus.data) / (2 * delta)
