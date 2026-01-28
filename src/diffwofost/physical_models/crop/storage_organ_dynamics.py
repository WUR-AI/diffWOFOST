import datetime
import torch
from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import SimulationObject
from pcse.base import StatesTemplate
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Any
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_params_shape


class WOFOST_Storage_Organ_Dynamics(SimulationObject):
    """Implementation of storage organ dynamics.

    Storage organs are the most simple component of the plant in WOFOST and
    consist of a static pool of biomass. Growth of the storage organs is the
    result of assimilate partitioning. Death of storage organs is not
    implemented and the corresponding rate variable (DRSO) is always set to
    zero.

    Pods are green elements of the plant canopy and can as such contribute
    to the total photosynthetic active area. This is expressed as the Pod
    Area Index which is obtained by multiplying pod biomass with a fixed
    Specific Pod Area (SPA).

    **Simulation parameters**

    | Name | Description                                   | Type   | Unit        |
    |------|===============================================|========|=============|
    | TDWI | Initial total crop dry weight                 | SCr    | kg ha⁻¹     |
    | SPA  | Specific Pod Area                             | SCr    | ha kg⁻¹     |

    **State variables**

    | Name | Description                                      | Pbl  | Unit        |
    |------|==================================================|======|=============|
    | PAI  | Pod Area Index                                   | Y    | -           |
    | WSO  | Weight of living storage organs                  | Y    | kg ha⁻¹     |
    | DWSO | Weight of dead storage organs                    | N    | kg ha⁻¹     |
    | TWSO | Total weight of storage organs                   | Y    | kg ha⁻¹     |

    **Rate variables**

    | Name | Description                                      | Pbl  | Unit        |
    |------|==================================================|======|=============|
    | GRSO | Growth rate storage organs                       | N    | kg ha⁻¹ d⁻¹ |
    | DRSO | Death rate storage organs                        | N    | kg ha⁻¹ d⁻¹ |
    | GWSO | Net change in storage organ biomass              | N    | kg ha⁻¹ d⁻¹ |

    **Signals send or handled**

    None

    **External dependencies**

    | Name | Description                        | Provided by         | Unit        |
    |------|====================================|=====================|=============|
    | ADMI | Above-ground dry matter increase   | CropSimulation      | kg ha⁻¹ d⁻¹ |
    | FO   | Fraction biomass to storage organs | DVS_Partitioning    | -           |
    | FR   | Fraction biomass to roots          | DVS_Partitioning    | -           |

    **Outputs:**

    | Name | Description                  | Provided by | Unit         |
    |------|------------------------------|-------------|--------------|
    | PAI  | Pod Area Index               | Y           | -            |
    | TWSO | Total weight storage organs  | Y           | kg ha⁻¹      |
    | WSO  | Weight living storage organs | Y           | kg ha⁻¹      |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it |
    |--------|----------------------------|
    | PAI    | SPA                       |
    | TWSO   | TDWI                      |
    | WSO    | TDWI                      |
    """

    params_shape = None  # Shape of the parameters tensors

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    class Parameters(ParamTemplate):
        SPA = Any()
        TDWI = Any()

        def __init__(self, parvalues):
            # Get dtype and device from ComputeConfig
            dtype = ComputeConfig.get_dtype()
            device = ComputeConfig.get_device()

            # Set default values
            self.SPA = [torch.tensor(-99.0, dtype=dtype, device=device)]
            self.TDWI = [torch.tensor(-99.0, dtype=dtype, device=device)]

            # Call parent init
            super().__init__(parvalues)

    class StateVariables(StatesTemplate):
        WSO = Any()  # Weight living storage organs
        DWSO = Any()  # Weight dead storage organs
        TWSO = Any()  # Total weight storage organs
        PAI = Any()  # Pod Area Index

        def __init__(self, kiosk, publish=None, **kwargs):
            # Get dtype and device from ComputeConfig
            dtype = ComputeConfig.get_dtype()
            device = ComputeConfig.get_device()

            # Set default values
            if "WSO" not in kwargs:
                self.WSO = [torch.tensor(-99.0, dtype=dtype, device=device)]
            if "DWSO" not in kwargs:
                self.DWSO = [torch.tensor(-99.0, dtype=dtype, device=device)]
            if "TWSO" not in kwargs:
                self.TWSO = [torch.tensor(-99.0, dtype=dtype, device=device)]
            if "PAI" not in kwargs:
                self.PAI = [torch.tensor(-99.0, dtype=dtype, device=device)]

            # Call parent init
            super().__init__(kiosk, publish=publish, **kwargs)

    class RateVariables(RatesTemplate):
        GRSO = Any()
        DRSO = Any()
        GWSO = Any()

        def __init__(self, kiosk, publish=None):
            # Get dtype and device from ComputeConfig
            dtype = ComputeConfig.get_dtype()
            device = ComputeConfig.get_device()

            # Set default values
            self.GRSO = torch.tensor(0.0, dtype=dtype, device=device)
            self.DRSO = torch.tensor(0.0, dtype=dtype, device=device)
            self.GWSO = torch.tensor(0.0, dtype=dtype, device=device)

            # Call parent init
            super().__init__(kiosk, publish=publish)

    def initialize(
        self, day: datetime.date, kiosk: VariableKiosk, parvalues: ParameterProvider
    ) -> None:
        """Initialize the storage organ dynamics model.

        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["GRSO"])

        # INITIAL STATES
        params = self.params
        self.params_shape = _get_params_shape(params)
        shape = self.params_shape

        # Initial storage organ biomass
        TDWI = _broadcast_to(params.TDWI, shape, dtype=self.dtype, device=self.device)
        SPA = _broadcast_to(params.SPA, shape, dtype=self.dtype, device=self.device)
        FO = _broadcast_to(self.kiosk["FO"], shape, dtype=self.dtype, device=self.device)
        FR = _broadcast_to(self.kiosk["FR"], shape, dtype=self.dtype, device=self.device)

        WSO = (TDWI * (1 - FR)) * FO
        DWSO = torch.zeros(shape, dtype=self.dtype, device=self.device)
        TWSO = WSO + DWSO
        # Initial Pod Area Index
        PAI = WSO * SPA

        self.states = self.StateVariables(
            kiosk, publish=["TWSO", "WSO", "PAI"], WSO=WSO, DWSO=DWSO, TWSO=TWSO, PAI=PAI
        )

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None) -> None:
        """Calculate the rates of change of the state variables.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            drv (WeatherDataContainer, optional): A dictionary-like container holding
                weather data elements as key/value.
        """
        rates = self.rates
        k = self.kiosk

        FO = _broadcast_to(k["FO"], self.params_shape, dtype=self.dtype, device=self.device)
        ADMI = _broadcast_to(k["ADMI"], self.params_shape, dtype=self.dtype, device=self.device)
        REALLOC_SO = _broadcast_to(
            k.get("REALLOC_SO", 0.0), self.params_shape, dtype=self.dtype, device=self.device
        )

        # Growth/death rate organs
        rates.GRSO = ADMI * FO
        rates.DRSO = torch.zeros(self.params_shape, dtype=self.dtype, device=self.device)
        rates.GWSO = rates.GRSO - rates.DRSO + REALLOC_SO

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """Integrate the state variables.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        params = self.params
        rates = self.rates
        states = self.states

        SPA = _broadcast_to(params.SPA, self.params_shape, dtype=self.dtype, device=self.device)

        # Stem biomass (living, dead, total)
        states.WSO = states.WSO + rates.GWSO
        states.DWSO = states.DWSO + rates.DRSO
        states.TWSO = states.WSO + states.DWSO

        # Calculate Pod Area Index (SAI)
        states.PAI = states.WSO * SPA
