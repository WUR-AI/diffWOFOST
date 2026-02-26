import datetime
import torch
from pcse.base import RatesTemplate
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor


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

    class Parameters(TensorParamTemplate):
        SPA = Tensor(-99.0)
        TDWI = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        WSO = Tensor(-99.0)  # Weight living storage organs
        DWSO = Tensor(-99.0)  # Weight dead storage organs
        TWSO = Tensor(-99.0)  # Total weight storage organs
        PAI = Tensor(-99.0)  # Pod Area Index

    class RateVariables(RatesTemplate):
        GRSO = Tensor(0.0)
        DRSO = Tensor(0.0)
        GWSO = Tensor(0.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | torch.Size | None = None,
    ) -> None:
        """Initialize the storage organ dynamics model."""
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, publish=["GRSO"])

        self._drso_zeros = torch.zeros(self.params.shape, dtype=self.dtype, device=self.device)

        # Initial storage organ biomass
        TDWI = self.params.TDWI
        SPA = self.params.SPA
        FO = self.kiosk["FO"]
        FR = self.kiosk["FR"]

        WSO = (TDWI * (1 - FR)) * FO
        DWSO = self._drso_zeros
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

        FO = k["FO"]
        ADMI = k["ADMI"]
        REALLOC_SO = k.get("REALLOC_SO", self._drso_zeros)

        # Growth/death rate organs
        rates.GRSO = ADMI * FO
        rates.DRSO = self._drso_zeros
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

        SPA = params.SPA

        # Stem biomass (living, dead, total)
        states.WSO = states.WSO + rates.GWSO
        states.DWSO = states.DWSO + rates.DRSO
        states.TWSO = states.WSO + states.DWSO

        # Calculate Pod Area Index (SAI)
        states.PAI = states.WSO * SPA
