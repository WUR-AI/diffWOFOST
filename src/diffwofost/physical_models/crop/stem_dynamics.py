import datetime
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to


class WOFOST_Stem_Dynamics(SimulationObject):
    """Implementation of stem biomass dynamics.

    Stem biomass increase results from the assimilates partitioned to
    the stem system. Stem death is defined as the current stem biomass
    multiplied by a relative death rate (`RDRSTB`). The latter as a function
    of the development stage (`DVS`).

    Stems are green elements of the plant canopy and can as such contribute
    to the total photosynthetic active area. This is expressed as the Stem
    Area Index which is obtained by multiplying stem biomass with the
    Specific Stem Area (SSATB), which is a function of DVS.

    **Simulation parameters**:

    =======  ============================================= =======  ============
     Name     Description                                   Type     Unit
    =======  ============================================= =======  ============
    TDWI     Initial total crop dry weight                  SCr       |kg ha-1|
    RDRSTB   Relative death rate of stems as a function     TCr       -
             of development stage
    SSATB    Specific Stem Area as a function of            TCr       |ha kg-1|
             development stage
    =======  ============================================= =======  ============


    **State variables**

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    SAI      Stem Area Index                                    Y     -
    WST      Weight of living stems                             Y     |kg ha-1|
    DWST     Weight of dead stems                               N     |kg ha-1|
    TWST     Total weight of stems                              Y     |kg ha-1|
    =======  ================================================= ==== ============

    **Rate variables**

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    GRST     Growth rate stem biomass                           N   |kg ha-1 d-1|
    DRST     Death rate stem biomass                            N   |kg ha-1 d-1|
    GWST     Net change in stem biomass                         N   |kg ha-1 d-1|
    =======  ================================================= ==== ============

    **Signals send or handled**

    None

    **External dependencies:**

    =======  =================================== =================  ============
     Name     Description                         Provided by         Unit
    =======  =================================== =================  ============
    DVS      Crop development stage              DVS_Phenology       -
    ADMI     Above-ground dry matter             CropSimulation     |kg ha-1 d-1|
             increase
    FR       Fraction biomass to roots           DVS_Partitioning    -
    FS       Fraction biomass to stems           DVS_Partitioning    -
    =======  =================================== =================  ============

    **Outputs:**

    | Name  | Description                     | Unit       |
    |-------|---------------------------------|------------|
    | WST   | Weight of living stems          | |kg ha-1|
    | SAI   | Stem Area Index                 | -          |
    | TWST  | Total weight of stems           | |kg ha-1|

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it |
    |--------|---------------------------|
    | WST    | TDWI, RDRSTB              |
    | SAI    | TDWI, SSATB               |
    | TWST   | TDWI, RDRSTB              |

    """  # noqa: E501

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    class Parameters(TensorParamTemplate):
        RDRSTB = AfgenTrait()
        SSATB = AfgenTrait()
        TDWI = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        WST = Tensor(-99.0)
        DWST = Tensor(-99.0)
        TWST = Tensor(-99.0)
        SAI = Tensor(-99.0)  # Stem Area Index

    class RateVariables(TensorRatesTemplate):
        GRST = Tensor(0.0)
        DRST = Tensor(0.0)
        GWST = Tensor(0.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | torch.Size | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            day (datetime.date): The starting date of the simulation.
            kiosk (VariableKiosk): A container for registering and publishing
                (internal and external) state variables. See PCSE documentation for
                details.
            parvalues (ParameterProvider): A dictionary-like container holding
                all parameter sets (crop, soil, site) as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
            shape (tuple | torch.Size | None): Target shape for the state and rate variables.
        """
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["DRST", "GRST"])

        # INITIAL STATES
        params = self.params
        shape = params.shape

        # Set initial stem biomass
        TDWI = params.TDWI
        FS = self.kiosk["FS"]
        FR = self.kiosk["FR"]
        WST = (TDWI * (1 - FR)) * FS
        DWST = torch.zeros(shape, dtype=self.dtype, device=self.device)
        TWST = WST + DWST

        # Initial Stem Area Index
        DVS = self.kiosk["DVS"]
        SSATB = params.SSATB
        SAI = WST * SSATB(DVS)

        self.states = self.StateVariables(
            kiosk, publish=["TWST", "WST", "SAI"], WST=WST, DWST=DWST, TWST=TWST, SAI=SAI
        )

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None) -> None:
        """Calculate the rates of change of the state variables.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            drv (WeatherDataContainer, optional): A dictionary-like container holding
                weather data elements as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
        """
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        # If DVS < 0, the crop has not yet emerged, so we zerofy the rates using mask.
        # Make a mask (0 if DVS < 0, 1 if DVS >= 0)
        DVS = k["DVS"]
        dvs_mask = DVS >= 0

        FS = k["FS"]
        ADMI = k["ADMI"]
        RDRSTB = p.RDRSTB

        # Growth/death rate stems
        r.GRST = dvs_mask * ADMI * FS
        r.DRST = dvs_mask * s.WST * RDRSTB(DVS)

        # Check if REALLOC_ST exists in kiosk, if not use 0
        REALLOC_ST = torch.zeros_like(r.GRST)
        if "REALLOC_ST" in k:
            REALLOC_ST = _broadcast_to(
                k["REALLOC_ST"], p.shape, dtype=self.dtype, device=self.device
            )

        r.GWST = r.GRST - r.DRST - REALLOC_ST

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """Integrate the state variables using the rates of change.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        p = self.params
        r = self.rates
        s = self.states

        # Stem biomass (living, dead, total)
        s.WST = s.WST + r.GWST
        s.DWST = s.DWST + r.DRST
        s.TWST = s.WST + s.DWST

        # Calculate Stem Area Index (SAI)
        DVS = self.kiosk["DVS"]
        SSATB = p.SSATB
        s.SAI = s.WST * SSATB(DVS)

    @prepare_states
    def _set_variable_WST(self, nWST):
        """Set the WST state variable and update dependent variables.

        Args:
            nWST: New value for WST (stem biomass).

        Returns:
            dict: Dictionary with increments for WST, SAI, and TWST.
        """
        s = self.states
        p = self.params
        k = self.kiosk

        oWST = s.WST.clone()
        oTWST = s.TWST.clone()
        oSAI = s.SAI.clone()

        s.WST = nWST
        s.TWST = s.DWST + nWST

        DVS = k["DVS"]
        SSATB = p.SSATB
        s.SAI = s.WST * SSATB(DVS)

        increments = {"WST": s.WST - oWST, "SAI": s.SAI - oSAI, "TWST": s.TWST - oTWST}
        return increments
