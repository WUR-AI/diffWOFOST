# Copyright (c) 2004-2024 Wageningen Environmental Research, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl) and Herman Berghuijs (herman.berghuijs@wur.nl), April 2024

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
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_params_shape


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
        RDRSTB = AfgenTrait()
        SSATB = AfgenTrait()
        TDWI = Any()

        def __init__(self, parvalues):
            # Get dtype and device from ComputeConfig
            dtype = ComputeConfig.get_dtype()
            device = ComputeConfig.get_device()

            # Set default values
            self.TDWI = [torch.tensor(-99.0, dtype=dtype, device=device)]

            # Call parent init
            super().__init__(parvalues)

    class StateVariables(StatesTemplate):
        WST = Any()
        DWST = Any()
        TWST = Any()
        SAI = Any()  # Stem Area Index

        def __init__(self, kiosk, publish=None, **kwargs):
            # Get dtype and device from ComputeConfig
            dtype = ComputeConfig.get_dtype()
            device = ComputeConfig.get_device()

            # Set default values
            if "WST" not in kwargs:
                self.WST = [torch.tensor(-99.0, dtype=dtype, device=device)]
            if "DWST" not in kwargs:
                self.DWST = [torch.tensor(-99.0, dtype=dtype, device=device)]
            if "TWST" not in kwargs:
                self.TWST = [torch.tensor(-99.0, dtype=dtype, device=device)]
            if "SAI" not in kwargs:
                self.SAI = torch.tensor(-99.0, dtype=dtype, device=device)

            # Call parent init
            super().__init__(kiosk, publish=publish, **kwargs)

    class RateVariables(RatesTemplate):
        GRST = Any()
        DRST = Any()
        GWST = Any()

        def __init__(self, kiosk, publish=None):
            # Get dtype and device from ComputeConfig
            dtype = ComputeConfig.get_dtype()
            device = ComputeConfig.get_device()

            # Set default values
            self.GRST = torch.tensor(0.0, dtype=dtype, device=device)
            self.DRST = torch.tensor(0.0, dtype=dtype, device=device)
            self.GWST = torch.tensor(0.0, dtype=dtype, device=device)

            # Call parent init
            super().__init__(kiosk, publish=publish)

    def initialize(
        self, day: datetime.date, kiosk: VariableKiosk, parvalues: ParameterProvider
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
        """
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["DRST", "GRST"])

        # INITIAL STATES
        params = self.params
        self.params_shape = _get_params_shape(params)
        shape = self.params_shape

        # Set initial stem biomass
        TDWI = _broadcast_to(params.TDWI, shape, dtype=self.dtype, device=self.device)
        FS = _broadcast_to(self.kiosk["FS"], shape, dtype=self.dtype, device=self.device)
        FR = _broadcast_to(self.kiosk["FR"], shape, dtype=self.dtype, device=self.device)
        WST = (TDWI * (1 - FR)) * FS
        DWST = torch.zeros(shape, dtype=self.dtype, device=self.device)
        TWST = WST + DWST

        # Initial Stem Area Index
        DVS = _broadcast_to(self.kiosk["DVS"], shape, dtype=self.dtype, device=self.device)
        SSATB = params.SSATB.to(device=self.device, dtype=self.dtype)
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
        DVS = _broadcast_to(k["DVS"], self.params_shape, dtype=self.dtype, device=self.device)
        dvs_mask = (DVS >= 0).to(dtype=self.dtype)

        FS = _broadcast_to(k["FS"], self.params_shape, dtype=self.dtype, device=self.device)
        ADMI = _broadcast_to(k["ADMI"], self.params_shape, dtype=self.dtype, device=self.device)
        RDRSTB = p.RDRSTB.to(device=self.device, dtype=self.dtype)

        # Growth/death rate stems
        r.GRST = dvs_mask * ADMI * FS
        r.DRST = dvs_mask * s.WST * RDRSTB(DVS)

        # Check if REALLOC_ST exists in kiosk, if not use 0
        REALLOC_ST = torch.zeros_like(r.GRST)
        if "REALLOC_ST" in k:
            REALLOC_ST = _broadcast_to(
                k["REALLOC_ST"], self.params_shape, dtype=self.dtype, device=self.device
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
        DVS = _broadcast_to(
            self.kiosk["DVS"], self.params_shape, dtype=self.dtype, device=self.device
        )
        SSATB = p.SSATB.to(device=self.device, dtype=self.dtype)
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

        oWST = s.WST.clone() if isinstance(s.WST, torch.Tensor) else s.WST
        oTWST = s.TWST.clone() if isinstance(s.TWST, torch.Tensor) else s.TWST
        oSAI = s.SAI.clone() if isinstance(s.SAI, torch.Tensor) else s.SAI

        s.WST = nWST
        s.TWST = s.DWST + nWST

        DVS = _broadcast_to(k["DVS"], self.params_shape, dtype=self.dtype, device=self.device)
        SSATB = p.SSATB.to(device=self.device, dtype=self.dtype)
        s.SAI = s.WST * SSATB(DVS)

        increments = {"WST": s.WST - oWST, "SAI": s.SAI - oSAI, "TWST": s.TWST - oTWST}
        return increments
