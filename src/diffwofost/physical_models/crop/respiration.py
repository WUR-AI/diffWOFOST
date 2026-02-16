"""Maintenance respiration for the WOFOST crop model."""

import datetime
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_drv


class WOFOST_Maintenance_Respiration(SimulationObject):
    """Maintenance respiration in WOFOST.

    WOFOST calculates the maintenance respiration as proportional to the dry
    weights of the plant organs to be maintained, where each plant organ can be
    assigned a different maintenance coefficient. Multiplying organ weight
    with the maintenance coeffients yields the relative maintenance respiration
    (`RMRES`) which is than corrected for senescence (parameter `RFSETB`). Finally,
    the actual maintenance respiration rate is calculated using the daily mean
    temperature, assuming a relative increase for each 10 degrees increase
    in temperature as defined by `Q10`.

    **Simulation parameters** (provide in cropdata dictionary)

    | Name   | Description                                               | Type | Unit             |
    |--------|---------------------------------------------------------- |------|------------------|
    | Q10    | Relative increase in maintenance respiration rate with    | SCr  | -                |
    |        | each 10 degrees increase in temperature                   |      | -                |
    | RMR    | Relative maintenance respiration rate for roots           | SCr  | kg CH₂O kg⁻¹ d⁻¹ |
    | RMS    | Relative maintenance respiration rate for stems           | SCr  | kg CH₂O kg⁻¹ d⁻¹ |
    | RML    | Relative maintenance respiration rate for leaves          | SCr  | kg CH₂O kg⁻¹ d⁻¹ |
    | RMO    | Relative maintenance respiration rate for storage organs  | SCr  | kg CH₂O kg⁻¹ d⁻¹ |
    | RFSETB | Reduction factor for senescence                           | TCr  | -                |

    **Rate variables**

    | Name  | Description                                | Pbl | Unit             |
    |-------|--------------------------------------------|----|-------------------|
    | PMRES | Potential maintenance respiration rate     | N  | kg CH₂O ha⁻¹ d⁻¹  |

    **Signals send or handled**

    None

    **External dependencies**

    | Name | Description                         | Provided by                    | Unit      |
    |------|-------------------------------------|--------------------------------|-----------|
    | DVS  | Crop development stage              | DVS_Phenology                  | -         |
    | WRT  | Dry weight of living roots          | WOFOST_Root_Dynamics           | kg ha⁻¹   |
    | WST  | Dry weight of living stems          | WOFOST_Stem_Dynamics           | kg ha⁻¹   |
    | WLV  | Dry weight of living leaves         | WOFOST_Leaf_Dynamics           | kg ha⁻¹   |
    | WSO  | Dry weight of living storage organs | WOFOST_Storage_Organ_Dynamics  | kg ha⁻¹   |

    **Outputs**

    | Name  | Description                                | Pbl | Unit                |
    |-------|--------------------------------------------|----|---------------------|
    | PMRES | Potential maintenance respiration rate     | N  | kg CH₂O ha⁻¹ d⁻¹   |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                |
    |--------|------------------------------------------|
    | PMRES  | Q10, RMR, RML, RMS, RMO, RFSETB          |
    """

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    class Parameters(TensorParamTemplate):
        Q10 = Tensor(-99.0)
        RMR = Tensor(-99.0)
        RML = Tensor(-99.0)
        RMS = Tensor(-99.0)
        RMO = Tensor(-99.0)
        RFSETB = AfgenTrait()

    class RateVariables(TensorRatesTemplate):
        PMRES = Tensor(0.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | None = None,
    ):
        """Initialize the maintenance respiration module.

        Args:
            day: Start date of the simulation
            kiosk: Variable kiosk of this PCSE instance
            parvalues: ParameterProvider object providing parameters as key/value pairs
            shape: Shape of the parameters tensors (optional)
        """
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, shape=shape)
        self.kiosk = kiosk

    @prepare_rates
    def calc_rates(self, day: datetime.date, drv: WeatherDataContainer):
        """Calculate maintenance respiration rates.

        Args:
            day: Current date
            drv: Weather data for the current day
        """
        p = self.params
        kk = self.kiosk
        r = self.rates

        Q10 = p.Q10
        RMR = p.RMR
        RML = p.RML
        RMS = p.RMS
        RMO = p.RMO

        WRT = kk["WRT"]
        WLV = kk["WLV"]
        WST = kk["WST"]
        WSO = kk["WSO"]
        # [!] DVS needs to be broadcasted explicetly because it is used
        # in torch.where and the kiosk does not format it correctly
        # TODO see #22
        DVS = _broadcast_to(kk["DVS"], p.shape, self.dtype, self.device)

        TEMP = _get_drv(drv.TEMP, p.shape, self.dtype, self.device)

        RMRES = RMR * WRT + RML * WLV + RMS * WST + RMO * WSO
        RMRES = RMRES * p.RFSETB(DVS)
        TEFF = Q10 ** ((TEMP - 25.0) / 10.0)
        PMRES = RMRES * TEFF

        # No maintenance respiration before emergence (DVS < 0).
        r.PMRES = torch.where(DVS < 0, torch.zeros_like(PMRES), PMRES)

    def __call__(self, day: datetime.date, drv: WeatherDataContainer):
        """Calculate and return maintenance respiration (PMRES)."""
        self.calc_rates(day, drv)
        return self.rates.PMRES

    def integrate(self, day: datetime.date, delt: float = 1.0):
        """No state variables to integrate for this module."""
        return
