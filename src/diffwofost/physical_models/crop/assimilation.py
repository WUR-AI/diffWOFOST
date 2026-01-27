"""SimulationObjects implementing |CO2| Assimilation for use with PCSE."""

import datetime
from collections import deque
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.util import astro
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_drv
from diffwofost.physical_models.utils import _get_params_shape


def _as_python_float(x) -> float:
    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().cpu()
        if x_cpu.numel() != 1:
            x_cpu = x_cpu.reshape(-1)[0]
        return float(x_cpu.item())
    return float(x)


def totass7(
    DAYL: torch.Tensor,
    AMAX: torch.Tensor,
    EFF: torch.Tensor,
    LAI: torch.Tensor,
    KDIF: torch.Tensor,
    AVRAD: torch.Tensor,
    DIFPP: torch.Tensor,
    DSINBE: torch.Tensor,
    SINLD: torch.Tensor,
    COSLD: torch.Tensor,
    *,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """Calculates daily total gross CO2 assimilation.

    This routine calculates the daily total gross CO2 assimilation
    by performing a Gaussian integration over time.
    At three different times of
    the day, irradiance is computed and used to calculate the instantaneous
    canopy assimilation, whereafter integration takes place. More information
    on this routine is given by Spitters et al. (1988).
    FORMAL PARAMETERS:  (I=input,O=output,C=control,IN=init,T=time)
    name   type meaning                                    units  class
    ----   ---- -------                                    -----  -----
    DAYL    R4  Astronomical daylength (base = 0 degrees)     h      I
    AMAX    R4  Assimilation rate at light saturation      kg CO2/   I
                                                          ha leaf/h
    EFF     R4  Initial light use efficiency              kg CO2/J/  I
                                                          ha/h m2 s
    LAI     R4  Leaf area index                             ha/ha    I
    KDIF    R4  Extinction coefficient for diffuse light             I
    AVRAD   R4  Daily shortwave radiation                  J m-2 d-1 I
    DIFPP   R4  Diffuse irradiation perpendicular to direction of
                light                                      J m-2 s-1 I
    DSINBE  R4  Daily total of effective solar height         s      I
    SINLD   R4  Seasonal offset of sine of solar height       -      I
    COSLD   R4  Amplitude of sine of solar height             -      I
    DTGA    R4  Daily total gross assimilation           kg CO2/ha/d O
    """
    xgauss = torch.tensor([0.1127017, 0.5000000, 0.8872983], dtype=DAYL.dtype, device=DAYL.device)
    wgauss = torch.tensor([0.2777778, 0.4444444, 0.2777778], dtype=DAYL.dtype, device=DAYL.device)
    pi = torch.tensor(torch.pi, dtype=DAYL.dtype, device=DAYL.device)

    # Only compute where it can be non-zero.
    mask = (AMAX > 0) & (LAI > 0) & (DAYL > 0)

    dtga = torch.zeros_like(AMAX)
    # Prevent division by zero in par calculation
    dsinbe_safe = torch.where(DSINBE > epsilon, DSINBE, torch.ones_like(DSINBE))

    for i in range(3):
        hour = 12.0 + 0.5 * DAYL * xgauss[i]
        sinb = torch.maximum(
            torch.zeros_like(DAYL),
            SINLD + COSLD * torch.cos(2.0 * pi * (hour + 12.0) / 24.0),
        )

        par = 0.5 * AVRAD * sinb * (1.0 + 0.4 * sinb) / dsinbe_safe
        pardif = torch.minimum(par, sinb * DIFPP)
        pardir = par - pardif

        fgros = assim7(AMAX, EFF, LAI, KDIF, sinb, pardir, pardif, epsilon=epsilon)
        dtga = dtga + fgros * wgauss[i]

    dtga = dtga * DAYL
    return torch.where(mask, dtga, torch.zeros_like(dtga))


def assim7(
    AMAX: torch.Tensor,
    EFF: torch.Tensor,
    LAI: torch.Tensor,
    KDIF: torch.Tensor,
    SINB: torch.Tensor,
    PARDIR: torch.Tensor,
    PARDIF: torch.Tensor,
    *,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """This routine calculates the gross CO2 assimilation rate of the whole crop.

    FGROS is calculated by performing a Gaussian integration
    over depth in the crop canopy. At three different depths in
    the canopy, i.e. for different values of LAI, the
    assimilation rate is computed for given fluxes of photosynthe-
    tically active radiation, whereafter integration over depth
    takes place. More information on this routine is given by
    Spitters et al. (1988). The input variables SINB, PARDIR
    and PARDIF are calculated in routine TOTASS.
    Subroutines and functions called: none.
    Called by routine TOTASS.
    """
    xgauss = torch.tensor([0.1127017, 0.5000000, 0.8872983], dtype=AMAX.dtype, device=AMAX.device)
    wgauss = torch.tensor([0.2777778, 0.4444444, 0.2777778], dtype=AMAX.dtype, device=AMAX.device)

    scv = torch.tensor(0.2, dtype=AMAX.dtype, device=AMAX.device)
    one = torch.tensor(1.0, dtype=AMAX.dtype, device=AMAX.device)

    # Prevent division by zero in extinction coefficient calculations
    sinb_safe = torch.where(SINB > epsilon, SINB, torch.ones_like(SINB))

    # Extinction coefficients
    refh = (one - torch.sqrt(one - scv)) / (one + torch.sqrt(one - scv))
    refs = refh * 2.0 / (one + 1.6 * sinb_safe)
    kdirbl = (0.5 / sinb_safe) * KDIF / (0.8 * torch.sqrt(one - scv))
    kdir_t = kdirbl * torch.sqrt(one - scv)

    # Integration over LAI (depth)
    fgros = torch.zeros_like(AMAX)
    amax_denom = torch.maximum(torch.tensor(2.0, dtype=AMAX.dtype, device=AMAX.device), AMAX)

    for i in range(3):
        laic = LAI * xgauss[i]

        visdf = (one - refs) * PARDIF * KDIF * torch.exp(-KDIF * laic)
        vist = (one - refs) * PARDIR * kdir_t * torch.exp(-kdir_t * laic)
        visd = (one - scv) * PARDIR * kdirbl * torch.exp(-kdirbl * laic)

        visshd = visdf + vist - visd
        fgrsh = AMAX * (one - torch.exp(-visshd * EFF / amax_denom))

        vispp = (one - scv) * PARDIR / sinb_safe

        exp_term = one - torch.exp(-vispp * EFF / amax_denom)
        # Prevent division by zero in sunlit leaf calculation
        eff_vispp = EFF * vispp
        eff_vispp_safe = torch.where(
            torch.abs(eff_vispp) > epsilon, eff_vispp, torch.ones_like(eff_vispp)
        )
        fgrsun_formula = AMAX * (one - (AMAX - fgrsh) * exp_term / eff_vispp_safe)
        fgrsun = torch.where(vispp <= 0.0, fgrsh, fgrsun_formula)

        fslla = torch.exp(-kdirbl * laic)
        fgl = fslla * fgrsun + (one - fslla) * fgrsh

        fgros = fgros + fgl * wgauss[i]

    fgros = fgros * LAI
    return fgros


class WOFOST72_Assimilation(SimulationObject):
    """Class implementing a WOFOST/SUCROS style assimilation routine.

    WOFOST calculates the daily gross CO2 assimilation rate of a crop
    from the absorbed radiation and the photosynthesis-light response curve
    of individual leaves. This response is dependent on temperature and
    leaf age. The absorbed radiation is calculated from the total incoming
    radiation and the leaf area. Daily gross CO2 assimilation is obtained
    by integrating the assimilation rates over the leaf layers and over the
    day.

    **Simulation parameters** (provide in cropdata dictionary)

    | Name   | Description                                                        | Type | Unit                               |
    |--------|--------------------------------------------------------------------|------|------------------------------------|
    | AMAXTB | Max. leaf CO2 assimilation rate as function of DVS                 | TCr  | kg CO2 ha⁻¹ leaf h⁻¹               |
    | EFFTB  | Light use effic. single leaf as a function of daily mean temperature                  | TCr  | kg CO2 ha⁻¹ h⁻¹ /(J m⁻² s⁻¹)      |
    | KDIFTB | Extinction coefficient for diffuse visible light as function of DVS| TCr  | -                                  |
    | TMPFTB | Reduction factor on AMAX as function of daily mean temperature                      | TCr  | -                                  |
    | TMNFTB | Reduction factor on AMAX as function of daily minimum temperature         | TCr  | -                                  |

    **Rate variables**
    This class returns the potential gross assimilation rate 'PGASS'
    directly from the `__call__()` method, but also includes it as a rate variable.

    | Name  | Description                  | Pbl | Unit             |
    |-------|------------------------------|-----|------------------|
    | PGASS | Potential gross assimilation | Y   | kg CH2O ha⁻¹ d⁻¹ |

    **External dependencies**

    | Name | Description            | Provided by   | Unit |
    |------|------------------------|---------------|------|
    | DVS  | Crop development stage | DVS_Phenology | -    |
    | LAI  | Leaf area index        | Leaf_dynamics | -    |

    **Weather inputs used**

    | Name  | Description                       | Unit      |
    |-------|-----------------------------------|-----------|
    | IRRAD | Daily shortwave radiation         | J m⁻² d⁻¹ |
    | DTEMP | Daily mean temperature            | °C        |
    | TMIN  | Daily minimum temperature         | °C        |
    | LAT   | Latitude                          | degrees   |

    **Outputs**

    | Name  | Description                  | Pbl | Unit             |
    |-------|------------------------------|-----|------------------|
    | PGASS | Potential gross assimilation | Y   | kg CH2O ha⁻¹ d⁻¹ |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                 |
    |--------|-------------------------------------------|
    | PGASS  | AMAXTB, EFFTB, KDIFTB, TMPFTB, TMNFTB     |
    """  # noqa: E501

    params_shape = None

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    class Parameters(TensorParamTemplate):
        AMAXTB = AfgenTrait()
        EFFTB = AfgenTrait()
        KDIFTB = AfgenTrait()
        TMPFTB = AfgenTrait()
        TMNFTB = AfgenTrait()

    class RateVariables(TensorRatesTemplate):
        PGASS = Tensor(0.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | torch.Size | None = None,
    ) -> None:
        """Initialize the assimilation module."""
        self.kiosk = kiosk
        self.params = self.Parameters(parvalues, shape=shape)
        self.params_shape = _get_params_shape(self.params)
        self.rates = self.RateVariables(kiosk, publish=["PGASS"], shape=shape)

        # 7-day running average buffer for TMIN (stored as tensors).
        self._tmn_window = deque(maxlen=7)
        self._tmn_window_mask = deque(maxlen=7)
        # Reused scalar constants
        self._epsilon = torch.tensor(1e-12, dtype=self.dtype, device=self.device)

    @prepare_rates
    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None) -> None:
        """Compute the potential gross assimilation rate (PGASS)."""
        p = self.params
        r = self.rates
        k = self.kiosk

        _exist_required_external_variables(k)

        # External states
        dvs = _broadcast_to(k["DVS"], self.params_shape, dtype=self.dtype, device=self.device)
        lai = _broadcast_to(k["LAI"], self.params_shape, dtype=self.dtype, device=self.device)

        # Weather drivers
        irrad = _get_drv(drv.IRRAD, self.params_shape, dtype=self.dtype, device=self.device)
        dtemp = _get_drv(drv.DTEMP, self.params_shape, dtype=self.dtype, device=self.device)
        tmin = _get_drv(drv.TMIN, self.params_shape, dtype=self.dtype, device=self.device)

        # Assimilation is zero before crop emergence (DVS < 0)
        dvs_mask = (dvs >= 0).to(dtype=self.dtype)
        # 7-day running average of TMIN
        self._tmn_window.appendleft(tmin * dvs_mask)
        self._tmn_window_mask.appendleft(dvs_mask)
        tmin_stack = torch.stack(list(self._tmn_window), dim=0)
        mask_stack = torch.stack(list(self._tmn_window_mask), dim=0)
        tminra = tmin_stack.sum(dim=0) / (mask_stack.sum(dim=0) + 1e-8)

        # Astronomical variables (computed with PCSE util; then broadcast to tensors)
        lat = _as_python_float(drv.LAT)
        irrad_for_astro = _as_python_float(drv.IRRAD)
        dayl, _daylp, sinld, cosld, difpp, _atmtr, dsinbe, _angot = astro(day, lat, irrad_for_astro)

        dayl_t = _broadcast_to(dayl, self.params_shape, dtype=self.dtype, device=self.device)
        sinld_t = _broadcast_to(sinld, self.params_shape, dtype=self.dtype, device=self.device)
        cosld_t = _broadcast_to(cosld, self.params_shape, dtype=self.dtype, device=self.device)
        difpp_t = _broadcast_to(difpp, self.params_shape, dtype=self.dtype, device=self.device)
        dsinbe_t = _broadcast_to(dsinbe, self.params_shape, dtype=self.dtype, device=self.device)

        # Parameter tables
        amax = p.AMAXTB(dvs)
        amax = amax * p.TMPFTB(dtemp)
        kdif = p.KDIFTB(dvs)
        eff = p.EFFTB(dtemp)

        dtga = totass7(
            dayl_t,
            amax,
            eff,
            lai,
            kdif,
            irrad,
            difpp_t,
            dsinbe_t,
            sinld_t,
            cosld_t,
            epsilon=self._epsilon,
        )

        # Correction for low minimum temperature potential
        dtga = dtga * p.TMNFTB(tminra)

        # Convert kg CO2 -> kg CH2O
        pgass = dtga * (30.0 / 44.0)

        # Assimilation is zero before crop emergence (DVS < 0)
        r.PGASS = pgass * dvs_mask
        return r.PGASS

    def __call__(self, day: datetime.date = None, drv: WeatherDataContainer = None) -> torch.Tensor:
        """Calculate and return the potential gross assimilation rate (PGASS)."""
        return self.calc_rates(day, drv)

    @prepare_states
    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """No state variables to integrate for this module."""
        return


def _exist_required_external_variables(kiosk):
    required_external_vars = ["DVS", "LAI"]
    for var in required_external_vars:
        if var not in kiosk:
            raise ValueError(f"Required external variable '{var}' not found in kiosk.")
