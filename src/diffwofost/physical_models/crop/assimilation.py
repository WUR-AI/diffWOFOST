"""SimulationObjects implementing |CO2| Assimilation for use with PCSE."""

import datetime
from collections import deque
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_drv
from diffwofost.physical_models.utils import astro

# ---------------------------------------------------------------------------
# Module-level cache: avoids recreating small constant tensors on every call.
# Keyed by (torch.dtype, str(device)) so different dtype/device combos each
# get their own set of pre-allocated tensors.
# ---------------------------------------------------------------------------
_TENSOR_CONSTANTS: dict = {}


def _get_tensor_constants(dtype: torch.dtype, device) -> dict:
    """Return cached constant tensors for *dtype* / *device*."""
    key = (dtype, str(device))
    if key not in _TENSOR_CONSTANTS:
        _TENSOR_CONSTANTS[key] = {
            "xgauss": torch.tensor([0.1127017, 0.5000000, 0.8872983], dtype=dtype, device=device),
            "wgauss": torch.tensor([0.2777778, 0.4444444, 0.2777778], dtype=dtype, device=device),
            "pi": torch.tensor(torch.pi, dtype=dtype, device=device),
            "scv": torch.tensor(0.2, dtype=dtype, device=device),
            "one": torch.tensor(1.0, dtype=dtype, device=device),
            "two": torch.tensor(2.0, dtype=dtype, device=device),
        }
    return _TENSOR_CONSTANTS[key]


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
    dtype: torch.Size | tuple,
    device: str,
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
    consts = _get_tensor_constants(dtype, device)
    xgauss = consts["xgauss"]
    wgauss = consts["wgauss"]
    pi = consts["pi"]

    # Only compute where it can be non-zero.
    mask = (AMAX > 0) & (LAI > 0) & (DAYL > 0)

    # Prevent division by zero in par calculation
    dsinbe_safe = torch.where(DSINBE > epsilon, DSINBE, torch.ones_like(DSINBE))

    # Vectorized 3-point Gaussian time quadrature: compute sinb, par, pardif,
    # pardir for all three quadrature points simultaneously via a leading
    # quadrature dimension of size 3, replacing the Python for-loop with a
    # single torch.cos call on a (3, *B) tensor.
    ndim = DAYL.dim()
    if ndim > 0:
        xg_v = xgauss.view(3, *([1] * ndim))  # (3, 1, .., 1)
        DAYL_q = DAYL.unsqueeze(0)  # (1, *B)
        SINLD_q = SINLD.unsqueeze(0) if SINLD.dim() > 0 else SINLD
        COSLD_q = COSLD.unsqueeze(0) if COSLD.dim() > 0 else COSLD
        AVRAD_q = AVRAD.unsqueeze(0) if AVRAD.dim() > 0 else AVRAD
        DIFPP_q = DIFPP.unsqueeze(0) if DIFPP.dim() > 0 else DIFPP
        dsinbe_q = dsinbe_safe.unsqueeze(0) if dsinbe_safe.dim() > 0 else dsinbe_safe
    else:
        xg_v = xgauss  # (3,)
        DAYL_q = DAYL
        SINLD_q = SINLD
        COSLD_q = COSLD
        AVRAD_q = AVRAD
        DIFPP_q = DIFPP
        dsinbe_q = dsinbe_safe

    hour = 12.0 + 0.5 * DAYL_q * xg_v  # (3, *B)
    sinb = torch.maximum(
        torch.zeros_like(hour),
        SINLD_q + COSLD_q * torch.cos(2.0 * pi * (hour + 12.0) / 24.0),
    )  # (3, *B) – one cos call
    par = 0.5 * AVRAD_q * sinb * (1.0 + 0.4 * sinb) / dsinbe_q  # (3, *B)
    pardif = torch.minimum(par, sinb * DIFPP_q)  # (3, *B)
    pardir = par - pardif  # (3, *B)

    # Call assim7 for each quadrature slice (sinb[i] etc. are already (*B))
    dtga = torch.zeros_like(AMAX)
    for i in range(3):
        fgros = assim7(AMAX, EFF, LAI, KDIF, sinb[i], pardir[i], pardif[i], epsilon=epsilon)
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
    consts = _get_tensor_constants(AMAX.dtype, AMAX.device)
    xgauss = consts["xgauss"]
    wgauss = consts["wgauss"]
    scv = consts["scv"]
    one = consts["one"]

    # Prevent division by zero in extinction coefficient calculations
    sinb_safe = torch.where(SINB > epsilon, SINB, torch.ones_like(SINB))

    # Extinction coefficients (loop-invariant: do not depend on laic)
    refh = (one - torch.sqrt(one - scv)) / (one + torch.sqrt(one - scv))
    refs = refh * 2.0 / (one + 1.6 * sinb_safe)
    kdirbl = (0.5 / sinb_safe) * KDIF / (0.8 * torch.sqrt(one - scv))
    kdir_t = kdirbl * torch.sqrt(one - scv)
    amax_denom = torch.maximum(consts["two"], AMAX)

    # vispp, exp_term, eff_vispp_safe are also loop-invariant (no laic dependence)
    vispp = (one - scv) * PARDIR / sinb_safe
    exp_term = one - torch.exp(-vispp * EFF / amax_denom)
    eff_vispp = EFF * vispp
    eff_vispp_safe = torch.where(
        torch.abs(eff_vispp) > epsilon, eff_vispp, torch.ones_like(eff_vispp)
    )

    # Vectorized 3-point Gaussian LAI quadrature
    ndim = LAI.dim()
    if ndim > 0:
        xg_v = xgauss.view(3, *([1] * ndim))  # (3, 1, .., 1)
        wg_v = wgauss.view(3, *([1] * ndim))  # (3, 1, .., 1)
        laic = LAI.unsqueeze(0) * xg_v  # (3, *B)
        # Unsqueeze all (*B) tensors to (1, *B) so they broadcast with (3, *B)
        refs_b = refs.unsqueeze(0)
        PARDIF_b = PARDIF.unsqueeze(0)
        KDIF_b = KDIF.unsqueeze(0)
        PARDIR_b = PARDIR.unsqueeze(0)
        kdir_t_b = kdir_t.unsqueeze(0)
        kdirbl_b = kdirbl.unsqueeze(0)
        AMAX_b = AMAX.unsqueeze(0)
        EFF_b = EFF.unsqueeze(0)
        amax_denom_b = amax_denom.unsqueeze(0)
        exp_term_b = exp_term.unsqueeze(0)
        eff_vispp_safe_b = eff_vispp_safe.unsqueeze(0)
        vispp_b = vispp.unsqueeze(0)
    else:
        # Scalar inputs: laic is (3,); skip unsqueezes (broadcasting handles it)
        xg_v = xgauss  # (3,)
        wg_v = wgauss  # (3,)
        laic = LAI * xgauss  # (3,)
        refs_b = refs
        PARDIF_b = PARDIF
        KDIF_b = KDIF
        PARDIR_b = PARDIR
        kdir_t_b = kdir_t
        kdirbl_b = kdirbl
        AMAX_b = AMAX
        EFF_b = EFF
        amax_denom_b = amax_denom
        exp_term_b = exp_term
        eff_vispp_safe_b = eff_vispp_safe
        vispp_b = vispp

    # exp(-kdirbl * laic) is shared between visd and fslla – compute once
    exp_kdirbl_laic = torch.exp(-kdirbl_b * laic)  # (3, *B)

    visdf = (one - refs_b) * PARDIF_b * KDIF_b * torch.exp(-KDIF_b * laic)  # (3, *B)
    vist = (one - refs_b) * PARDIR_b * kdir_t_b * torch.exp(-kdir_t_b * laic)  # (3, *B)
    visd = (one - scv) * PARDIR_b * kdirbl_b * exp_kdirbl_laic  # (3, *B)

    visshd = visdf + vist - visd
    fgrsh = AMAX_b * (one - torch.exp(-visshd * EFF_b / amax_denom_b))  # (3, *B)

    # Prevent division by zero in sunlit leaf calculation
    fgrsun_formula = AMAX_b * (one - (AMAX_b - fgrsh) * exp_term_b / eff_vispp_safe_b)
    fgrsun = torch.where(vispp_b <= 0.0, fgrsh, fgrsun_formula)

    fslla = exp_kdirbl_laic  # reuse shared exponential
    fgl = fslla * fgrsun + (one - fslla) * fgrsh

    # Weighted sum over the quadrature dimension (leading dim 0) → (*B)
    fgros = (fgl * wg_v).sum(0)
    return fgros * LAI


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

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return getattr(self, "_device", ComputeConfig.get_device())

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return getattr(self, "_dtype", ComputeConfig.get_dtype())

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
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()

        self.kiosk = kiosk
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, publish=["PGASS"], shape=shape)

        # 7-day running average buffer for TMIN (stored as tensors).
        self._tmn_window = deque(maxlen=7)
        self._tmn_window_mask = deque(maxlen=7)
        # Reused scalar constants
        self._epsilon = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        # Cache for astro() results keyed by (day, lat).  astro() only depends
        # on day and latitude so the same result can be reused across batch
        # elements (which share the same weather driver).
        self._astro_cache: dict = {}

    def calc_rates(self, day: datetime.date = None, drv: WeatherDataContainer = None) -> None:
        """Compute the potential gross assimilation rate (PGASS)."""
        p = self.params
        r = self.rates
        k = self.kiosk

        _exist_required_external_variables(k)

        # External states
        dvs = _broadcast_to(k["DVS"], self.params.shape, dtype=self.dtype, device=self.device)
        lai = _broadcast_to(k["LAI"], self.params.shape, dtype=self.dtype, device=self.device)

        # Weather drivers
        irrad = _get_drv(drv.IRRAD, self.params.shape, dtype=self.dtype, device=self.device)
        dtemp = _get_drv(drv.DTEMP, self.params.shape, dtype=self.dtype, device=self.device)
        tmin = _get_drv(drv.TMIN, self.params.shape, dtype=self.dtype, device=self.device)

        # Assimilation is zero before crop emergence (DVS < 0)
        dvs_mask = dvs >= 0
        # 7-day running average of TMIN
        self._tmn_window.appendleft(tmin * dvs_mask)
        self._tmn_window_mask.appendleft(dvs_mask)
        tmin_stack = torch.stack(list(self._tmn_window), dim=0)
        mask_stack = torch.stack(list(self._tmn_window_mask), dim=0)
        tminra = tmin_stack.sum(dim=0) / (mask_stack.sum(dim=0) + 1e-8)

        # Astronomical variables computed via vectorized torch astro routine.
        # latitude and radiation are passed directly – they may be scalars or
        # tensors; the function returns torch.Tensor results in all cases.
        dayl, _daylp, sinld, cosld, difpp, _atmtr, dsinbe, _angot = astro(
            day, drv.LAT, drv.IRRAD, dtype=self.dtype, device=self.device
        )

        dayl_t = _broadcast_to(dayl, self.params.shape, dtype=self.dtype, device=self.device)
        sinld_t = _broadcast_to(sinld, self.params.shape, dtype=self.dtype, device=self.device)
        cosld_t = _broadcast_to(cosld, self.params.shape, dtype=self.dtype, device=self.device)
        difpp_t = _broadcast_to(difpp, self.params.shape, dtype=self.dtype, device=self.device)
        dsinbe_t = _broadcast_to(dsinbe, self.params.shape, dtype=self.dtype, device=self.device)

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
            dtype=self.dtype,
            device=self.device,
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

    def integrate(self, day: datetime.date = None, delt=1.0) -> None:
        """No state variables to integrate for this module."""
        return


def _exist_required_external_variables(kiosk):
    required_external_vars = ["DVS", "LAI"]
    for var in required_external_vars:
        if var not in kiosk:
            raise ValueError(f"Required external variable '{var}' not found in kiosk.")
