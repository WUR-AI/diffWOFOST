"""SimulationObjects implementing |CO2| Assimilation for use with PCSE."""

import datetime
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.util import doy
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.deferred import defer
from diffwofost.physical_models.kernels import kernel
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _raise_latitude_error
from diffwofost.physical_models.utils import astro_core
from diffwofost.physical_models.utils import solar_declination_table

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
        # Scalar inputs: laic is (3,); skip unsqueezes (broadcasting handles it).
        # xgauss is already the three quadrature points, so no extra view is needed.
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


def totass8(
    amax_lnb,
    amax_ref,
    amax_slp,
    dayl,
    co2amax,
    tmpf,
    eff,
    kn,
    lai,
    nlv,
    kdif,
    avrad,
    difpp,
    dsinbe,
    sinld,
    cosld,
    *,
    epsilon,
    dtype,
    device,
):
    """Daily gross CO2 assimilation with a leaf-nitrogen profile through the canopy."""
    consts = _get_tensor_constants(dtype, device)
    xgauss = consts["xgauss"]
    wgauss = consts["wgauss"]
    pi = consts["pi"]
    mask = (lai > 0) & (dayl > 0)
    dsinbe_safe = torch.where(dsinbe > epsilon, dsinbe, torch.ones_like(dsinbe))
    ndim = dayl.dim()
    if ndim > 0:
        xg_v = xgauss.view(3, *([1] * ndim))
        dayl_q = dayl.unsqueeze(0)
        sinld_q = sinld.unsqueeze(0) if sinld.dim() > 0 else sinld
        cosld_q = cosld.unsqueeze(0) if cosld.dim() > 0 else cosld
        avrad_q = avrad.unsqueeze(0) if avrad.dim() > 0 else avrad
        difpp_q = difpp.unsqueeze(0) if difpp.dim() > 0 else difpp
        dsinbe_q = dsinbe_safe.unsqueeze(0) if dsinbe_safe.dim() > 0 else dsinbe_safe
    else:
        xg_v = xgauss
        dayl_q = dayl
        sinld_q = sinld
        cosld_q = cosld
        avrad_q = avrad
        difpp_q = difpp
        dsinbe_q = dsinbe_safe
    hour = 12.0 + 0.5 * dayl_q * xg_v
    sinb = torch.maximum(
        torch.zeros_like(hour),
        sinld_q + cosld_q * torch.cos(2.0 * pi * (hour + 12.0) / 24.0),
    )
    par = 0.5 * avrad_q * sinb * (1.0 + 0.4 * sinb) / dsinbe_q
    pardif = torch.minimum(par, sinb * difpp_q)
    pardir = par - pardif
    dtga = torch.zeros_like(lai)
    for i in range(3):
        fgros = assim8(
            amax_lnb,
            amax_ref,
            amax_slp,
            co2amax,
            tmpf,
            eff,
            kn,
            lai,
            nlv,
            kdif,
            sinb[i],
            pardir[i],
            pardif[i],
            epsilon=epsilon,
        )
        dtga = dtga + fgros * wgauss[i]
    dtga = dtga * dayl
    return torch.where(mask, dtga, torch.zeros_like(dtga))


def assim8(
    amax_lnb,
    amax_ref,
    amax_slp,
    co2amax,
    tmpf,
    eff,
    kn,
    lai,
    nlv,
    kdif,
    sinb,
    pardir,
    pardif,
    *,
    epsilon,
):
    """Canopy assimilation with AMAX declining through the canopy with leaf nitrogen."""
    consts = _get_tensor_constants(amax_ref.dtype, amax_ref.device)
    xgauss = consts["xgauss"]
    wgauss = consts["wgauss"]
    scv = consts["scv"]
    one = consts["one"]
    sinb_safe = torch.where(sinb > epsilon, sinb, torch.ones_like(sinb))
    refh = (one - torch.sqrt(one - scv)) / (one + torch.sqrt(one - scv))
    refs = refh * 2.0 / (one + 1.6 * sinb_safe)
    kdirbl = (0.5 / sinb_safe) * kdif / (0.8 * torch.sqrt(one - scv))
    kdir_t = kdirbl * torch.sqrt(one - scv)

    ndim = lai.dim()
    if ndim > 0:
        xg_v = xgauss.view(3, *([1] * ndim))
        wg_v = wgauss.view(3, *([1] * ndim))
        laic = lai.unsqueeze(0) * xg_v

        def _batch(value):
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return value.unsqueeze(0)
            return value

        refs_b = _batch(refs)
        pardif_b = _batch(pardif)
        kdif_b = _batch(kdif)
        pardir_b = _batch(pardir)
        kdir_t_b = _batch(kdir_t)
        kdirbl_b = _batch(kdirbl)
        eff_b = _batch(eff)
        kn_b = _batch(kn)
        nlv_b = _batch(nlv)
        lai_b = _batch(lai)
        co2_b = _batch(co2amax)
        tmpf_b = _batch(tmpf)
        slp_b = _batch(amax_slp)
        lnb_b = _batch(amax_lnb)
        ref_b = _batch(amax_ref)
        vispp = (one - scv) * pardir / sinb_safe
        vispp_b = _batch(vispp)
    else:
        # xgauss is already the three quadrature points, so no extra view is needed.
        wg_v = wgauss
        laic = lai * xgauss
        refs_b = refs
        pardif_b = pardif
        kdif_b = kdif
        pardir_b = pardir
        kdir_t_b = kdir_t
        kdirbl_b = kdirbl
        eff_b = eff
        kn_b = kn
        nlv_b = nlv
        lai_b = lai
        co2_b = co2amax
        tmpf_b = tmpf
        slp_b = amax_slp
        lnb_b = amax_lnb
        ref_b = amax_ref
        vispp = (one - scv) * pardir / sinb_safe
        vispp_b = vispp

    # Calculate AMAX with gradient in canopy ORYZA
    # ``torch.where`` evaluates both branches, so keep the denominators finite
    # even when LAI is zero. The selected SLN still follows PCSE: the nitrogen
    # profile when LAI >= 0.01, otherwise NLV / LAI.
    safe_lai = torch.clamp(lai_b, min=epsilon)
    use_profile = lai_b >= 0.01
    sln_profile = nlv_b * kn_b * torch.exp(-kn_b * laic) / (1.0 - torch.exp(-kn_b * safe_lai))
    sln_uniform = nlv_b / safe_lai
    sln = torch.where(use_profile, sln_profile, sln_uniform)
    # minimum/maximum broadcast a batched AMAX_REF; clamp rejects a tensor
    # bound mixed with a Python float.
    leaf_response = slp_b * (sln - lnb_b)
    leaf_response = torch.minimum(
        torch.maximum(leaf_response, torch.zeros_like(leaf_response)), ref_b
    )
    amax = co2_b * tmpf_b * leaf_response
    amax_denom = torch.maximum(consts["two"], amax)

    # absorbed diffuse radiation (VISDF),light from direct
    # origine (VIST) and direct light (VISD)
    exp_kdirbl_laic = torch.exp(-kdirbl_b * laic)
    visdf = (one - refs_b) * pardif_b * kdif_b * torch.exp(-kdif_b * laic)
    vist = (one - refs_b) * pardir_b * kdir_t_b * torch.exp(-kdir_t_b * laic)
    visd = (one - scv) * pardir_b * kdirbl_b * exp_kdirbl_laic
    # absorbed flux in W/m2 for shaded leaves and assimilation
    visshd = visdf + vist - visd
    fgrsh = amax * (one - torch.exp(-visshd * eff_b / amax_denom))
    # direct light absorbed by leaves perpendicular on direct
    # beam and assimilation of sunlit leaf area
    exp_term = one - torch.exp(-vispp_b * eff_b / amax_denom)
    eff_vispp = eff_b * vispp_b
    eff_vispp_safe = torch.where(
        torch.abs(eff_vispp) > epsilon, eff_vispp, torch.ones_like(eff_vispp)
    )
    fgrsun_formula = amax * (one - (amax - fgrsh) * exp_term / eff_vispp_safe)
    fgrsun = torch.where(vispp_b <= 0.0, fgrsh, fgrsun_formula)
    # fraction of sunlit leaf area (FSLLA) and local
    # assimilation rate (FGL)
    fslla = exp_kdirbl_laic
    fgl = fslla * fgrsun + (one - fslla) * fgrsh
    # integration
    fgros = (fgl * wg_v).sum(0)
    return fgros * lai


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

        # 7-day running average buffer for TMIN: a ring buffer of the last 7 days
        # (entries of elements that had not emerged, or of days not yet simulated, are 0).
        self._tmn_window = None
        self._tmn_window_mask = None
        self._tmn_days = 0

        # Reused scalar constants
        self._epsilon = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        self._solar = solar_declination_table(self.dtype, self.device)

    def calc_rates(self, day: datetime.date, drv: dict) -> torch.Tensor:
        """Compute the potential gross assimilation rate (PGASS)."""
        p = self.params
        r = self.rates
        k = self.kiosk

        _exist_required_external_variables(k)

        # External states
        dvs = _broadcast_to(k["DVS"], self.params.shape, dtype=self.dtype, device=self.device)
        lai = _broadcast_to(k["LAI"], self.params.shape, dtype=self.dtype, device=self.device)

        # Assimilation is zero before crop emergence (DVS < 0)
        dvs_mask = dvs >= 0
        # 7-day running average of TMIN
        tmin = drv["TMIN"] * dvs_mask

        if self._tmn_window is None:
            self._tmn_window = torch.zeros((7, *tmin.shape), dtype=self.dtype, device=self.device)
            self._tmn_window_mask = torch.zeros_like(self._tmn_window)
        slot = self._tmn_days % 7
        self._tmn_days += 1
        self._tmn_window[slot] = tmin
        self._tmn_window_mask[slot] = dvs_mask

        latitude = torch.as_tensor(drv["LAT"], dtype=self.dtype, device=self.device)
        defer(torch.any(latitude.abs() > 90.0), _raise_latitude_error)
        r.PGASS = _potential_assimilation7(
            dvs,
            lai,
            torch.as_tensor(drv["IRRAD"], dtype=self.dtype, device=self.device),
            drv["DTEMP"],
            self._tmn_window,
            self._tmn_window_mask,
            latitude,
            self._solar[doy(day)],
            p.AMAXTB,
            p.TMPFTB,
            p.KDIFTB,
            p.EFFTB,
            p.TMNFTB,
            self._epsilon,
        )
        return r.PGASS

    def __call__(self, day: datetime.date, drv: dict) -> torch.Tensor:
        """Calculate and return the potential gross assimilation rate (PGASS)."""
        return self.calc_rates(day, drv)

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """No state variables to integrate for this module."""
        return


@kernel
def _potential_assimilation7(
    dvs, lai, irrad, dtemp, tmn_window, tmn_window_mask, latitude, solar,
    AMAXTB, TMPFTB, KDIFTB, EFFTB, TMNFTB, epsilon,
):  # fmt: skip
    """Potential gross assimilation rate (PGASS) of one day, see `WOFOST72_Assimilation`."""
    dvs_mask = dvs >= 0
    tminra = tmn_window.sum(dim=0) / (tmn_window_mask.sum(dim=0) + 1e-8)

    # Astronomical variables; latitude and radiation may be scalars or tensors.
    dayl, _daylp, sinld, cosld, difpp, _atmtr, dsinbe, _angot = astro_core(
        latitude, irrad, solar[0], solar[1], solar[2]
    )

    # Parameter tables
    amax = AMAXTB(dvs)
    amax = amax * TMPFTB(dtemp)
    kdif = KDIFTB(dvs)
    eff = EFFTB(dtemp)

    dtga = totass7(
        dayl,
        amax,
        eff,
        lai,
        kdif,
        irrad,
        difpp,
        dsinbe,
        sinld,
        cosld,
        epsilon=epsilon,
        dtype=dvs.dtype,
        device=dvs.device,
    )

    # Correction for low minimum temperature potential
    dtga = dtga * TMNFTB(tminra)

    # Convert kg CO2 -> kg CH2O; assimilation is zero before crop emergence (DVS < 0)
    return dtga * (30.0 / 44.0) * dvs_mask


def _exist_required_external_variables(kiosk):
    required_external_vars = ["DVS", "LAI"]
    for var in required_external_vars:
        if var not in kiosk:
            raise ValueError(f"Required external variable '{var}' not found in kiosk.")


class WOFOST81_Assimilation(SimulationObject):
    """Class implementing a WOFOST/SUCROS style assimilation routine including
    effect of changes in atmospheric CO2 concentration and impact of
    leaf nitrogen content on the maximum assimilation rate.

    WOFOST calculates the daily gross |CO2| assimilation rate of a crop
    from the absorbed radiation and the photosynthesis-light response curve
    of individual leaves. This response is dependent on temperature and
    leaf age. The absorbed radiation is calculated from the total incoming
    radiation and the leaf area. Daily gross |CO2| assimilation is obtained
    by integrating the assimilation rates over the leaf layers and over the
    day.

    *Simulation parameters* (To be provided in cropdata dictionary):

    | Name      | Description                                    | Type | Unit |
    |-----------|------------------------------------------------|------|------|
    | AMAX_LNB  | Specific leaf nitrogen below which there is    | Cr   | |kg ha-1| |
    |           | no gross photosynthesis                        |      |      |
    | AMAX_REF  | Maximum leaf CO2 assim. rate under reference   | TCr  | |kg ha-1 hr-1| |
    |           | conditions and high specific leaf nitrogen.    |      |      |
    | AMAX_SLP  | Slope of linear response of AMAX to specific   | Cr   | |kg hr-1 kg-1| |
    |           | leaf nitrogen content at reference conditions  |      |      |
    | KN        | Extinction coefficient of N in the canopy      | Cr   | -    |
    | EFFTB     | Light use effic. single leaf as a function     | TCr  | |kg ha-1 hr-1 |
    |           | of daily mean temperature                      |      | /(J m-2 s-1)| |
    | KDIFTB    | Extinction coefficient for diffuse visible     | TCr  | -    |
    |           | as function of DVS                             |      |      |
    | TMPFTB    | Reduction factor of AMAX as function of        | TCr  | -    |
    |           | daily mean temperature.                        |      |      |
    | TMNFTB    | Reduction factor of AMAX as function of        | TCr  | -    |
    |           | daily minimum temperature.                     |      |      |
    | CO2AMAXTB | Correction factor for AMAX given atmos-        | TCr  | -    |
    |           | pheric CO2 concentration.                      |      |      |
    | CO2EFFTB  | Correction factor for EFF given atmos-         | TCr  | -    |
    |           | pheric CO2 concentration.                      |      |      |
    | CO2       | Atmopheric CO2 concentration                   | SCr  | ppm  |

    *State and rate variables*

    `WOFOST_Assimilation` has no state/rate variables, but calculates the
    rate of assimilation which is returned directly from the `__call__()`
    method.

    *Signals sent or handled*

    None

    *External dependencies:*

    | Name | Description            | Provided by   | Unit    |
    |------|------------------------|---------------|---------|
    | DVS  | Crop development stage | DVS_Phenology | -       |
    | LAI  | Leaf area index        | leaf_dynamics | -       |
    | NLV  | Leaf nitrogen amount   | n_dynamics    | kg ha-1 |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                                      |
    |--------|----------------------------------------------------------------|
    | PGASS  | AMAX_LNB, AMAX_REF, AMAX_SLP, EFFTB, KDIFTB, TMPFTB, TMNFTB,   |
    |        | CO2AMAXTB, CO2EFFTB, CO2, KN                                   |

    [!NOTE]
    ``TMNFTB`` is applied to the mean of the emerged days in a 7-day window.
    The window itself is a buffer, not a torch operation.
    """  # noqa: D205

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return getattr(self, "_device", ComputeConfig.get_device())

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return getattr(self, "_dtype", ComputeConfig.get_dtype())

    class Parameters(TensorParamTemplate):
        AMAX_LNB = Tensor(-99.0)
        AMAX_REF = Tensor(-99.0)
        AMAX_SLP = Tensor(-99.0)
        EFFTB = AfgenTrait()
        KDIFTB = AfgenTrait()
        TMPFTB = AfgenTrait()
        TMNFTB = AfgenTrait()
        CO2AMAXTB = AfgenTrait()
        CO2EFFTB = AfgenTrait()
        CO2 = Tensor(-99.0)
        KN = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        PGASS = Tensor(0.0)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Store parameters and the 7-day minimum-temperature window."""
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()

        self.kiosk = kiosk
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, publish=["PGASS"], shape=shape)

        # 7-day running average buffer for TMIN: a ring buffer of the last 7 days
        # (entries of elements that had not emerged, or of days not yet simulated, are 0).
        self._tmn_window = None
        self._tmn_window_mask = None
        self._tmn_days = 0

        # Reused scalar constants
        self._epsilon = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        self._solar = solar_declination_table(self.dtype, self.device)

    def calc_rates(self, day: datetime.date, drv: dict) -> torch.Tensor:
        """Potential gross assimilation in kg CH2O ha-1 d-1."""
        params = self.params
        rates = self.rates
        kiosk = self.kiosk

        _exist_required_external_variables(kiosk)

        # Published states from the kiosk.
        dvs = _broadcast_to(kiosk["DVS"], self.params.shape, dtype=self.dtype, device=self.device)
        lai = _broadcast_to(kiosk["LAI"], self.params.shape, dtype=self.dtype, device=self.device)
        nlv = _broadcast_to(
            kiosk["NamountLV"], self.params.shape, dtype=self.dtype, device=self.device
        )

        # TMPFTB uses the daily mean temperature. EFFTB uses the daytime
        # temperature, matching PCSE's WOFOST 8.1 assimilation.
        temp = drv["TEMP"]
        dtemp = drv["DTEMP"]

        # 7-day running average of TMIN.
        # PCSE starts filling this window only once Wofost81 leaves the emerging
        # stage and calls assimilation. Pre-emergence days are excluded from the
        # average so a mixed batch does not pollute emerged members.
        # Assimilation is zero before crop emergence (DVS < 0)
        dvs_mask = dvs >= 0
        tmin = drv["TMIN"] * dvs_mask

        # 7-day running average of TMIN
        if self._tmn_window is None:
            self._tmn_window = torch.zeros((7, *tmin.shape), dtype=self.dtype, device=self.device)
            self._tmn_window_mask = torch.zeros_like(self._tmn_window)
        slot = self._tmn_days % 7
        self._tmn_days += 1
        self._tmn_window[slot] = tmin
        self._tmn_window_mask[slot] = dvs_mask

        latitude = torch.as_tensor(drv["LAT"], dtype=self.dtype, device=self.device)
        defer(torch.any(latitude.abs() > 90.0), _raise_latitude_error)

        irrad = torch.as_tensor(drv["IRRAD"], dtype=self.dtype, device=self.device)

        # Assimilation in kg CH2O per ha.
        # Same as PCSE: no DVS factor here. Wofost81 skips this module while
        # STAGE is emerging for the whole batch.
        rates.PGASS = _potential_assimilation8(
            dvs,
            lai,
            nlv,
            irrad,
            dtemp,
            self._tmn_window,
            self._tmn_window_mask,
            latitude,
            params.CO2,
            params.AMAX_LNB,
            params.AMAX_REF,
            params.AMAX_SLP,
            params.KN,
            params.CO2AMAXTB,
            params.CO2EFFTB,
            temp,
            self._solar[doy(day)],
            params.TMPFTB,
            params.KDIFTB,
            params.EFFTB,
            params.TMNFTB,
            self._epsilon
        )
        return rates.PGASS

    def __call__(self, day: datetime.date, drv: dict) -> torch.Tensor:
        """PCSE calls the assimilation module directly."""
        return self.calc_rates(day, drv)

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """Assimilation has no states to integrate."""
        return


@kernel
def _potential_assimilation8(
    dvs,
    lai,
    nlv,
    irrad,
    dtemp,
    tmn_window,
    tmn_window_mask,
    latitude,
    CO2,
    AMAX_LNB,
    AMAX_REF,
    AMAX_SLP,
    KN,
    CO2AMAXTB,
    CO2EFFTB,
    temp,
    solar,
    TMPFTB,
    KDIFTB,
    EFFTB,
    TMNFTB,
    epsilon,
):  # fmt: skip
    """Potential gross assimilation rate (PGASS) of one day, see `WOFOST72_Assimilation`."""
    # Astronomical variables; latitude and radiation may be scalars or tensors.
    dayl, _daylp, sinld, cosld, difpp, _atmtr, dsinbe, _angot = astro_core(
        latitude, irrad, solar[0], solar[1], solar[2]
    )

    # Daily gross assimilation.
    # CO2 and temperature response factors of AMAX.
    tmpf = TMPFTB(temp)
    eff = EFFTB(dtemp) * CO2EFFTB(CO2)

    # Gross assimilation, then the low-minimum-temperature correction.
    dtga = totass8(
        AMAX_LNB,
        AMAX_REF,
        AMAX_SLP,
        dayl,
        CO2AMAXTB(CO2),
        tmpf,
        eff,
        KN,
        lai,
        nlv,
        KDIFTB(dvs),
        irrad,
        difpp,
        dsinbe,
        sinld,
        cosld,
        epsilon=epsilon,
        dtype=dvs.dtype,
        device=dvs.device,
    )

    # Correction for low minimum temperature potential
    tminra = tmn_window.sum(dim=0) / (tmn_window_mask.sum(dim=0) + 1e-8)
    dtga = dtga * TMNFTB(tminra)

    # Assimilation in kg CH2O per ha.
    # Same as PCSE: no DVS factor here. Wofost81 skips this module while
    # STAGE is emerging for the whole batch.
    return dtga * (30.0 / 44.0)
