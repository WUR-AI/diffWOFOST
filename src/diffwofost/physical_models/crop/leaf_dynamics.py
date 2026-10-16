"""Leaf dynamics for the WOFOST crop model."""

import datetime
import torch
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.kernels import kernel
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait


class WOFOST_Leaf_Dynamics(SimulationObject):
    """Leaf dynamics for the WOFOST crop model.

    Implementation of biomass partitioning to leaves, growth and senenscence
    of leaves. WOFOST keeps track of the biomass that has been partitioned to
    the leaves for each day (variable `LV`), which is called a leaf class).
    For each leaf class the leaf age (variable 'LVAGE') and specific leaf area
    (variable `SLA`) are also registered. Total living leaf biomass is
    calculated by summing the biomass values for all leaf classes. Similarly,
    leaf area is calculated by summing leaf biomass times specific leaf area
    (`LV` * `SLA`).

    Senescense of the leaves can occur as a result of physiological age,
    drought stress or self-shading.

    Leaves die from the oldest class onwards, so the living leaf classes are
    always the youngest ones. Instead of updating the weight of every leaf
    class every day, the weights and areas are stored as running totals over
    the leaf classes at the time they were formed (`LVCUM`, `LACUM`), together
    with the total weight that has died (`DLVCUM`). The living weight of class
    `i` is then `clip(LVCUM[i] - DLVCUM, 0, LVCUM[i] - LVCUM[i-1])`, and the
    oldest living class is found with a binary search. Leaf age (`LVAGE`) is
    accumulated per class as in PCSE; it decreases with the class index, so
    the classes older than `SPAN` are the oldest ones and their living weight
    is a difference of two running totals as well.

    **Simulation parameters** (provide in cropdata dictionary)

    | Name   | Description                                                        | Type | Unit       |
    |--------|------------------------------------------------------------------  |------|------------|
    | RGRLAI | Maximum relative increase in LAI.                                  |SCr   | ha ha⁻¹ d⁻¹|
    | SPAN   | Life span of leaves growing at 35 Celsius                          |SCr   | d          |
    | TBASE  | Lower threshold temp. for ageing of leaves                         |SCr   | C          |
    | PERDL  | Max. relative death rate of leaves due to water stress             |SCr   |            |
    | TDWI   | Initial total crop dry weight                                      |SCr   | kg ha⁻¹    |
    | KDIFTB | Extinction coefficient for diffuse visible light as function of DVS|TCr   |            |
    | SLATB  | Specific leaf area as a function of DVS                            |TCr   | ha kg⁻¹    |

    **State variables**

    | Name   | Description                                           | Pbl  | Unit        |
    |--------|-------------------------------------------------------|------|-------------|
    | LVCUM  | Leaf biomass formed up to and including each leaf class | N  | kg ha⁻¹     |
    | LACUM  | Leaf area formed up to and including each leaf class  | N    | ha ha⁻¹     |
    | SLA    | Specific leaf area per leaf class                     | N    | ha kg⁻¹     |
    | LVAGE  | Leaf age per leaf class                               | N    | d           |
    | DLVCUM | Leaf biomass died from the leaf classes since the start | N  | kg ha⁻¹     |
    | LAIEM  | LAI at emergence                                      | N    | -           |
    | LASUM  | Total leaf area as sum of LV*SLA, not including stem and pod area | N | -  |
    | LAIEXP | LAI value under theoretical exponential growth        | N    | -           |
    | LAIMAX | Maximum LAI reached during growth cycle               | N    | -           |
    | LAI    | Leaf area index, including stem and pod area          | Y    | -           |
    | WLV    | Dry weight of living leaves                           | Y    | kg ha⁻¹     |
    | DWLV   | Dry weight of dead leaves                             | N    | kg ha⁻¹     |
    | TWLV   | Dry weight of total leaves (living + dead)            | Y    | kg ha⁻¹     |

    **Rate variables**

    | Name   | Description                                           | Pbl  | Unit          |
    |--------|-------------------------------------------------------|------|---------------|
    | GRLV   | Growth rate leaves                                    | N    | kg ha⁻¹ d⁻¹   |
    | DSLV1  | Death rate leaves due to water stress                 | N    | kg ha⁻¹ d⁻¹   |
    | DSLV2  | Death rate leaves due to self-shading                 | N    | kg ha⁻¹ d⁻¹   |
    | DSLV3  | Death rate leaves due to frost kill                   | N    | kg ha⁻¹ d⁻¹   |
    | DSLV   | Maximum of DSLV1, DSLV2, DSLV3                        | N    | kg ha⁻¹ d⁻¹   |
    | DALV   | Death rate leaves due to aging                        | N    | kg ha⁻¹ d⁻¹   |
    | DRLV   | Death rate leaves as a combination of DSLV and DALV   | N    | kg ha⁻¹ d⁻¹   |
    | SLAT   | Specific leaf area for current time step, adjusted for source/sink limited leaf expansion rate | N | ha kg⁻¹ |
    | FYSAGE | Increase in physiological leaf age                    | N    | -             |
    | GLAIEX | Sink-limited leaf expansion rate (exponential curve)  | N    | ha ha⁻¹ d⁻¹   |
    | GLASOL | Source-limited leaf expansion rate (biomass increase) | N    | ha ha⁻¹ d⁻¹   |

    **External dependencies**

    | Name      | Description                       | Provided by                    | Unit           |
    |-----------|-----------------------------------|--------------------------------|----------------|
    | DVS       | Crop development stage            | DVS_Phenology                  | -              |
    | FL        | Fraction biomass to leaves        | DVS_Partitioning               | -              |
    | FR        | Fraction biomass to roots         | DVS_Partitioning               | -              |
    | SAI       | Stem area index                   | WOFOST_Stem_Dynamics           | -              |
    | PAI       | Pod area index                    | WOFOST_Storage_Organ_Dynamics  | -              |
    | TRA       | Transpiration rate                | Evapotranspiration             | cm day⁻¹ ?     |
    | TRAMX     | Maximum transpiration rate        | Evapotranspiration             | cm day⁻¹ ?     |
    | ADMI      | Above-ground dry matter increase  | CropSimulation                 | kg ha⁻¹ d⁻¹    |
    | RFTRA     | Reduction factor for transpiration (water & oxygen)   | Y          | -              |
    | RF_FROST  | Reduction factor frost kill       | FROSTOL (optional)             | -              |

    **Outputs**

    | Name   | Description                                           | Pbl  | Unit        |
    |--------|-------------------------------------------------------|------|-------------|
    | LAI    | Leaf area index, including stem and pod area          | Y    | -           |
    | TWLV   | Dry weight of total leaves (living + dead)            | Y    | kg ha⁻¹     |

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                |
    |--------|------------------------------------------|
    | LAI    | TDWI, SPAN, RGRLAI, TBASE, KDIFTB, SLATB |
    | TWLV   | TDWI, PERDL                              |

    [!NOTE]
    Notice that the following gradients are zero:
        - ∂SPAN/∂TWLV
        - ∂PERDL/∂TWLV
        - ∂KDIFTB/∂LAI
    """  # noqa: E501

    # The following parameters are used to initialize and control the arrays that store information
    # on the leaf classes during the time integration: leaf area, age, and biomass.
    START_DATE = None  # Start date of the simulation
    MAX_DAYS = 365  # Maximum number of days that can be simulated in one run (i.e. array lenghts)

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return getattr(self, "_device", ComputeConfig.get_device())

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return getattr(self, "_dtype", ComputeConfig.get_dtype())

    class Parameters(TensorParamTemplate):
        RGRLAI = Tensor(-99.0)
        SPAN = Tensor(-99.0)
        TBASE = Tensor(-99.0)
        PERDL = Tensor(-99.0)
        TDWI = Tensor(-99.0)
        SLATB = AfgenTrait()
        KDIFTB = AfgenTrait()

    class StateVariables(TensorStatesTemplate):
        LVCUM = Tensor(-99.0)
        LACUM = Tensor(-99.0)
        SLA = Tensor(-99.0)
        LVAGE = Tensor(-99.0)
        DLVCUM = Tensor(-99.0)
        LAIEM = Tensor(-99.0)
        LASUM = Tensor(-99.0)
        LAIEXP = Tensor(-99.0)
        LAIMAX = Tensor(-99.0)
        LAI = Tensor(-99.0)
        WLV = Tensor(-99.0)
        DWLV = Tensor(-99.0)
        TWLV = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        GRLV = Tensor(0.0)
        DSLV1 = Tensor(0.0)
        DSLV2 = Tensor(0.0)
        DSLV3 = Tensor(0.0)
        DSLV = Tensor(0.0)
        DALV = Tensor(0.0)
        DRLV = Tensor(0.0)
        SLAT = Tensor(0.0)
        FYSAGE = Tensor(0.0)
        GLAIEX = Tensor(0.0)
        GLASOL = Tensor(0.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | torch.Size | None = None,
    ) -> None:
        """Initialize the WOFOST_Leaf_Dynamics simulation object.

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
        self.START_DATE = day
        self.kiosk = kiosk

        # Get defaults from ComputeConfig if not already set
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()

        # TODO check if parvalues are already torch.nn.Parameters
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(kiosk, shape=shape)

        # Create scalar constants once at the beginning to avoid recreating them
        self._zero = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self._epsilon = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        self._sigmoid_sharpness = torch.tensor(1000, dtype=self.dtype, device=self.device)
        self._sigmoid_epsilon = torch.tensor(1e-14, dtype=self.dtype, device=self.device)

        # CALCULATE INITIAL STATE VARIABLES
        # check for required external variables
        _exist_required_external_variables(self.kiosk)

        params = self.params

        # Initial leaf biomass
        WLV = (params.TDWI * (1 - self.kiosk["FR"])) * self.kiosk["FL"]
        DWLV = 0.0
        TWLV = WLV + DWLV

        # Initialize leaf classes (leading axis). Classes not formed yet hold +inf in the
        # running totals, which keeps them sorted and never selected.
        WLV = torch.as_tensor(WLV, dtype=self.dtype, device=self.device).expand(params.shape)
        cls_shape = (self.MAX_DAYS, *params.shape)
        inf = float("inf")
        LVCUM = torch.full(cls_shape, inf, dtype=self.dtype, device=self.device)
        LACUM = torch.full(cls_shape, inf, dtype=self.dtype, device=self.device)
        LVAGE = torch.zeros(cls_shape, dtype=self.dtype, device=self.device)
        SLA = torch.zeros(cls_shape, dtype=self.dtype, device=self.device)
        SLA0 = params.SLATB(self.kiosk["DVS"])
        LAIEM = WLV * SLA0
        SLA[0] = SLA0
        LVCUM[0] = WLV
        LACUM[0] = LAIEM
        self._last = 0  # index of the youngest leaf class formed so far
        # number of leaf classes older than SPAN (see _leaf_ageing)
        self._n_older = (LVAGE[:1] > params.SPAN).sum(dim=0)

        # Initial values for leaf area
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        LAI = LASUM + self.kiosk["SAI"] + self.kiosk["PAI"]

        # Initialize StateVariables object
        self.states = self.StateVariables(
            kiosk,
            publish=["LAI", "TWLV", "WLV"],
            do_not_broadcast=["LVCUM", "LACUM", "SLA", "LVAGE"],
            LVCUM=LVCUM,
            LACUM=LACUM,
            SLA=SLA,
            LVAGE=LVAGE,
            DLVCUM=0.0,
            LAIEM=LAIEM,
            LASUM=LASUM,
            LAIEXP=LAIEXP,
            LAIMAX=LAIMAX,
            LAI=LAI,
            WLV=WLV,
            DWLV=DWLV,
            TWLV=TWLV,
            shape=shape,
        )

    def calc_rates(self, day: datetime.date, drv: dict) -> None:
        """Calculate the rates of change for the leaf dynamics.

        Args:
            day (datetime.date): The current date of the simulation.
            drv (dict): A container holding weather data elements as key/value. The values are
                arrays or scalars.
        """
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        # Using a sigmoid here instead of a conditional statement on the value of
        # SPAN because the latter would not allow for the gradient to be tracked.
        # the if statement `p.SPAN.requires_grad` to avoid unnecessary
        # approximation when SPAN is not a learnable parameter.
        # here we use STE (straight through estimator) method: the forward pass
        # uses the hard mask (see _leaf_rates), the backward pass the soft mask.
        # TODO: sharpness can be exposed as a parameter
        DALV_soft = None
        if p.SPAN.requires_grad:
            n = self._last + 1
            LV_alive = torch.clamp(s.LVCUM[:n] - s.DLVCUM, min=0.0)
            LV = torch.diff(LV_alive, dim=0, prepend=torch.zeros_like(LV_alive[:1]))
            soft_mask = torch.sigmoid(
                (s.LVAGE[:n] - p.SPAN - self._sigmoid_epsilon) / self._sigmoid_sharpness
            )
            DALV_soft = torch.sum((soft_mask - soft_mask.detach()) * LV, dim=0)

        (
            r.GRLV,
            r.DSLV1,
            r.DSLV2,
            r.DSLV3,
            r.DSLV,
            r.DALV,
            r.DRLV,
            r.FYSAGE,
            r.SLAT,
            r.GLAIEX,
            r.GLASOL,
        ) = _leaf_rates(
            k["DVS"],
            k.ADMI,
            k.FL,
            k.RFTRA,
            k.RF_FROST if "RF_FROST" in self.kiosk else None,
            drv["TEMP"],
            s.WLV,
            s.LAI,
            s.LAIEXP,
            s.LVCUM,
            self._n_older,
            s.DLVCUM,
            DALV_soft,
            r.GLAIEX,
            r.GLASOL,
            p.RGRLAI,
            p.SPAN,
            p.TBASE,
            p.PERDL,
            p.SLATB,
            p.KDIFTB,
            self._zero,
            self._epsilon,
        )

    def integrate(self, day: datetime.date, delt=1.0) -> None:
        """Integrate the leaf dynamics state variables.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        rates = self.rates
        states = self.states
        last = self._last
        idx = int((day - self.START_DATE).days / delt)
        if idx <= last:
            msg = f"Leaf class {idx} was already formed (last class is {last})."
            raise ValueError(msg)

        if idx > last + 1:
            # days on which integrate() was skipped (before emergence) form empty classes
            gap = slice(last + 1, idx)
            states.LVCUM[gap] = _row(states.LVCUM, last)
            states.LACUM[gap] = _row(states.LACUM, last)
            states.SLA[gap] = 0.0
            states.LVAGE[gap] = 0.0

        # leaf death and the new leaf class
        states.DLVCUM, LV_formed, LA_formed = _leaf_death_and_growth(
            _row(states.LVCUM, last),
            _row(states.LACUM, last),
            states.DLVCUM,
            rates.DRLV,
            rates.GRLV,
            rates.SLAT,
        )
        _set(states.LVCUM, idx, LV_formed)
        _set(states.LACUM, idx, LA_formed)
        _set(states.SLA, idx, rates.SLAT)
        self._last = idx

        # Integration of physiological age of the classes formed before today
        # (the new class has age 0)
        self._n_older = _leaf_ageing(states.LVAGE[:idx], rates.FYSAGE, self.params.SPAN)

        (
            states.LASUM,
            states.LAI,
            states.LAIMAX,
            states.LAIEXP,
            states.WLV,
            states.DWLV,
            states.TWLV,
        ) = _leaf_area_and_weight(
            states.LVCUM,
            states.LACUM,
            states.SLA,
            states.DLVCUM,
            LV_formed,
            LA_formed,
            self.kiosk["SAI"],
            self.kiosk["PAI"],
            states.LAIMAX,
            states.LAIEXP,
            states.DWLV,
            rates.GLAIEX,
            rates.DRLV,
        )


@kernel
def _leaf_rates(
    DVS, ADMI, FL, RFTRA, RF_FROST, TEMP,
    WLV, LAI, LAIEXP, LVCUM, n_older, DLVCUM, DALV_soft, GLAIEX_old, GLASOL_old,
    RGRLAI, SPAN, TBASE, PERDL, SLATB, KDIFTB, zero, epsilon,
):  # fmt: skip
    """Rates of change of the leaf dynamics, see `WOFOST_Leaf_Dynamics.calc_rates`."""
    # If DVS < 0, the crop has not yet emerged, so we zerofy the rates using mask
    # A mask (0 if DVS < 0, 1 if DVS >= 0)
    dvs_mask = DVS >= zero

    # Growth rate leaves
    # weight of new leaves
    GRLV = dvs_mask * ADMI * FL

    # death of leaves due to water/oxygen stress
    DSLV1 = dvs_mask * WLV * (1.0 - RFTRA) * PERDL

    # death due to self shading cause by high LAI
    LAICR = 3.2 / KDIFTB(DVS)
    DSLV2 = dvs_mask * WLV * torch.clamp(0.03 * (LAI - LAICR) / LAICR, 0.0, 0.03)

    # Death of leaves due to frost damage as determined by
    # Reduction Factor Frost "RF_FROST"
    if RF_FROST is not None:
        DSLV3 = WLV * RF_FROST
    else:
        DSLV3 = torch.zeros_like(WLV)
    DSLV3 = dvs_mask * DSLV3

    # leaf death equals maximum of water stress, shading and frost
    DSLV = torch.maximum(torch.maximum(DSLV1, DSLV2), DSLV3)
    DSLV = dvs_mask * DSLV

    # Determine how much leaf biomass classes have to die because their age
    # exceeds SPAN; their living biomass is accumulated in DALV.
    # Note that the actual leaf death is imposed on the leaf classes during
    # the state integration step.
    # Leaf age decreases with the class index, so the classes older than SPAN
    # are the first n_older ones, and their living biomass is
    # LVCUM[n_older - 1] - DLVCUM.
    LV_older = _take(LVCUM, torch.clamp(n_older - 1, min=0))
    DALV = torch.where(n_older > 0, torch.clamp(LV_older - DLVCUM, min=0.0), zero)
    if DALV_soft is not None:
        DALV = DALV + DALV_soft
    DALV = dvs_mask * DALV

    # Total death rate leaves
    DRLV = torch.maximum(DSLV, DALV)

    # physiologic ageing of leaves per time step
    FYSAGE = (TEMP - TBASE) / (35.0 - TBASE)
    FYSAGE = dvs_mask * torch.clamp(FYSAGE, 0.0)

    # specific leaf area of leaves per time step
    SLAT = dvs_mask * SLATB(DVS)

    # leaf area not to exceed exponential growth curve
    is_lai_exp = LAIEXP < 6.0
    DTEFF = torch.clamp(TEMP - TBASE, 0.0)

    # NOTE: conditional statements do not allow for the gradient to be
    # tracked through the condition. Thus, the gradient with respect to
    # parameters that contribute to `is_lai_exp` (e.g. RGRLAI and TBASE)
    # are expected to be incorrect.

    GLAIEX = torch.where(
        dvs_mask,
        torch.where(is_lai_exp, LAIEXP * RGRLAI * DTEFF, GLAIEX_old),
        zero,
    )

    # source-limited increase in leaf area
    GLASOL = torch.where(
        dvs_mask,
        torch.where(is_lai_exp, GRLV * SLAT, GLASOL_old),
        zero,
    )

    # sink-limited increase in leaf area
    GLA = torch.minimum(GLAIEX, GLASOL)

    # adjustment of specific leaf area of youngest leaf class
    SLAT = torch.where(
        dvs_mask,
        torch.where(is_lai_exp & (GRLV > epsilon), GLA / (GRLV + epsilon), SLAT),
        zero,
    )
    return GRLV, DSLV1, DSLV2, DSLV3, DSLV, DALV, DRLV, FYSAGE, SLAT, GLAIEX, GLASOL


@kernel
def _leaf_death_and_growth(LV_formed, LA_formed, DLVCUM, DRLV, GRLV, SLAT):
    """Leaf death and the new leaf class, see `WOFOST_Leaf_Dynamics.integrate`.

    `LV_formed` and `LA_formed` are the running totals of the youngest leaf class.
    """
    # --------- leave death ---------
    # Leaf death is imposed on leaves from the oldest ones: the living
    # biomass is the formed biomass minus DLVCUM, at most all of it.
    # NOTE: the gradient is not tracked through the choice of the oldest
    # living class.
    DLVCUM = torch.minimum(DLVCUM + DRLV, LV_formed)

    # --------- leave growth ---------
    return DLVCUM, LV_formed + GRLV, LA_formed + GRLV * SLAT


@kernel(dynamic=None)
def _leaf_ageing(LVAGE, FYSAGE, SPAN):
    """Age the leaf classes `LVAGE` in place; return the number older than SPAN.

    The ages are accumulated per class, as in PCSE, so that the comparison with
    SPAN gives the same result. Older classes have summed more of the same
    increments, so age decreases with the class index and the classes older
    than SPAN are the first ones. (Classes that have died keep ageing, which
    does not matter because they have no living biomass.)
    """
    LVAGE.add_(FYSAGE)
    return (LVAGE > SPAN).sum(dim=0)


@kernel
def _leaf_area_and_weight(
    LVCUM, LACUM, SLA, DLVCUM, LV_formed, LA_formed, SAI, PAI, LAIMAX, LAIEXP, DWLV, GLAIEX, DRLV
):
    """Leaf area and weight after integration, see `WOFOST_Leaf_Dynamics.integrate`."""
    # calculation of new leaf area: all classes younger than the oldest
    # living one plus the living part of that one
    oldest = _search(LVCUM, DLVCUM)
    LV_oldest = _take(LVCUM, oldest) - DLVCUM
    LA_older = _take(LACUM, oldest)
    LASUM = (LA_formed - LA_older) + LV_oldest * _take(SLA, oldest)

    # Total leaf area Index as sum of leaf, pod and stem area
    LAI = LASUM + SAI + PAI
    LAIMAX = torch.maximum(LAI, LAIMAX)

    # exponential growth curve
    LAIEXP = LAIEXP + GLAIEX

    # Update leaf biomass states
    WLV = LV_formed - DLVCUM
    DWLV = DWLV + DRLV
    TWLV = WLV + DWLV
    return LASUM, LAI, LAIMAX, LAIEXP, WLV, DWLV, TWLV


class _TakeClass(torch.autograd.Function):
    """Gather along the leaf-class axis without saving the source for the backward pass.

    The leaf-class arrays are updated in place every day. torch.gather saves its
    source for the backward pass, which autograd then refuses after the update;
    the backward of a gather only needs the indices.
    """

    @staticmethod
    def forward(ctx, src, index):
        ctx.save_for_backward(index)
        ctx.src_shape = src.shape
        return torch.gather(src, 0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        grad_src = torch.zeros(ctx.src_shape, dtype=grad.dtype, device=grad.device)
        return grad_src.scatter_add_(0, index, grad), None


def _take(src, index):
    """Return src[index] per element, for leaf-class arrays (leaf class on axis 0)."""
    index = index.expand(src.shape[1:]).unsqueeze(0)
    if torch.is_grad_enabled() and src.requires_grad:
        return _TakeClass.apply(src, index).squeeze(0)
    return torch.gather(src, 0, index).squeeze(0)


def _search(sorted_seq, values, max_index=None):
    """Number of leaf classes in sorted_seq (axis 0, ascending) smaller than values.

    This is also the index of the first leaf class not smaller than values; it is
    limited to `max_index` (by default the last valid index). Leaf classes not formed
    yet hold +inf. The binary search has a fixed number of steps, so that it fuses
    into a single kernel when compiled; neighbouring elements in a similar
    development stage probe the same leaf classes.
    """
    n = sorted_seq.shape[0]
    if max_index is None:
        max_index = n - 1
    sorted_seq = sorted_seq.detach()
    values = values.detach().expand(sorted_seq.shape[1:])
    lo = torch.zeros(values.shape, dtype=torch.long, device=sorted_seq.device)
    hi = torch.full(values.shape, n, dtype=torch.long, device=sorted_seq.device)
    for _ in range(n.bit_length()):
        mid = (lo + hi) // 2
        probe = torch.gather(sorted_seq, 0, torch.clamp(mid, max=n - 1).unsqueeze(0)).squeeze(0)
        active = lo < hi
        right = active & (probe < values)
        lo = torch.where(right, mid + 1, lo)
        hi = torch.where(active & ~right, mid, hi)
    return torch.clamp(lo, max=max_index)


def _row(src, index):
    """Leaf class `index` of src; a copy if autograd records.

    src is updated in place later, and operations such as torch.minimum save their
    inputs for the backward pass even when those do not require gradients.
    """
    if torch.is_grad_enabled():
        return src[index].clone()
    return src[index]


def _set(dst, index, value):
    """Set the leaf class `index` of dst in place."""
    dst[index] = value


def _exist_required_external_variables(kiosk):
    """Check if all required external variables are available in the kiosk.

    Args:
        kiosk (VariableKiosk): The variable kiosk to check.

    Raises:
        ValueError: If any required external variable is missing.

    """
    required_external_vars_at_init = ["DVS", "FL", "FR", "SAI", "PAI"]
    for var in required_external_vars_at_init:
        if var not in kiosk:
            raise ValueError(
                f"Required external variables '{var}' is missing in the kiosk."
                f" Ensure that all required variables {required_external_vars_at_init}"
                " are provided."
            )
