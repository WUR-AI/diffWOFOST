"""Leaf dynamics for the WOFOST crop model."""

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
from diffwofost.physical_models.utils import _get_drv


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
    | LV     | Leaf biomass per leaf class                           | N    | kg ha⁻¹     |
    | SLA    | Specific leaf area per leaf class                     | N    | ha kg⁻¹     |
    | LVAGE  | Leaf age per leaf class                               | N    | d           |
    | LVSUM  | Sum of LV                                             | N    | kg ha⁻¹     |
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
        LV = Tensor(-99.0)
        SLA = Tensor(-99.0)
        LVAGE = Tensor(-99.0)
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

        # Initialize leaf classes (SLA, age and weight)
        SLA = torch.zeros((self.MAX_DAYS, *params.shape), dtype=self.dtype, device=self.device)
        LVAGE = torch.zeros((self.MAX_DAYS, *params.shape), dtype=self.dtype, device=self.device)
        LV = torch.zeros((self.MAX_DAYS, *params.shape), dtype=self.dtype, device=self.device)
        SLA[0, ...] = params.SLATB(self.kiosk["DVS"])
        LV[0, ...] = WLV

        # Initial values for leaf area
        LAIEM = LV[0, ...] * SLA[0, ...]
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        LAI = LASUM + self.kiosk["SAI"] + self.kiosk["PAI"]

        # Initialize StateVariables object
        self.states = self.StateVariables(
            kiosk,
            publish=["LAI", "TWLV", "WLV"],
            do_not_broadcast=["SLA", "LVAGE", "LV"],
            LV=LV,
            SLA=SLA,
            LVAGE=LVAGE,
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

    def _calc_LAI(self):
        # Total leaf area Index as sum of leaf, pod and stem area
        SAI = self.kiosk["SAI"]
        PAI = self.kiosk["PAI"]
        total_LAI = self.states.LASUM + SAI + PAI
        return total_LAI

    @prepare_rates
    def calc_rates(self, day: datetime.date, drv: WeatherDataContainer) -> None:
        """Calculate the rates of change for the leaf dynamics.

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

        # If DVS < 0, the crop has not yet emerged, so we zerofy the rates using mask
        # A mask (0 if DVS < 0, 1 if DVS >= 0)
        dvs_mask = k["DVS"] >= self._zero

        # Growth rate leaves
        # weight of new leaves
        r.GRLV = dvs_mask * k.ADMI * k.FL

        # death of leaves due to water/oxygen stress
        r.DSLV1 = dvs_mask * s.WLV * (1.0 - k.RFTRA) * p.PERDL

        # death due to self shading cause by high LAI
        LAICR = 3.2 / p.KDIFTB(k["DVS"])
        r.DSLV2 = dvs_mask * s.WLV * torch.clamp(0.03 * (s.LAI - LAICR) / LAICR, 0.0, 0.03)

        # Death of leaves due to frost damage as determined by
        # Reduction Factor Frost "RF_FROST"
        if "RF_FROST" in self.kiosk:
            r.DSLV3 = s.WLV * k.RF_FROST
        else:
            r.DSLV3 = torch.zeros_like(s.WLV)

        r.DSLV3 = dvs_mask * r.DSLV3

        # leaf death equals maximum of water stress, shading and frost
        r.DSLV = torch.maximum(torch.maximum(r.DSLV1, r.DSLV2), r.DSLV3)
        r.DSLV = dvs_mask * r.DSLV

        # Determine how much leaf biomass classes have to die in states.LV,
        # given the a life span > SPAN, these classes will be accumulated
        # in DALV.
        # Note that the actual leaf death is imposed on the array LV during the
        # state integration step.

        # Using a sigmoid here instead of a conditional statement on the value of
        # SPAN because the latter would not allow for the gradient to be tracked.
        # the if statement `p.SPAN.requires_grad` to avoid unnecessary
        # approximation when SPAN is not a learnable parameter.
        # here we use STE (straight through estimator) method.
        # TODO: sharpness can be exposed as a parameter
        if p.SPAN.requires_grad:
            # soft mask using sigmoid
            soft_mask = torch.sigmoid(
                (s.LVAGE - p.SPAN - self._sigmoid_epsilon) / self._sigmoid_sharpness
            )

            # originial hard mask
            hard_mask = s.LVAGE > p.SPAN

            # STE method. Here detach is used to stop the gradient flow. This
            # way, during backpropagation, the gradient is computed only through
            # the `soft_mask``, while during the forward pass, the `hard_mask``
            # is used.
            span_mask = hard_mask.detach() + soft_mask - soft_mask.detach()
        else:
            span_mask = s.LVAGE > p.SPAN

        r.DALV = torch.sum(span_mask * s.LV, dim=0)
        r.DALV = dvs_mask * r.DALV

        # Total death rate leaves
        r.DRLV = torch.maximum(r.DSLV, r.DALV)

        # Get the temperature from the drv
        TEMP = _get_drv(drv.TEMP, p.shape, self.dtype, self.device)

        # physiologic ageing of leaves per time step
        FYSAGE = (TEMP - p.TBASE) / (35.0 - p.TBASE)
        r.FYSAGE = dvs_mask * torch.clamp(FYSAGE, 0.0)

        # specific leaf area of leaves per time step
        r.SLAT = dvs_mask * p.SLATB(k["DVS"])

        # leaf area not to exceed exponential growth curve
        is_lai_exp = s.LAIEXP < 6.0
        DTEFF = torch.clamp(TEMP - p.TBASE, 0.0)

        # NOTE: conditional statements do not allow for the gradient to be
        # tracked through the condition. Thus, the gradient with respect to
        # parameters that contribute to `is_lai_exp` (e.g. RGRLAI and TBASE)
        # are expected to be incorrect.

        r.GLAIEX = torch.where(
            dvs_mask,
            torch.where(is_lai_exp, s.LAIEXP * p.RGRLAI * DTEFF, r.GLAIEX),
            self._zero,
        )

        # source-limited increase in leaf area
        r.GLASOL = torch.where(
            dvs_mask,
            torch.where(is_lai_exp, r.GRLV * r.SLAT, r.GLASOL),
            self._zero,
        )

        # sink-limited increase in leaf area
        GLA = torch.minimum(r.GLAIEX, r.GLASOL)

        # adjustment of specific leaf area of youngest leaf class
        r.SLAT = torch.where(
            dvs_mask,
            torch.where(
                is_lai_exp & (r.GRLV > self._epsilon), GLA / (r.GRLV + self._epsilon), r.SLAT
            ),
            self._zero,
        )

    @prepare_states
    def integrate(self, day: datetime.date, delt=1.0) -> None:
        """Integrate the leaf dynamics state variables.

        Args:
            day (datetime.date, optional): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        # TODO check if DVS < 0 and skip integration needed
        rates = self.rates
        states = self.states

        # --------- leave death ---------
        tLV = states.LV.clone()
        tSLA = states.SLA.clone()
        tLVAGE = states.LVAGE.clone()

        # Leaf death is imposed on leaves from the oldest ones.
        # Calculate the cumulative sum of weights after leaf death, and
        # find out which leaf classes are dead (negative weights)
        weight_cumsum = tLV.cumsum(dim=0) - rates.DRLV
        is_alive = weight_cumsum >= 0

        # Adjust value of oldest leaf class, i.e. the first non-zero
        # weight along the time axis (the last dimension).
        # Cast argument to int because torch.argmax requires it to be numeric
        idx_oldest = torch.argmax(is_alive.type(torch.int), dim=0, keepdim=True)
        new_biomass = torch.take_along_dim(weight_cumsum, indices=idx_oldest, dim=0)
        tLV = torch.scatter(tLV, dim=0, index=idx_oldest, src=new_biomass)

        # Integration of physiological age
        # Zero out all dead leaf classes
        # NOTE: conditional statements do not allow for the gradient to be
        # tracked through the condition. Thus, the gradient with respect to
        # parameters that contribute to `is_alive` are expected to be incorrect.
        tLV = torch.where(is_alive, tLV, 0.0)
        tLVAGE = tLVAGE + rates.FYSAGE
        tLVAGE = torch.where(is_alive, tLVAGE, 0.0)
        tSLA = torch.where(is_alive, tSLA, 0.0)

        # --------- leave growth ---------
        idx = int((day - self.START_DATE).days / delt)
        tLV[idx, ...] = rates.GRLV
        tSLA[idx, ...] = rates.SLAT
        tLVAGE[idx, ...] = 0.0

        # calculation of new leaf area
        states.LASUM = torch.sum(tLV * tSLA, dim=0)
        states.LAI = self._calc_LAI()
        states.LAIMAX = torch.maximum(states.LAI, states.LAIMAX)

        # exponential growth curve
        states.LAIEXP = states.LAIEXP + rates.GLAIEX

        # Update leaf biomass states
        states.WLV = torch.sum(tLV, dim=0)
        states.DWLV = states.DWLV + rates.DRLV
        states.TWLV = states.WLV + states.DWLV

        # Store final leaf biomass deques
        self.states.LV = tLV
        self.states.SLA = tSLA
        self.states.LVAGE = tLVAGE


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
