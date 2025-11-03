"""Leaf dynamics for the WOFOST crop model."""

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
from diffwofost.physical_models.afgen import Afgen
from diffwofost.physical_models.afgen import AfgenTrait

DTYPE = torch.float64  # Default data type for tensors in this module


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
    """  # noqa: E501

    # The following parameters are used to initialize and control the arrays that store information
    # on the leaf classes during the time integration: leaf area, age, and biomass.
    START_DATE = None  # Start date of the simulation
    MAX_DAYS = 300  # Maximum number of days that can be simulated in one run (i.e. array lenghts)

    class Parameters(ParamTemplate):
        RGRLAI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        SPAN = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        TBASE = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        PERDL = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        TDWI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        SLATB = AfgenTrait()  # FIXME
        KDIFTB = AfgenTrait()  # FIXME

    class StateVariables(StatesTemplate):
        LV = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        SLA = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        LVAGE = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        LAIEM = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        LASUM = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        LAIEXP = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        LAIMAX = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        LAI = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        WLV = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        DWLV = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))
        TWLV = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))

    class RateVariables(RatesTemplate):
        GRLV = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DSLV1 = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DSLV2 = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DSLV3 = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DSLV = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DALV = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        DRLV = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        SLAT = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        FYSAGE = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        GLAIEX = Any(default_value=torch.tensor(0.0, dtype=DTYPE))
        GLASOL = Any(default_value=torch.tensor(0.0, dtype=DTYPE))

    def initialize(
        self, day: datetime.date, kiosk: VariableKiosk, parvalues: ParameterProvider
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
        """
        self.START_DATE = day
        self.kiosk = kiosk
        # TODO check if parvalues are already torch.nn.Parameters
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk)

        # CALCULATE INITIAL STATE VARIABLES
        # check for required external variables
        _exist_required_external_variables(self.kiosk)
        # TODO check if external variables are already torch tensors

        FL = self.kiosk["FL"]
        FR = self.kiosk["FR"]
        DVS = self.kiosk["DVS"]

        params = self.params
        shape = _get_params_shape(params)

        # Initial leaf biomass
        WLV = (params.TDWI * (1 - FR)) * FL
        DWLV = torch.zeros(shape, dtype=DTYPE)
        TWLV = WLV + DWLV

        # Initialize leaf classes (SLA, age and weight)
        SLA = torch.zeros((*shape, self.MAX_DAYS), dtype=DTYPE)
        LVAGE = torch.zeros((*shape, self.MAX_DAYS), dtype=DTYPE)
        LV = torch.zeros((*shape, self.MAX_DAYS), dtype=DTYPE)
        SLA[..., 0] = params.SLATB(DVS)
        LV[..., 0] = WLV

        # Initial values for leaf area
        LAIEM = LV[..., 0] * SLA[..., 0]
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        LAI = LASUM + self.kiosk["SAI"] + self.kiosk["PAI"]

        # Initialize StateVariables object
        self.states = self.StateVariables(
            kiosk,
            publish=["LAI", "TWLV", "WLV"],
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
        # Make a mask (0 if DVS < 0, 1 if DVS >= 0)
        DVS = torch.as_tensor(k["DVS"], dtype=DTYPE)
        mask = (DVS >= 0).to(dtype=DTYPE)

        # Growth rate leaves
        # weight of new leaves
        r.GRLV = mask * k.ADMI * k.FL

        # death of leaves due to water/oxygen stress
        r.DSLV1 = mask * s.WLV * (1.0 - k.RFTRA) * p.PERDL

        # death due to self shading cause by high LAI
        DVS = self.kiosk["DVS"]
        LAICR = 3.2 / p.KDIFTB(DVS)
        r.DSLV2 = mask * s.WLV * torch.clamp(0.03 * (s.LAI - LAICR) / LAICR, 0.0, 0.03)

        # Death of leaves due to frost damage as determined by
        # Reduction Factor Frost "RF_FROST"
        if "RF_FROST" in self.kiosk:
            r.DSLV3 = mask * s.WLV * k.RF_FROST
        else:
            r.DSLV3 = torch.zeros_like(s.WLV, dtype=DTYPE)

        # leaf death equals maximum of water stress, shading and frost
        r.DSLV = torch.maximum(torch.maximum(r.DSLV1, r.DSLV2), r.DSLV3)

        # Determine how much leaf biomass classes have to die in states.LV,
        # given the a life span > SPAN, these classes will be accumulated
        # in DALV.
        # Note that the actual leaf death is imposed on the array LV during the
        # state integration step.
        tSPAN = _broadcast_to(p.SPAN, s.LVAGE.shape)  # Broadcast to same shape
        # Using a sigmoid here instead of a conditional statement on the value of
        # SPAN because the latter would not allow for the gradient to be tracked.
        sharpness = torch.tensor(1000.0, dtype=DTYPE)  # FIXME
        weight = torch.sigmoid((s.LVAGE - tSPAN) * sharpness)
        r.DALV = torch.sum(weight * s.LV, dim=-1)

        # Total death rate leaves
        r.DRLV = torch.maximum(r.DSLV, r.DALV)

        # physiologic ageing of leaves per time step
        FYSAGE = (drv.TEMP - p.TBASE) / (35.0 - p.TBASE)
        r.FYSAGE = mask * torch.clamp(FYSAGE, 0.0)

        # specific leaf area of leaves per time step
        r.SLAT = mask * torch.tensor(p.SLATB(DVS), dtype=DTYPE)

        # leaf area not to exceed exponential growth curve
        is_lai_exp = s.LAIEXP < 6.0
        DTEFF = torch.clamp(drv.TEMP - p.TBASE, 0.0)
        # NOTE: conditional statements do not allow for the gradient to be
        # tracked through the condition. Thus, the gradient with respect to
        # parameters that contribute to `is_lai_exp` (e.g. RGRLAI and TBASE)
        # are expected to be incorrect.
        r.GLAIEX = torch.where(is_lai_exp, s.LAIEXP * p.RGRLAI * DTEFF, r.GLAIEX)
        # source-limited increase in leaf area
        r.GLASOL = torch.where(is_lai_exp, r.GRLV * r.SLAT, r.GLASOL)
        # sink-limited increase in leaf area
        GLA = torch.minimum(r.GLAIEX, r.GLASOL)
        # adjustment of specific leaf area of youngest leaf class
        r.SLAT = torch.where(is_lai_exp & (r.GRLV > 0.0), GLA / r.GRLV, r.SLAT)

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
        tDRLV = _broadcast_to(rates.DRLV, tLV.shape)

        # Leaf death is imposed on leaves from the oldest ones.
        # Calculate the cumulative sum of weights after leaf death, and
        # find out which leaf classes are dead (negative weights)
        weight_cumsum = tLV.cumsum(dim=-1) - tDRLV
        is_alive = weight_cumsum >= 0

        # Adjust value of oldest leaf class, i.e. the first non-zero
        # weight along the time axis (the last dimension).
        # Cast argument to int because torch.argmax requires it to be numeric
        idx_oldest = torch.argmax(is_alive.type(torch.int), dim=-1, keepdim=True)
        new_biomass = torch.take_along_dim(weight_cumsum, indices=idx_oldest, dim=-1)
        tLV = torch.scatter(tLV, dim=-1, index=idx_oldest, src=new_biomass)

        # Zero out all dead leaf classes
        # NOTE: conditional statements do not allow for the gradient to be
        # tracked through the condition. Thus, the gradient with respect to
        # parameters that contribute to `is_alive` are expected to be incorrect.
        tLV = torch.where(is_alive, tLV, 0.0)

        # Integration of physiological age
        tLVAGE = tLVAGE + rates.FYSAGE
        tLVAGE = torch.where(is_alive, tLVAGE, 0.0)
        tSLA = torch.where(is_alive, tSLA, 0.0)

        # --------- leave growth ---------
        idx = int((day - self.START_DATE).days / delt)
        tLV[..., idx] = rates.GRLV
        tSLA[..., idx] = rates.SLAT
        tLVAGE[..., idx] = 0.0

        # calculation of new leaf area
        states.LASUM = torch.sum(tLV * tSLA, dim=-1)
        states.LAI = self._calc_LAI()
        states.LAIMAX = torch.maximum(states.LAI, states.LAIMAX)

        # exponential growth curve
        states.LAIEXP = states.LAIEXP + rates.GLAIEX

        # Update leaf biomass states
        states.WLV = torch.sum(tLV, dim=-1)
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


def _get_params_shape(params):
    """Get the parameters shape.

    Parameters can have arbitrary number of dimensions, but all parameters that are not zero-
    dimensional should have the same shape.
    """
    shape = ()
    for parname in params.trait_names():
        # Skip special traitlets attributes
        if parname.startswith("trait"):
            continue
        param = getattr(params, parname)
        # Skip Afgen parameters:
        if isinstance(param, Afgen):
            continue
        # Parameters that are not zero dimensional should all have the same shape
        if param.shape and not shape:
            shape = param.shape
        elif param.shape:
            assert param.shape == shape, (
                "All parameters should have the same shape (or have no dimensions)"
            )
    return shape


def _broadcast_to(x, shape):
    """Create a view of tensor X with the given shape."""
    if x.dim() == 0:
        # For 0-d tensors, we simply broadcast to the given shape
        return torch.broadcast_to(x, shape)
    # The given shape should match x in all but the last axis, which represents
    # the dimension along which the time integration is carried out.
    # We first append an axis to x, then expand to the given shape
    return x.unsqueeze(-1).expand(shape)
