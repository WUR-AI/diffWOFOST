"""Leaf dynamics for the WOFOST crop model."""

import torch
from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import SimulationObject
from pcse.base import StatesTemplate
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Any
from pcse.util import AfgenTrait

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

    *Simulation parameters* (provide in cropdata dictionary)

    =======  ============================================= =======  ============
     Name     Description                                   Type     Unit
    =======  ============================================= =======  ============
    RGRLAI   Maximum relative increase in LAI.              SCr     ha ha-1 d-1
    SPAN     Life span of leaves growing at 35 Celsius      SCr     |d|
    TBASE    Lower threshold temp. for ageing of leaves     SCr     |C|
    PERDL    Max. relative death rate of leaves due to      SCr
             water stress
    TDWI     Initial total crop dry weight                  SCr     |kg ha-1|
    KDIFTB   Extinction coefficient for diffuse visible     TCr
             light as function of DVS
    SLATB    Specific leaf area as a function of DVS        TCr     |ha kg-1|
    =======  ============================================= =======  ============

    *State variables*

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    LV       Leaf biomass per leaf class                        N    |kg ha-1|
    SLA      Specific leaf area per leaf class                  N    |ha kg-1|
    LVAGE    Leaf age per leaf class                            N    |d|
    LVSUM    Sum of LV                                          N    |kg ha-1|
    LAIEM    LAI at emergence                                   N    -
    LASUM    Total leaf area as sum of LV*SLA,                  N    -
             not including stem and pod area                    N
    LAIEXP   LAI value under theoretical exponential growth     N    -
    LAIMAX   Maximum LAI reached during growth cycle            N    -
    LAI      Leaf area index, including stem and pod area       Y    -
    WLV      Dry weight of living leaves                        Y    |kg ha-1|
    DWLV     Dry weight of dead leaves                          N    |kg ha-1|
    TWLV     Dry weight of total leaves (living + dead)         Y    |kg ha-1|
    =======  ================================================= ==== ============


    *Rate variables*

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    GRLV     Growth rate leaves                                 N   |kg ha-1 d-1|
    DSLV1    Death rate leaves due to water stress              N   |kg ha-1 d-1|
    DSLV2    Death rate leaves due to self-shading              N   |kg ha-1 d-1|
    DSLV3    Death rate leaves due to frost kill                N   |kg ha-1 d-1|
    DSLV     Maximum of DLSV1, DSLV2, DSLV3                     N   |kg ha-1 d-1|
    DALV     Death rate leaves due to aging.                    N   |kg ha-1 d-1|
    DRLV     Death rate leaves as a combination of DSLV and     N   |kg ha-1 d-1|
             DALV
    SLAT     Specific leaf area for current time step,          N   |ha kg-1|
             adjusted for source/sink limited leaf expansion
             rate.
    FYSAGE   Increase in physiological leaf age                 N   -
    GLAIEX   Sink-limited leaf expansion rate (exponential      N   |ha ha-1 d-1|
             curve)
    GLASOL   Source-limited leaf expansion rate (biomass        N   |ha ha-1 d-1|
             increase)
    =======  ================================================= ==== ============


    *External dependencies:*

    ======== ============================== =============================== ===========
     Name     Description                         Provided by               Unit
    ======== ============================== =============================== ===========
    DVS      Crop development stage         DVS_Phenology                    -
    FL       Fraction biomass to leaves     DVS_Partitioning                 -
    FR       Fraction biomass to roots      DVS_Partitioning                 -
    SAI      Stem area index                WOFOST_Stem_Dynamics             -
    PAI      Pod area index                 WOFOST_Storage_Organ_Dynamics    -
    TRA      Transpiration rate             Evapotranspiration              |cm day-1| ?
    TRAMX    Maximum transpiration rate     Evapotranspiration              |cm day-1| ?
    ADMI     Above-ground dry matter        CropSimulation                  |kg ha-1 d-1|
             increase
    RFTRA    Reduction factor for               Y                            -
             transpiration (wat & ox)
    RF_FROST Reduction factor frost kill    FROSTOL(optional)                -
    ======== ============================== =============================== ===========

    *Outputs:*
    LAI, TWLV
    """

    START_DATE = None
    MAX_DAYS = 300

    class Parameters(ParamTemplate):
        RGRLAI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        SPAN = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        TBASE = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        PERDL = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        TDWI = Any(default_value=[torch.tensor(-99.0, dtype=DTYPE)])
        SLATB = AfgenTrait()  # FIXEME
        KDIFTB = AfgenTrait()  # FIXEME

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

    def initialize(self, day, kiosk, parvalues):
        """Initialize the WOFOST_Leaf_Dynamics simulation object.

        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
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

        # Initial leaf biomass
        WLV = (params.TDWI * (1 - FR)) * FL
        DWLV = torch.tensor(0.0, dtype=DTYPE)
        TWLV = WLV + DWLV

        # Initialize leaf classes (SLA, age and weight)
        SLA = torch.zeros(self.MAX_DAYS, dtype=DTYPE)
        LVAGE = torch.zeros(self.MAX_DAYS, dtype=DTYPE)
        LV = torch.zeros(self.MAX_DAYS, dtype=DTYPE)
        SLA[0] = params.SLATB(DVS)
        LV[0] = WLV

        # Initial values for leaf area
        LAIEM = LV[0] * SLA[0]
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
    def calc_rates(self, day, drv):
        """Calculate the rates of change for the leaf dynamics."""
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
        r.DSLV = torch.max(torch.stack([r.DSLV1, r.DSLV2, r.DSLV3]))

        # Determine how much leaf biomass classes have to die in states.LV,
        # given the a life span > SPAN, these classes will be accumulated
        # in DALV.
        # Note that the actual leaf death is imposed on the array LV during the
        # state integration step.
        sharpness = torch.tensor(1000., dtype=DTYPE)  # FIXME
        weight = torch.sigmoid((s.LVAGE - p.SPAN) * sharpness)
        r.DALV = torch.sum(weight * s.LV)

        # Total death rate leaves
        r.DRLV = torch.max(r.DSLV, r.DALV)

        # physiologic ageing of leaves per time step
        FYSAGE = (drv.TEMP - p.TBASE) / (35.0 - p.TBASE)
        r.FYSAGE = mask * torch.max(torch.tensor(0.0, dtype=DTYPE), FYSAGE)

        # specific leaf area of leaves per time step
        r.SLAT = mask * torch.tensor(p.SLATB(DVS), dtype=DTYPE)

        # leaf area not to exceed exponential growth curve
        if s.LAIEXP < 6.0:
            DTEFF = torch.max(torch.tensor(0.0, dtype=DTYPE), drv.TEMP - p.TBASE)
            r.GLAIEX = s.LAIEXP * p.RGRLAI * DTEFF
            # source-limited increase in leaf area
            r.GLASOL = r.GRLV * r.SLAT
            # sink-limited increase in leaf area
            GLA = torch.min(r.GLAIEX, r.GLASOL)
            # adjustment of specific leaf area of youngest leaf class
            if r.GRLV > 0.0:
                r.SLAT = GLA / r.GRLV

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Integrate the leaf dynamics state variables."""
        # TODO check if DVS < 0 and skip integration needed
        rates = self.rates
        states = self.states

        # --------- leave death ---------
        tLV = states.LV.clone()
        tSLA = states.SLA.clone()
        tLVAGE = states.LVAGE.clone()
        tDRLV = rates.DRLV

        # Leaf death is imposed on leaves from the oldest ones.
        # Calculate the cumulative sum of weights after leaf death, and
        # find out which leaf classes are dead (negative weights)
        weight_cumsum = tLV.cumsum(0) - tDRLV
        is_dead = weight_cumsum < 0
        # Adjust value of oldest leaf class (first non-zero weights)
        idx_alive, = (~is_dead).nonzero(as_tuple=True)
        idx_oldest = idx_alive[0]
        tLV[idx_oldest] = weight_cumsum[idx_oldest]
        # Zero out all dead leaf classes
        tLV = tLV.where(~is_dead, 0.)
        # Integration of physiological age
        tLVAGE = tLVAGE + rates.FYSAGE
        tLVAGE = tLVAGE.where(~is_dead, 0.)
        tSLA = tSLA.where(~is_dead, 0.)

        # --------- leave growth ---------
        idx = int((day - self.START_DATE).days / delt)
        tLV[idx] = rates.GRLV
        tSLA[idx] = rates.SLAT
        tLVAGE[idx] = 0.

        # calculation of new leaf area
        states.LASUM = torch.sum(tLV * tSLA)
        states.LAI = self._calc_LAI()
        states.LAIMAX = torch.max(states.LAI, states.LAIMAX)

        # exponential growth curve
        states.LAIEXP = states.LAIEXP + rates.GLAIEX

        # Update leaf biomass states
        states.WLV = torch.sum(tLV)
        states.DWLV = states.DWLV + rates.DRLV
        states.TWLV = states.WLV + states.DWLV

        # Store final leaf biomass deques
        self.states.LV = tLV
        self.states.SLA = tSLA
        self.states.LVAGE = tLVAGE

    @prepare_states
    def _set_variable_LAI(self, nLAI):  # FIXEME
        """Updates the value of LAI to to the new value provided as input.

        Related state variables will be updated as well and the increments
        to all adjusted state variables will be returned as a dict.
        """
        states = self.states

        # Store old values of states
        oWLV = states.WLV
        oLAI = states.LAI
        oTWLV = states.TWLV
        oLASUM = states.LASUM

        # Reduce oLAI for pod and stem area. SAI and PAI will not be adjusted
        # because this is often only a small component of the total leaf
        # area. For all current crop files in WOFOST SPA and SSA are zero
        # anyway
        SAI = self.kiosk["SAI"]
        PAI = self.kiosk["PAI"]
        adj_nLAI = torch.max(nLAI - SAI - PAI, 0.0)
        adj_oLAI = torch.max(oLAI - SAI - PAI, 0.0)

        # LAI Adjustment factor for leaf biomass LV (rLAI)
        if adj_oLAI > 0:
            rLAI = adj_nLAI / adj_oLAI
            LV = [lv * rLAI for lv in states.LV]
        # If adj_oLAI == 0 then add the leave biomass directly to the
        # youngest leave age class (LV[0])
        else:
            LV = [nLAI / states.SLA[0]]

        states.LASUM = torch.sum(
            torch.tensor([lv * sla for lv, sla in zip(LV, states.SLA, strict=False)], dtype=DTYPE)
        )
        states.LV = LV
        states.LAI = self._calc_LAI()
        states.WLV = torch.sum(states.LV)
        states.TWLV = states.WLV + states.DWLV

        increments = {
            "LAI": states.LAI - oLAI,
            "LAISUM": states.LASUM - oLASUM,
            "WLV": states.WLV - oWLV,
            "TWLV": states.TWLV - oTWLV,
        }
        return increments


def _exist_required_external_variables(kiosk):
    """Check if all required external variables are available in the kiosk."""
    required_external_vars_at_init = ["DVS", "FL", "FR", "SAI", "PAI"]
    for var in required_external_vars_at_init:
        if var not in kiosk:
            raise ValueError(
                f"Required external variables '{var}' is missing in the kiosk."
                f" Ensure that all required variables {required_external_vars_at_init}"
                " are provided."
            )
