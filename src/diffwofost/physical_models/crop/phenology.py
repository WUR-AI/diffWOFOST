"""Phenological development and vernalisation models for WOFOST.

This module implements:
- Vernalisation: modification of phenological development due to cold
exposure.
- DVS_Phenology: main phenology progression (DVS scale: 0 emergence, 1
anthesis, 2 maturity).
"""

import datetime
import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import ParamTemplate
from pcse.base import RatesTemplate
from pcse.base import SimulationObject
from pcse.base import StatesTemplate
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Any
from pcse.traitlets import Bool
from pcse.traitlets import Enum
from pcse.traitlets import Instance
from pcse.util import daylength
from pcse.util import limit
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to  # added
from diffwofost.physical_models.utils import _get_params_shape  # added

DTYPE = torch.float64  # Default data type for tensors in this module


def tclamp(x, low, high):
    """Clamp tensor or scalar between low and high.

    [!] These function can be removed once we fully switch to torch tensors.

    Uses torch.clamp for tensors; falls back to pcse.util.limit for Python scalars.

    Args:
        x: torch.Tensor or scalar.
        low: lower bound.
        high: upper bound.

    Returns:
        Clamped value with same type as input.
    """
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, low, high)
    # fallback for scalar
    return limit(low, high, x)


class Vernalisation(SimulationObject):
    """Modification of phenological development due to vernalisation.

    The vernalization approach here is based on the work of Lenny van
    Bussel (2011), which in turn is based on Wang and Engel (1998). The
    basic principle is that winter wheat needs a certain number of days
    with temperatures within an optimum temperature range to complete
    its vernalisation requirement. Until the vernalisation requirement
    is fulfilled, the crop development is delayed.

    The rate of vernalization (VERNR) is defined by the temperature
    response function VERNRTB. Within the optimal temperature range 1
    day is added to the vernalisation state (VERN). The reduction on the
    phenological development is calculated from the base and saturated
    vernalisation requirements (VERNBASE and VERNSAT). The reduction
    factor (VERNFAC) is scaled linearly between VERNBASE and VERNSAT.

    A critical development stage (VERNDVS) is used to stop the effect of
    vernalisation when this DVS is reached. This is done to improve
    model stability in order to avoid that Anthesis is never reached
    due to a somewhat too high VERNSAT. Nevertheless, a warning is
    written to the log file, if this happens.

    * Van Bussel, 2011. From field to globe: Upscaling of crop growth
      modelling. Wageningen PhD thesis. http://edepot.wur.nl/180295
    * Wang and Engel, 1998. Simulation of phenological development of
      wheat crops. Agric. Systems 58:1 pp 1-24

    *Simulation parameters* (provide in cropdata dictionary)

    | Name     | Description                                                   | Type | Unit |
    |----------|---------------------------------------------------------------|------|------|
    | VERNSAT  | Saturated vernalisation requirements                          | SCr  | days |
    | VERNBASE | Base vernalisation requirements                               | SCr  | days |
    | VERNRTB  | Rate of vernalisation as a function of daily mean temperature | TCr  | -    |
    | VERNDVS  | Critical development stage after which the effect of          | SCr  | -    |
    |          | vernalisation  is halted                                      |      |      |

    **State variables**

    | Name          | Description                                        | Pbl | Unit |
    |---------------|----------------------------------------------------|-----|------|
    | VERN          | Vernalisation state                                | N   | days |
    | DOV           | Day when vernalisation requirements are fulfilled. | N   | -    |
    | ISVERNALISED  | Flag indicated that vernalisation requirement has been reached | Y | - |

    **Rate variables**

    | Name    | Description                                                      | Pbl | Unit |
    |---------|------------------------------------------------------------------|-----|------|
    | VERNR   | Rate of vernalisation                                            | N   | -    |
    | VERNFAC | Reduction factor on development rate due to vernalisation effect.| Y   | -    |

    **External dependencies:**

    | Name | Description                                            | Provided by | Unit |
    |------|--------------------------------------------------------|-------------|------|
    | DVS  | Development stage (only to test if critical VERNDVS    | Phenology   | -    |
    |      | for vernalisation reached)                             |             |      |
    """

    # Helper variable to indicate that DVS > VERNDVS
    _force_vernalisation = Bool(False)

    params_shape = None  # Shape of the parameters tensors

    class Parameters(ParamTemplate):
        VERNSAT = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Saturated vernalisation requirements
        VERNBASE = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Base vernalisation requirements
        VERNRTB = AfgenTrait()  # Vernalisation temperature response
        VERNDVS = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Critical DVS for vernalisation fulfillment

    class RateVariables(RatesTemplate):
        VERNR = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Rate of vernalisation
        VERNFAC = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Red. factor for phenol. devel.

    class StateVariables(StatesTemplate):
        VERN = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Vernalisation state
        DOV = Instance(datetime.date)  # Day when vernalisation
        # requirements are fulfilled
        ISVERNALISED = Bool()  # True when VERNSAT is reached and
        # Forced when DVS > VERNDVS

    def initialize(self, day, kiosk, parvalues):
        """Initialize the Vernalisation sub-module.

        Args:
            day (datetime.date): Simulation start date.
            kiosk: Shared PCSE kiosk for inter-module variable exchange.
            parvalues: ParameterProvider/dict containing VERNSAT, VERNBASE,
                VERNRTB and VERNDVS.

        Side Effects:
            - Instantiates params, rates and states containers.
            - Publishes VERNFAC (rate) and ISVERNALISED (state) to kiosk.

        Initial State:
            VERN = 0.0 (no vernalisation accrued),
            DOV = None (fulfillment date unknown),
            ISVERNALISED = False.

        """
        self.params = self.Parameters(parvalues)
        self.rates = self.RateVariables(kiosk, publish=["VERNFAC"])
        self.kiosk = kiosk

        # Define initial states
        self.states = self.StateVariables(
            kiosk, VERN=0.0, VERNFAC=0.0, DOV=None, ISVERNALISED=False, publish=["ISVERNALISED"]
        )
        self.params_shape = _get_params_shape(self.params)

    @prepare_rates
    def calc_rates(self, day, drv):
        """Calculate vernalisation rates.

        Args:
            day (datetime.date): Current simulation date.
            drv: Driver object providing TEMP.

        Logic:
            - If not vernalised and DVS < VERNDVS: accumulate VERN via VERNRTB(TEMP) and
              compute VERNFAC scaled between VERNBASE and VERNSAT.
            - If DVS >= VERNDVS before fulfillment: stop accumulation, set VERNFAC=1, flag forced.
            - After fulfillment: VERNR=0, VERNFAC=1.
        """
        params = self.params

        # broadcast critical params
        VERNDVS = _broadcast_to(params.VERNDVS, self.params_shape)
        VERNSAT = _broadcast_to(params.VERNSAT, self.params_shape)
        VERNBASE = _broadcast_to(params.VERNBASE, self.params_shape)
        DVS = _broadcast_to(self.kiosk["DVS"], self.params_shape)

        if not self.states.ISVERNALISED:
            if torch.all(DVS < VERNDVS):
                self.rates.VERNR = _broadcast_to(params.VERNRTB(drv.TEMP), self.params_shape)
                r = (self.states.VERN - VERNBASE) / (VERNSAT - VERNBASE)
                self.rates.VERNFAC = tclamp(r, 0.0, 1.0)
            else:
                self.rates.VERNR = torch.zeros(self.params_shape, dtype=DTYPE)
                self.rates.VERNFAC = torch.ones(self.params_shape, dtype=DTYPE)
                self._force_vernalisation = True
        else:
            self.rates.VERNR = torch.zeros(self.params_shape, dtype=DTYPE)
            self.rates.VERNFAC = torch.ones(self.params_shape, dtype=DTYPE)

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Advance vernalisation state.

        Args:
            day (datetime.date): Current simulation date.
            delt (float, optional): Timestep length in days (default 1.0).

        Updates:
            - VERN += VERNR
            - When VERN >= VERNSAT: sets ISVERNALISED=True and records DOV.
            - When critical DVS already passed (forced): sets ISVERNALISED=True
              without assigning DOV and logs a warning.
            - Otherwise keeps ISVERNALISED False.

        Notes:
            VERNFAC is computed in calc_rates and published for use in phenology.

        """
        states = self.states
        rates = self.rates
        params = self.params

        VERNSAT = _broadcast_to(params.VERNSAT, self.params_shape)
        states.VERN = states.VERN + rates.VERNR

        reached = states.VERN >= VERNSAT
        if torch.all(reached):
            states.ISVERNALISED = True
            if states.DOV is None:
                states.DOV = day
                msg = "Vernalization requirements reached at day %s."
                self.logger.info(msg % day)

        elif self._force_vernalisation:  # Critical DVS for vernalisation reached
            # Force vernalisation, but do not set DOV
            states.ISVERNALISED = True

            # Write log message to warn about forced vernalisation
            msg = (
                "Critical DVS for vernalization (VERNDVS) reached "
                + "at day %s, "
                + "but vernalization requirements not yet fulfilled. "
                + "Forcing vernalization now (VERN=%f)."
            )
            self.logger.warning(msg % (day, states.VERN))

        else:  # Reduction factor for phenologic development
            states.ISVERNALISED = False


class DVS_Phenology(SimulationObject):
    """Implements the algorithms for phenologic development in WOFOST.

    Phenologic development in WOFOST is expresses using a unitless scale
    which takes the values 0 at emergence, 1 at Anthesis (flowering) and
    2 at maturity. This type of phenological development is mainly
    representative for cereal crops. All other crops that are simulated
    with WOFOST are forced into this scheme as well, although this may
    not be appropriate for all crops. For example, for potatoes
    development stage 1 represents the start of tuber formation rather
    than flowering.

    Phenological development is mainly governed by temperature and can
    be modified by the effects of day length and vernalization during
    the period before Anthesis. After Anthesis, only temperature
    influences the development rate.

    **Simulation parameters**

    | Name    | Description                                               | Type | Unit |
    |---------|-----------------------------------------------------------|------|------|
    | TSUMEM  | Temperature sum from sowing to emergence                  | SCr  | |C| day  |
    | TBASEM  | Base temperature for emergence                            | SCr  | |C|      |
    | TEFFMX  | Maximum effective temperature for emergence               | SCr  | |C|      |
    | TSUM1   | Temperature sum from emergence to anthesis                | SCr  | |C| day  |
    | TSUM2   | Temperature sum from anthesis to maturity                 | SCr  | |C| day  |
    | IDSL    | Switch for development options: temp only (0), +daylength | SCr  | - |
    |         | (1), +vernalization (>=2)                                 |      |   |
    | DLO     | Optimal daylength for phenological development            | SCr  | hr       |
    | DLC     | Critical daylength for phenological development           | SCr  | hr       |
    | DVSI    | Initial development stage at emergence (may be >0 for     | SCr  | -        |
    |         | transplanted crops)                                       |      |          |
    | DVSEND  | Final development stage                                   | SCr  | -        |
    | DTSMTB  | Daily increase in temperature sum as a function of daily  | TCr  | |C|      |
    |         | mean temperature                                          |      |          |

    **State variables**

    | Name  | Description                                              | Pbl | Unit    |
    |-------|----------------------------------------------------------|-----|---------|
    | DVS   | Development stage                                        | Y   | -       |
    | TSUM  | Temperature sum                                          | N   | |C| day |
    | TSUME | Temperature sum for emergence                            | N   | |C| day |
    | DOS   | Day of sowing                                            | N   | -       |
    | DOE   | Day of emergence                                         | N   | -       |
    | DOA   | Day of Anthesis                                          | N   | -       |
    | DOM   | Day of maturity                                          | N   | -       |
    | DOH   | Day of harvest                                           | N   | -       |
    | STAGE | Current stage (`emerging|vegetative|reproductive|mature`) | N  | -       |

    **Rate variables**

    | Name   | Description                                         | Pbl | Unit  |
    |--------|-----------------------------------------------------|-----|-------|
    | DTSUME | Increase in temperature sum for emergence           | N   | |C|   |
    | DTSUM  | Increase in temperature sum for anthesis or maturity| N   | |C|   |
    | DVR    | Development rate                                    | Y   | |day-1| |

    **External dependencies:**

    None

    **Signals sent or handled**

    `DVS_Phenology` sends the `crop_finish` signal when maturity is
    reached and the `end_type` is 'maturity' or 'earliest'.
    """

    # Placeholder for start/stop types and vernalisation module
    vernalisation = Instance(Vernalisation)

    params_shape = None  # Shape of the parameters tensors

    class Parameters(ParamTemplate):
        TSUMEM = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Temp. sum for emergence
        TBASEM = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Base temp. for emergence
        TEFFMX = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Max eff temperature for emergence
        TSUM1 = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Temperature sum emergence to anthesis
        TSUM2 = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Temperature sum anthesis to maturity
        IDSL = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Switch for photoperiod (1) and vernalisation (2)
        DLO = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Optimal day length for phenol. development
        DLC = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Critical day length for phenol. development
        DVSI = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Initial development stage
        DVSEND = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Final development stage
        DTSMTB = AfgenTrait()  # Temperature response function for phenol.
        # development.
        CROP_START_TYPE = Enum(["sowing", "emergence"])
        CROP_END_TYPE = Enum(["maturity", "harvest", "earliest"])

    class RateVariables(RatesTemplate):
        DTSUME = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # increase in temperature sum for emergence
        DTSUM = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # increase in temperature sum
        DVR = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # development rate

    class StateVariables(StatesTemplate):
        DVS = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Development stage
        TSUM = Any(default_value=torch.tensor(-99.0, dtype=DTYPE))  # Temperature sum state
        TSUME = Any(
            default_value=torch.tensor(-99.0, dtype=DTYPE)
        )  # Temperature sum for emergence state
        # States which register phenological events
        DOS = Instance(datetime.date)  # Day of sowing
        DOE = Instance(datetime.date)  # Day of emergence
        DOA = Instance(datetime.date)  # Day of anthesis
        DOM = Instance(datetime.date)  # Day of maturity
        DOH = Instance(datetime.date)  # Day of harvest
        STAGE = Enum(["emerging", "vegetative", "reproductive", "mature"])

    def initialize(self, day, kiosk, parvalues):
        """:param day: start date of the simulation

        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.params = self.Parameters(parvalues)
        self.params_shape = _get_params_shape(self.params)
        self.rates = self.RateVariables(kiosk)
        self.kiosk = kiosk

        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

        # Define initial states
        DVS, DOS, DOE, STAGE = self._get_initial_stage(day)
        DVS = _broadcast_to(DVS, self.params_shape)  # ensure tensor shape
        self.states = self.StateVariables(
            kiosk,
            publish="DVS",
            TSUM=0.0,
            TSUME=0.0,
            DVS=DVS,
            DOS=DOS,
            DOE=DOE,
            DOA=None,
            DOM=None,
            DOH=None,
            STAGE=STAGE,
        )

        # initialize vernalisation for IDSL=2
        if self.params.IDSL >= 2:
            self.vernalisation = Vernalisation(day, kiosk, parvalues)

    def _get_initial_stage(self, day):
        """Determine initial phenological state at simulation start.

        Args:
            day (datetime.date): Simulation start day.

        Returns:
            tuple: (DVS, DOS, DOE, STAGE)
                DVS (float): Initial development stage (-0.1 if sowing start,
                    DVSI if emergence start).
                DOS (date|None): Sowing date if start type 'sowing'.
                DOE (date|None): Emergence date if start type 'emergence'.
                STAGE (str): One of 'emerging' or 'vegetative'.

        Behavior:
            - If CROP_START_TYPE == 'emergence': assumes emergence already
              occurred; sends crop_emerged signal.
            - If CROP_START_TYPE == 'sowing': pre-emergence phase begins.

        Raises:
            PCSEError: For unknown CROP_START_TYPE.

        """
        p = self.params

        # Define initial stage type (emergence/sowing) and fill the
        # respective day of sowing/emergence (DOS/DOE)
        if p.CROP_START_TYPE == "emergence":
            STAGE = "vegetative"
            DOE = day
            DOS = None
            DVS = p.DVSI

            # send signal to indicate crop emergence
            self._send_signal(signals.crop_emerged)

        elif p.CROP_START_TYPE == "sowing":
            STAGE = "emerging"
            DOS = day
            DOE = None
            DVS = -0.1

        else:
            msg = f"Unknown start type: {p.CROP_START_TYPE}"
            raise exc.PCSEError(msg)

        return DVS, DOS, DOE, STAGE

    @prepare_rates
    def calc_rates(self, day, drv):
        """Compute daily phenological development rates.

        Args:
            day (datetime.date): Current simulation date.
            drv: Meteorological driver object with at least TEMP and LAT.

        Logic:
            1. Photoperiod reduction (DVRED) if IDSL >= 1 using daylength.
            2. Vernalisation factor (VERNFAC) if IDSL >= 2 and in vegetative stage.
            3. Stage-specific:
               - emerging: temperature sum for emergence (DTSUME), DVR via TSUMEM.
               - vegetative: temperature sum (DTSUM) scaled by VERNFAC and DVRED.
               - reproductive: temperature sum (DTSUM) only temperature-driven.
               - mature: all rates zero.

        Sets:
            r.DTSUME, r.DTSUM, r.DVR.

        Raises:
            PCSEError: If STAGE unrecognized.

        """
        p = self.params
        r = self.rates
        s = self.states
        shape = self.params_shape

        # Day length sensitivity
        DVRED = 1.0
        if torch.all(p.IDSL >= 1):
            DAYLP = daylength(day, drv.LAT)
            DAYLP_t = _broadcast_to(DAYLP, shape)
            DLC = _broadcast_to(p.DLC, shape)
            DLO = _broadcast_to(p.DLO, shape)
            DVRED = tclamp((DAYLP_t - DLC) / (DLO - DLC), 0.0, 1.0)

        VERNFAC = _broadcast_to(1.0, shape)
        if torch.all(p.IDSL >= 2) and s.STAGE == "vegetative":
            self.vernalisation.calc_rates(day, drv)
            VERNFAC = _broadcast_to(self.kiosk["VERNFAC"], shape)

        TEMP = _broadcast_to(drv.TEMP, shape)

        # Development rates
        if s.STAGE == "emerging":
            TEFFMX = _broadcast_to(p.TEFFMX, shape)
            TBASEM = _broadcast_to(p.TBASEM, shape)
            r.DTSUME = tclamp(TEMP - TBASEM, 0.0, TEFFMX - TBASEM)
            r.DTSUM = torch.zeros(shape, dtype=DTYPE)
            TSUMEM = _broadcast_to(p.TSUMEM, shape)
            r.DVR = 0.1 * r.DTSUME / TSUMEM
        elif s.STAGE == "vegetative":
            r.DTSUME = torch.zeros(shape, dtype=DTYPE)
            base_rate = _broadcast_to(p.DTSMTB(drv.TEMP), shape)
            TSUM1 = _broadcast_to(p.TSUM1, shape)
            r.DTSUM = base_rate * VERNFAC * DVRED
            r.DVR = r.DTSUM / TSUM1
        elif s.STAGE == "reproductive":
            r.DTSUME = torch.zeros(shape, dtype=DTYPE)
            base_rate = _broadcast_to(p.DTSMTB(drv.TEMP), shape)
            TSUM2 = _broadcast_to(p.TSUM2, shape)
            r.DTSUM = base_rate
            r.DVR = r.DTSUM / TSUM2
        elif s.STAGE == "mature":
            r.DTSUME = torch.zeros(shape, dtype=DTYPE)
            r.DTSUM = torch.zeros(shape, dtype=DTYPE)
            r.DVR = torch.zeros(shape, dtype=DTYPE)
        else:
            msg = "Unrecognized STAGE defined in phenology submodule: %s"
            raise exc.PCSEError(msg, self.states.STAGE)

        msg = "Finished rate calculation for %s"
        self.logger.debug(msg % day)

    @prepare_states
    def integrate(self, day, delt=1.0):
        """Integrate phenology states and manage stage transitions.

        Args:
            day (datetime.date): Current simulation day.
            delt (float, optional): Timestep length in days (default 1.0).

        Sequence:
            - Integrates vernalisation module if active and in vegetative stage.
            - Accumulates TSUME, TSUM, advances DVS by DVR.
            - Checks threshold crossings to move through stages:
                emerging -> vegetative (DVS >= 0)
                vegetative -> reproductive (DVS >= 1)
                reproductive -> mature (DVS >= DVSEND)

        Side Effects:
            - Emits crop_emerged signal on emergence.
            - Emits crop_finish signal at maturity if end type matches.

        Notes:
            Caps DVS at stage boundary values.

        Raises:
            PCSEError: If STAGE undefined.

        """
        p = self.params
        r = self.rates
        s = self.states
        shape = self.params_shape

        # Integrate vernalisation module
        if p.IDSL >= 2:
            if s.STAGE == "vegetative":
                self.vernalisation.integrate(day, delt)
            else:
                self.vernalisation.touch()

        # Integrate phenologic states
        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        s.TSUM = s.TSUM + r.DTSUM

        # Check if a new stage is reached
        if s.STAGE == "emerging":
            if torch.all(s.DVS >= 0.0):
                self._next_stage(day)
                s.DVS = torch.clamp(s.DVS, max=0.0)
        elif s.STAGE == "vegetative":
            if torch.all(s.DVS >= 1.0):
                self._next_stage(day)
                s.DVS = torch.clamp(s.DVS, max=1.0)
        elif s.STAGE == "reproductive":
            DVSEND = _broadcast_to(p.DVSEND, shape)
            if torch.all(s.DVS >= DVSEND):
                self._next_stage(day)
                s.DVS = torch.minimum(s.DVS, DVSEND)
        elif s.STAGE == "mature":
            pass
        else:  # Problem no stage defined
            msg = "No STAGE defined in phenology submodule"
            raise exc.PCSEError(msg)

        msg = "Finished state integration for %s"
        self.logger.debug(msg % day)

    def _next_stage(self, day):
        """Advance to next phenological stage and record event date.

        Args:
            day (datetime.date): Date when transition occurs.

        Transitions:
            emerging -> vegetative (records DOE, sends crop_emerged)
            vegetative -> reproductive (records DOA)
            reproductive -> mature (records DOM, may send crop_finish)
            mature -> Error (cannot advance further)

        Emits:
            crop_finish signal at maturity when CROP_END_TYPE in ['maturity','earliest'].

        Raises:
            PCSEError: If called in mature stage or with invalid current stage.

        """
        s = self.states
        p = self.params

        current_STAGE = s.STAGE
        if s.STAGE == "emerging":
            s.STAGE = "vegetative"
            s.DOE = day
            # send signal to indicate crop emergence
            self._send_signal(signals.crop_emerged)

        elif s.STAGE == "vegetative":
            s.STAGE = "reproductive"
            s.DOA = day

        elif s.STAGE == "reproductive":
            s.STAGE = "mature"
            s.DOM = day
            if p.CROP_END_TYPE in ["maturity", "earliest"]:
                self._send_signal(
                    signal=signals.crop_finish, day=day, finish_type="maturity", crop_delete=True
                )
        elif s.STAGE == "mature":
            msg = "Cannot move to next phenology stage: maturity already reached!"
            raise exc.PCSEError(msg)

        else:  # Problem no stage defined
            msg = "No STAGE defined in phenology submodule."
            raise exc.PCSEError(msg)

        msg = "Changed phenological stage '%s' to '%s' on %s"
        self.logger.info(msg % (current_STAGE, s.STAGE, day))

    def _on_CROP_FINISH(self, day, finish_type=None):
        """Handle external crop finish signal to set harvest date.

        Args:
            day (datetime.date): Date provided by finish event.
            finish_type (str|None): 'harvest', 'earliest', or other finish reason.

        Behavior:
            - If finish_type in ('harvest','earliest'): registers DOH for finalization.

        Notes:
            Maturity-driven finish is triggered internally in _next_stage; this
            handler captures management-induced harvests.

        """
        if finish_type in ["harvest", "earliest"]:
            self._for_finalize["DOH"] = day
