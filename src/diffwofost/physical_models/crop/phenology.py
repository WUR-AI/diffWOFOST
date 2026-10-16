"""Phenological development and vernalisation models for WOFOST.

This module implements:
- Vernalisation: modification of phenological development due to cold
exposure.
- DVS_Phenology: main phenology progression (DVS scale: 0 emergence, 1
anthesis, 2 maturity).
"""

import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import SimulationObject
from pcse.traitlets import Enum
from pcse.traitlets import Instance
from pcse.util import doy
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.deferred import defer
from diffwofost.physical_models.kernels import kernel
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _raise_latitude_error
from diffwofost.physical_models.utils import _restore_state
from diffwofost.physical_models.utils import _snapshot_state
from diffwofost.physical_models.utils import daylength_core
from diffwofost.physical_models.utils import solar_declination_table


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

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return getattr(self, "_device", ComputeConfig.get_device())

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return getattr(self, "_dtype", ComputeConfig.get_dtype())

    class Parameters(TensorParamTemplate):
        VERNSAT = Tensor(-99.0)
        VERNBASE = Tensor(-99.0)
        VERNRTB = AfgenTrait()
        VERNDVS = Tensor(-99.0)

    class RateVariables(TensorRatesTemplate):
        VERNR = Tensor(0.0)
        VERNFAC = Tensor(0.0)

    class StateVariables(TensorStatesTemplate):
        VERN = Tensor(-99.0)
        DOV = Tensor(-99.0)
        ISVERNALISED = Tensor(False, dtype=bool)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Initialize the Vernalisation sub-module.

        Args:
            day (datetime.date): Simulation start date.
            kiosk: Shared PCSE kiosk for inter-module variable exchange.
            parvalues: ParameterProvider/dict containing VERNSAT, VERNBASE,
                VERNRTB and VERNDVS.
            shape (tuple | torch.Size | None): Target shape for the state and rate variables.

        Side Effects:
            - Instantiates params, rates and states containers.
            - Publishes VERNFAC (rate) and ISVERNALISED (state) to kiosk.

        Initial State:
            VERN = 0.0 (no vernalisation accrued),
            DOV = None (fulfillment date unknown),
            ISVERNALISED = False.

        """
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()

        self.params = self.Parameters(parvalues, shape=shape)

        # Small epsilon tensor reused in multiple safe divisions.
        self._epsilon = torch.tensor(1e-8, dtype=self.dtype, device=self.device)

        # Common constant tensors (same shape/dtype/device as this module).
        self._ones = torch.ones(self.params.shape, dtype=self.dtype, device=self.device)
        self._zeros = torch.zeros(self.params.shape, dtype=self.dtype, device=self.device)

        # Explicitly initialize rates
        self.rates = self.RateVariables(kiosk, publish=["VERNFAC"], shape=shape)

        self.kiosk = kiosk

        # Define initial states
        self.states = self.StateVariables(
            kiosk,
            VERN=0.0,
            DOV=-1.0,  # -1 indicates not yet fulfilled
            ISVERNALISED=False,
            publish=["ISVERNALISED"],
            shape=shape,
        )
        # Per-element force flag (False for all elements initially)
        self._force_vernalisation = torch.zeros(
            self.params.shape, dtype=torch.bool, device=self.device
        )

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
        VERNDVS = params.VERNDVS
        VERNSAT = params.VERNSAT
        VERNBASE = params.VERNBASE
        DVS = self.kiosk["DVS"]

        TEMP = drv["TEMP"]

        # Operate elementwise only on elements not yet vernalised
        not_vernalised = ~self.states.ISVERNALISED
        vegetative_mask = not_vernalised & (DVS >= 0.0) & (DVS < VERNDVS)
        past_threshold_mask = not_vernalised & (DVS >= VERNDVS)

        # VERNR only for vegetative elements
        self.rates.VERNR = torch.where(
            vegetative_mask,
            params.VERNRTB(TEMP),
            self._zeros,
        )

        # compute VERNFAC from current VERN for vegetative elements; others = 1
        safe_den = VERNSAT - VERNBASE
        safe_den = safe_den.sign() * torch.maximum(torch.abs(safe_den), self._epsilon)
        r = (self.states.VERN - VERNBASE) / safe_den
        vernfac_computed = torch.clamp(r, 0.0, 1.0)
        self.rates.VERNFAC = torch.where(
            vegetative_mask,
            vernfac_computed,
            self._ones,
        )

        # mark per-element force flags for elements that passed VERNDVS but aren't vernalised
        self._force_vernalisation = self._force_vernalisation | past_threshold_mask

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

        VERNSAT = params.VERNSAT
        # accumulate vernalisation per element
        states.VERN = states.VERN + rates.VERNR

        # elements that reached requirement
        reached = states.VERN >= VERNSAT
        # update ISVERNALISED per-element
        states.ISVERNALISED = states.ISVERNALISED | reached

        # set DOV only for newly reached elements
        newly_reached_and_no_dov = reached & (states.DOV < 0.0)
        states.DOV = torch.where(newly_reached_and_no_dov, float(day.toordinal()), states.DOV)
        defer(
            torch.any(newly_reached_and_no_dov),
            lambda: self.logger.info(f"Vernalization requirements reached at day {day}."),
        )

        # forced vernalisation per-element
        forced_mask = self._force_vernalisation & (~states.ISVERNALISED)
        states.ISVERNALISED = states.ISVERNALISED | forced_mask
        defer(
            torch.any(forced_mask),
            lambda: self.logger.warning(
                "Critical DVS for vernalization (VERNDVS) reached at"
                + f" day {day} for some elements; forcing vernalization now."
            ),
        )
        # clear force bits for those elements
        self._force_vernalisation = self._force_vernalisation & (~forced_mask)


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

    | Name            | Description                                               | Type | Unit    |
    |-----------------|-----------------------------------------------------------|------|---------|
    | TSUMEM          | Temperature sum from sowing to emergence                  | SCr  | |C| day |
    | TBASEM          | Base temperature for emergence                            | SCr  | |C|     |
    | TEFFMX          | Maximum effective temperature for emergence               | SCr  | |C|     |
    | TSUM2           | Temperature sum from anthesis to maturity                 | SCr  | |C| day |
    | IDSL            | Switch for development options: temp only (0), +daylength | SCr  | -       |
    |                 | (1), +vernalization (>=2)                                 |      |         |
    | DLO             | Optimal daylength for phenological development            | SCr  | hr      |
    | DLC             | Critical daylength for phenological development           | SCr  | hr      |
    | DVSI            | Initial development stage at emergence (may be >0 for     | SCr  | -       |
    |                 | transplanted crops)                                       |      |         |
    | DVSEND          | Final development stage                                   | SCr  | -       |
    | DTSMTB          | Daily increase in temperature sum as a function of daily  | TCr  | |C|     |
    |                 | mean temperature                                          |      |         |
    | CROP_START_DATE | Per-element override day (ordinal) for this element's     | SCr  | day     |
    |                 | crop start (sowing/emergence date, per CROP_START_TYPE);  |      | ordinal |
    |                 | -1 (default) uses the simulation's own start day for that |      |         |
    |                 | element, i.e. the current single-start-date behaviour     |      |         |

    **State variables**

    | Name      | Description                                       | Pbl | Unit    |
    |-----------|---------------------------------------------------|-----|---------|
    | DVS       | Development stage                                 | Y   | -       |
    | TSUM      | Temperature sum                                   | N   | |C| day |
    | TSUME     | Temperature sum for emergence                     | N   | |C| day |
    | DOS       | Day of sowing                                     | N   | -       |
    | DOE       | Day of emergence                                  | N   | -       |
    | DOA       | Day of Anthesis                                   | N   | -       |
    | DOM       | Day of maturity                                   | N   | -       |
    | DOH       | Day of harvest                                    | N   | -       |
    | STAGE     | Stage (`not_started`/`emerging`/`vegetative`/     | N   | -       |
    |           | `reproductive`/`mature`)                          |     |         |
    | IS_ACTIVE | Whether this element has started (see note below) | Y   | -       |

    An element whose CROP_START_DATE lies in the future starts with
    STAGE=-1 ("not started"). The state and rate variables DVS/TSUM/TSUME/
    DTSUME/DTSUM/DVR are set NaN, until the simulation day reaches its own
    start date; IS_ACTIVE is False for as long as this is the case. Once an
    element has started, IS_ACTIVE stays True for the rest of the run,
    including after the element individually reaches maturity: a matured
    element keeps reporting its frozen final DVS/TSUM/TSUME and zero rates
    (other batch elements may keep progressing in the meantime). DOS/DOE/
    DOA/DOM/DOH are historical markers and are never masked to NaN by IS_ACTIVE.

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

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                |
    |--------|------------------------------------------|
    | DVS    | TSUMEM, TBASEM, TEFFMX,TSUM1, TSUM2, DLO, DLC, DVSI, |
    |        | DVSEND, DTSMTB, VERNSAT, VERNBASE, VERNDVS|
    | TSUM   | DVSI, DVSEND, DTSMTB, VERNSAT, VERNBASE, VERNDVS  |
    | TSUME  | TBASEM, TEFFMX |

    [!NOTE]
    Notice that the gradient ∂DVS/∂TEFFMX is zero.

    [!NOTE]
    The parameter IDSL it is not differentiable since it is a switch.
    Similarly, CROP_START_DATE is not differentiable (it is a day-ordinal index).
    """

    # Placeholder for start/stop types and vernalisation module
    vernalisation = Instance(Vernalisation)

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return getattr(self, "_device", ComputeConfig.get_device())

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return getattr(self, "_dtype", ComputeConfig.get_dtype())

    class Parameters(TensorParamTemplate):
        TSUMEM = Tensor(-99.0)
        TBASEM = Tensor(-99.0)
        TEFFMX = Tensor(-99.0)
        TSUM1 = Tensor(-99.0)
        TSUM2 = Tensor(-99.0)
        IDSL = Tensor(-99.0)
        DLO = Tensor(-99.0)
        DLC = Tensor(-99.0)
        DVSI = Tensor(-99.0)
        DVSEND = Tensor(-99.0)
        DTSMTB = AfgenTrait()
        CROP_START_TYPE = Enum(["sowing", "emergence"])
        CROP_END_TYPE = Enum(["maturity", "harvest", "earliest"])
        CROP_START_DATE = Tensor(-1.0)

    class RateVariables(TensorRatesTemplate):
        DTSUME = Tensor(0.0)
        DTSUM = Tensor(0.0)
        DVR = Tensor(0.0)

    class StateVariables(TensorStatesTemplate):
        DVS = Tensor(-99.0)
        TSUM = Tensor(-99.0)
        TSUME = Tensor(-99.0)
        DOS = Tensor(-99.0)
        DOE = Tensor(-99.0)
        DOA = Tensor(-99.0)
        DOM = Tensor(-99.0)
        DOH = Tensor(-99.0)
        STAGE = Tensor(-99.0)
        IS_ACTIVE = Tensor(True, dtype=bool)

    def initialize(self, day, kiosk, parvalues, shape=None):
        """Initialize the DVS_Phenology module.

        Args:
            day: start date of the simulation
            kiosk: variable kiosk of this PCSE instance
            parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
            shape: optional shape for state and rate tensors (default None for scalar)
        """
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()

        # CROP_START_DATE is optional: we inject here the "not provided" sentinel.
        if "CROP_START_DATE" not in parvalues:
            parvalues = dict(parvalues)
            parvalues["CROP_START_DATE"] = -1.0

        self.params = self.Parameters(parvalues, shape=shape)

        # Initialize vernalisation for IDSL>=2
        if torch.any(self.params.IDSL >= 2):
            self.vernalisation = Vernalisation(day, kiosk, parvalues, shape=shape)
        else:
            self.vernalisation = None

        # Create scalar constants once at the beginning to avoid recreating them
        self._ones = torch.ones(self.params.shape, dtype=self.dtype, device=self.device)
        self._zeros = torch.zeros(self.params.shape, dtype=self.dtype, device=self.device)
        self._nan = torch.full(self.params.shape, torch.nan, dtype=self.dtype, device=self.device)
        self._epsilon = torch.tensor(1e-8, dtype=self.dtype, device=self.device)
        self._solar = solar_declination_table(self.dtype, self.device)
        # day ordinal of the start day on the device, see integrate()
        self._day0 = day.toordinal()
        self._ordinal0 = torch.tensor(float(self._day0), dtype=self.dtype, device=self.device)

        # Initialize rates and kiosk
        self.rates = self.RateVariables(kiosk, shape=shape)
        self.kiosk = kiosk

        # see issue #60
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

        # Define initial states
        DVS, DOS, DOE, STAGE, TSUM, TSUME, IS_ACTIVE = self._get_initial_stage(day)

        self.states = self.StateVariables(
            kiosk,
            publish=["DVS", "IS_ACTIVE"],
            TSUM=TSUM,
            TSUME=TSUME,
            DVS=DVS,
            DOS=DOS,
            DOE=DOE,
            DOA=-1,  # not yet occurred
            DOM=-1,  # not yet occurred
            DOH=-1,  # not yet occurred
            STAGE=STAGE,
            IS_ACTIVE=IS_ACTIVE,
            shape=shape,
        )

    def _get_initial_stage(self, day):
        """Determine initial phenological state at simulation start.

        Elements whose CROP_START_DATE lies after the simulation's own start day
        are not yet active: STAGE is set to -1 ("not started") and DVS/TSUM/TSUME
        are set to NaN until their own start day is reached (see `integrate`).

        Args:
            day (datetime.date): Simulation start day.

        Returns:
            tuple: (DVS, DOS, DOE, STAGE, TSUM, TSUME, IS_ACTIVE)
                DVS (Tensor): Initial development stage (-0.1 if sowing start,
                    or DVSI if emergence start; NaN if not yet active).
                DOS (Tensor): Sowing date ordinal (or -1 if not applicable).
                DOE (Tensor): Emergence date ordinal (or -1 if not applicable).
                STAGE (Tensor): Integer stage code (-1=not started, 0=emerging, 1=vegetative).
                TSUM (Tensor): Initial temperature sum (0.0, or NaN if not active).
                TSUME (Tensor): Initial temperature sum for emergence (0.0, or NaN if not active).
                IS_ACTIVE (Tensor): Whether the element has already started.
        """
        p = self.params
        day0 = torch.full(p.shape, day.toordinal(), dtype=self.dtype, device=self.device)

        # Per-element effective start day: CROP_START_DATE if provided (>=0) and
        # not in the past relative to day0, otherwise day0 (current date).
        has_override = p.CROP_START_DATE >= 0
        effective_start = torch.where(has_override, p.CROP_START_DATE, day0)
        effective_start = torch.maximum(effective_start, day0)
        self._start_ordinal = effective_start

        is_active = effective_start <= day0

        # Define initial stage type (emergence/sowing) and fill the
        # respective day of sowing/emergence (DOS/DOE)
        if p.CROP_START_TYPE == "emergence":
            stage_today = 1.0  # 1 = vegetative
            doe_today = day0
            dos_today = -self._ones  # Not applicable
            dvs_today = p.DVSI

        elif p.CROP_START_TYPE == "sowing":
            stage_today = 0.0  # 0 = emerging
            dos_today = day0
            doe_today = -self._ones  # Not yet occurred
            dvs_today = -0.1 * self._ones

        else:
            msg = f"Unknown start type: {p.CROP_START_TYPE}"
            raise exc.PCSEError(msg)

        DVS = torch.where(is_active, dvs_today, self._nan)
        STAGE = torch.where(is_active, stage_today, -1.0)
        DOS = torch.where(is_active, dos_today, -1.0)
        DOE = torch.where(is_active, doe_today, -1.0)
        TSUM = torch.where(is_active, self._zeros, self._nan)
        TSUME = torch.where(is_active, self._zeros, self._nan)

        return DVS, DOS, DOE, STAGE, TSUM, TSUME, is_active

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
        # Vernalisation factor - always computed if module exists, even before CROP_START_DATE is
        # reached; harmless since VERNFAC is unused outside STAGE==1.
        # Revisit if per-element vernalisation timing matters.
        VERNFAC = None
        if self.vernalisation is not None:
            # Always call calc_rates (it handles stage internally now)
            self.vernalisation.calc_rates(day, drv)
            VERNFAC = self.kiosk["VERNFAC"]

        latitude = torch.as_tensor(drv["LAT"], dtype=self.dtype, device=self.device)
        defer(torch.any(latitude.abs() > 90.0), _raise_latitude_error)
        r.DTSUME, r.DTSUM, r.DVR = _phenology_rates(
            s.STAGE,
            s.IS_ACTIVE,
            drv["TEMP"],
            latitude,
            self._solar[doy(day)],
            VERNFAC,
            p.IDSL,
            p.DLO,
            p.DLC,
            p.TBASEM,
            p.TEFFMX,
            p.TSUMEM,
            p.TSUM1,
            p.TSUM2,
            p.DTSMTB,
            self._ones,
            self._zeros,
            self._nan,
            self._epsilon,
        )

        msg = "Finished rate calculation for %s"
        self.logger.debug(msg % day)

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
            - Emits crop_finish signal at maturity if end type matches.

        Notes:
            Caps DVS at stage boundary values.

        Raises:
            PCSEError: If STAGE undefined.

        """
        p = self.params
        r = self.rates
        s = self.states

        # Integrate vernalisation module
        if self.vernalisation:
            # Save a copy of state
            state_copy = _snapshot_state(self.vernalisation.states)
            mask_IDSL = p.IDSL >= 2

            # Check if any element is in vegetative stage i.e. stage 1
            mask_STAGE = mask_IDSL & (s.STAGE == 1)
            self.vernalisation.integrate(day, delt)
            state_integrated = _snapshot_state(self.vernalisation.states)

            # Restore original state
            _restore_state(self.vernalisation.states, state_copy)
            self.vernalisation.touch()
            state_touched = _snapshot_state(self.vernalisation.states)

            # Apply the masks
            for name in state_copy:
                # results of vernalisation module
                vernalisation_states = torch.where(
                    mask_STAGE, state_integrated[name], state_touched[name]
                )
                setattr(
                    self.vernalisation.states,
                    name,
                    torch.where(mask_IDSL, vernalisation_states, state_copy[name]),
                )

        # Day ordinal as a device tensor (a Python number that changes every day
        # would recompile the compiled kernel every day)
        day_ordinal = self._ordinal0 + float(day.toordinal() - self._day0)
        (s.TSUME, s.DVS, s.TSUM, s.STAGE, s.DOS, s.DOE, s.DOA, s.DOM, s.IS_ACTIVE) = (
            _phenology_integrate(
                s.TSUME,
                s.DVS,
                s.TSUM,
                s.STAGE,
                s.DOS,
                s.DOE,
                s.DOA,
                s.DOM,
                r.DTSUME,
                r.DVR,
                r.DTSUM,
                day_ordinal,
                self._start_ordinal,
                p.CROP_START_TYPE == "emergence",
                p.DVSI,
                p.DVSEND,
                self._ones,
                self._zeros,
                self._nan,
            )
        )

        # Send crop_finish signal if maturity reached for all elements that have
        # started (elements still pending a future start have STAGE == -1, which
        # keeps this condition False, so the run is not terminated early).
        # The condition is evaluated at the end of the simulated day (see
        # `diffwofost.physical_models.deferred`); the engine acts on the signal in
        # the next time step either way.
        if p.CROP_END_TYPE in ["maturity", "earliest"]:
            defer(
                s.STAGE == 3,
                lambda: self._send_signal(
                    signal=signals.crop_finish,
                    day=day,
                    finish_type="maturity",
                    crop_delete=True,
                ),
            )

        msg = "Finished state integration for %s"
        self.logger.debug(msg % day)

    def _on_CROP_FINISH(self, day, finish_type=None):
        """Handle external crop finish signal to set harvest date.

        Args:
            day (datetime.date): Date provided by finish event.
            finish_type (str|None): 'harvest', 'earliest', or other finish reason.

        Behavior:
            - If finish_type in ('harvest','earliest'): registers DOH for finalization.

        Notes:
            Maturity-driven finish is triggered internally as part of integrate; this
            handler captures management-induced harvests.

        """
        if finish_type in ["harvest", "earliest"]:
            self._for_finalize["DOH"] = torch.full(
                self.params.shape, day.toordinal(), dtype=self.dtype, device=self.device
            )


@kernel
def _phenology_rates(
    STAGE, IS_ACTIVE, TEMP, latitude, solar, VERNFAC,
    IDSL, DLO, DLC, TBASEM, TEFFMX, TSUMEM, TSUM1, TSUM2, DTSMTB, ones, zeros, nan, epsilon,
):  # fmt: skip
    """Daily phenological development rates, see `DVS_Phenology.calc_rates`."""
    # Day length sensitivity
    DAYLP = daylength_core(latitude, solar[0], solar[1])
    # Compute DVRED conditionally based on IDSL >= 1
    safe_den = DLO - DLC
    safe_den = safe_den.sign() * torch.maximum(torch.abs(safe_den), epsilon)
    dvred_active = torch.clamp((DAYLP - DLC) / safe_den, 0.0, 1.0)
    DVRED = torch.where(IDSL >= 1, dvred_active, ones)

    # Apply vernalisation only where IDSL >= 2 AND in vegetative stage
    if VERNFAC is None:
        VERNFAC = ones
    else:
        VERNFAC = torch.where((IDSL >= 2) & (STAGE == 1), VERNFAC, ones)

    # Initialize all rate variables
    DTSUME = zeros
    DTSUM = zeros
    DVR = zeros

    # Emerging stage (STAGE == 0)
    is_emerging = STAGE == 0
    temp_diff = TEMP - TBASEM
    max_diff = torch.clamp(TEFFMX - TBASEM, min=0.0)
    dtsume_emerging = torch.minimum(torch.clamp(temp_diff, min=0.0), max_diff)
    safe_den_em = TSUMEM.sign() * torch.maximum(torch.abs(TSUMEM), epsilon)
    dvr_emerging = 0.1 * dtsume_emerging / safe_den_em
    DTSUME = torch.where(is_emerging, dtsume_emerging, DTSUME)
    DVR = torch.where(is_emerging, dvr_emerging, DVR)

    # Vegetative stage (STAGE == 1)
    is_vegetative = STAGE == 1
    dtsum_vegetative = DTSMTB(TEMP) * VERNFAC * DVRED
    safe_den_v1 = TSUM1.sign() * torch.maximum(torch.abs(TSUM1), epsilon)
    dvr_vegetative = dtsum_vegetative / safe_den_v1
    DTSUM = torch.where(is_vegetative, dtsum_vegetative, DTSUM)
    DVR = torch.where(is_vegetative, dvr_vegetative, DVR)

    # Reproductive stage (STAGE == 2)
    is_reproductive = STAGE == 2
    dtsum_reproductive = DTSMTB(TEMP)
    safe_den_v2 = TSUM2.sign() * torch.maximum(torch.abs(TSUM2), epsilon)
    dvr_reproductive = dtsum_reproductive / safe_den_v2
    DTSUM = torch.where(is_reproductive, dtsum_reproductive, DTSUM)
    DVR = torch.where(is_reproductive, dvr_reproductive, DVR)

    # Mature stage (STAGE == 3) keeps zeros (already initialised).
    # Not-yet-started (STAGE == -1) elements report invalid rates rather than
    # the (physically meaningless) zeros above.
    DTSUME = torch.where(IS_ACTIVE, DTSUME, nan)
    DTSUM = torch.where(IS_ACTIVE, DTSUM, nan)
    DVR = torch.where(IS_ACTIVE, DVR, nan)
    return DTSUME, DTSUM, DVR


@kernel
def _phenology_integrate(
    TSUME, DVS, TSUM, STAGE, DOS, DOE, DOA, DOM, DTSUME, DVR, DTSUM,
    day_ordinal, start_ordinal, start_at_emergence, DVSI, DVSEND, ones, zeros, nan,
):  # fmt: skip
    """Phenology states and stage transitions, see `DVS_Phenology.integrate`."""
    # Integrate phenologic states
    TSUME = TSUME + DTSUME
    DVS = DVS + DVR
    TSUM = TSUM + DTSUM

    # Check transition for not started -> emerging/vegetative (STAGE -1 -> 0/1)
    is_pending = STAGE == -1
    should_start = is_pending & (day_ordinal >= start_ordinal)
    if start_at_emergence:
        start_stage = 1.0
        start_dvs = DVSI
    else:
        start_stage = 0.0
        start_dvs = -0.1 * ones

    STAGE = torch.where(should_start, start_stage, STAGE)
    DOE = torch.where(should_start & start_at_emergence, day_ordinal, DOE)
    DOS = torch.where(should_start & (not start_at_emergence), day_ordinal, DOS)
    DVS = torch.where(should_start, start_dvs, DVS)
    TSUM = torch.where(should_start, zeros, TSUM)
    TSUME = torch.where(should_start, zeros, TSUME)

    # Check transitions for emerging -> vegetative (STAGE 0 -> 1)
    is_emerging = STAGE == 0
    should_emerge = is_emerging & (DVS >= 0.0)
    STAGE = torch.where(should_emerge, 1.0, STAGE)
    DOE = torch.where(should_emerge, day_ordinal, DOE)
    DVS = torch.where(should_emerge, torch.clamp(DVS, max=0.0), DVS)

    # Check transitions for vegetative -> reproductive (STAGE 1 -> 2)
    is_vegetative = STAGE == 1
    should_flower = is_vegetative & (DVS >= 1.0)
    STAGE = torch.where(should_flower, 2.0, STAGE)
    DOA = torch.where(should_flower, day_ordinal, DOA)
    DVS = torch.where(should_flower, torch.clamp(DVS, max=1.0), DVS)

    # Check transitions for reproductive -> mature (STAGE 2 -> 3)
    is_reproductive = STAGE == 2
    should_mature = is_reproductive & (DVS >= DVSEND)
    STAGE = torch.where(should_mature, 3.0, STAGE)
    DOM = torch.where(should_mature, day_ordinal, DOM)
    DVS = torch.where(should_mature, torch.minimum(DVS, DVSEND), DVS)

    # An element is active from the day it starts onwards, including after it
    # individually reaches maturity: a matured element keeps reporting its
    # frozen final DVS/TSUM/TSUME and zero rates. Only an element that has not
    # yet reached its own start date is inactive. DOS/DOE/DOA/DOM/DOH are
    # historical markers and are not masked here.
    IS_ACTIVE = STAGE != -1
    DVS = torch.where(IS_ACTIVE, DVS, nan)
    TSUM = torch.where(IS_ACTIVE, TSUM, nan)
    TSUME = torch.where(IS_ACTIVE, TSUME, nan)
    return TSUME, DVS, TSUM, STAGE, DOS, DOE, DOA, DOM, IS_ACTIVE
