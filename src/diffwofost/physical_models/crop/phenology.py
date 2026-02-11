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
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Enum
from pcse.traitlets import Instance
from pcse.util import daylength
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import _broadcast_to
from diffwofost.physical_models.utils import _get_drv
from diffwofost.physical_models.utils import _restore_state
from diffwofost.physical_models.utils import _snapshot_state


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
        VERNDVS = params.VERNDVS
        VERNSAT = params.VERNSAT
        VERNBASE = params.VERNBASE
        DVS = self.kiosk["DVS"]

        TEMP = _get_drv(drv.TEMP, self.params.shape, self.dtype, self.device)

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
        if torch.any(past_threshold_mask):
            self._force_vernalisation = self._force_vernalisation | past_threshold_mask

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

        VERNSAT = params.VERNSAT
        # accumulate vernalisation per element
        states.VERN = states.VERN + rates.VERNR

        # elements that reached requirement
        reached = states.VERN >= VERNSAT
        # update ISVERNALISED per-element
        states.ISVERNALISED = states.ISVERNALISED | reached

        # set DOV only for newly reached elements
        newly_reached_and_no_dov = reached & (states.DOV < 0.0)
        if torch.any(newly_reached_and_no_dov):
            day_ordinal = torch.tensor(day.toordinal(), dtype=self.dtype, device=self.device)
            states.DOV = torch.where(newly_reached_and_no_dov, day_ordinal, states.DOV)
            self.logger.info(f"Vernalization requirements reached at day {day}.")

        # forced vernalisation per-element
        forced_mask = self._force_vernalisation & (~states.ISVERNALISED)
        if torch.any(forced_mask):
            states.ISVERNALISED = states.ISVERNALISED | forced_mask
            self.logger.warning(
                "Critical DVS for vernalization (VERNDVS) reached at"
                + f" day {day} for some elements; forcing vernalization now."
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

    **Gradient mapping (which parameters have a gradient):**

    | Output | Parameters influencing it                |
    |--------|------------------------------------------|
    | DVS    | ... |
    | TSUM   | ... |

    [!NOTE]
    Notice that the gradient ∂DVS/∂TEFFMX is zero.

    [!NOTE]
    The parameter IDSL it is not differentiable since it is a switch.
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

    def initialize(self, day, kiosk, parvalues, shape=None):
        """:param day: start date of the simulation

        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self._device = ComputeConfig.get_device()
        self._dtype = ComputeConfig.get_dtype()

        self.params = self.Parameters(parvalues, shape=shape)

        # Initialize vernalisation for IDSL>=2
        if torch.any(self.params.IDSL >= 2):
            self.vernalisation = Vernalisation(day, kiosk, parvalues, shape=shape)
        else:
            self.vernalisation = None

        # Create scalar constants once at the beginning to avoid recreating them
        self._ones = torch.ones(self.params.shape, dtype=self.dtype, device=self.device)
        self._zeros = torch.zeros(self.params.shape, dtype=self.dtype, device=self.device)
        self._epsilon = torch.tensor(1e-8, dtype=self.dtype, device=self.device)

        # Initialize rates and kiosk
        self.rates = self.RateVariables(kiosk, shape=shape)
        self.kiosk = kiosk

        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

        # Define initial states
        DVS, DOS, DOE, STAGE = self._get_initial_stage(day)

        self.states = self.StateVariables(
            kiosk,
            publish="DVS",
            TSUM=0.0,
            TSUME=0.0,
            DVS=DVS,
            DOS=DOS,
            DOE=DOE,
            DOA=-1.0,  # not yet occurred
            DOM=-1.0,  # not yet occurred
            DOH=-1.0,  # not yet occurred
            STAGE=STAGE,
            shape=shape,
        )

    def _get_initial_stage(self, day):
        """Determine initial phenological state at simulation start.

        Args:
            day (datetime.date): Simulation start day.

        Returns:
            tuple: (DVS, DOS, DOE, STAGE)
                DVS (Tensor): Initial development stage (-0.1 if sowing start,
                    or DVSI if emergence start).
                DOS (Tensor): Sowing date ordinal (or -1 if not applicable).
                DOE (Tensor): Emergence date ordinal (or -1 if not applicable).
                STAGE (Tensor): Integer stage code (0=emerging, 1=vegetative).
        """
        p = self.params
        day_ordinal = day.toordinal()

        # Define initial stage type (emergence/sowing) and fill the
        # respective day of sowing/emergence (DOS/DOE)
        if p.CROP_START_TYPE == "emergence":
            STAGE = 1  # 1 = vegetative
            DOE = day_ordinal
            DOS = -1.0  # Not applicable
            DVS = p.DVSI

            # send signal to indicate crop emergence
            self._send_signal(signals.crop_emerged)

        elif p.CROP_START_TYPE == "sowing":
            STAGE = 0  # 0 = emerging
            DOS = day_ordinal
            DOE = -1.0  # Not yet occurred
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

        # Day length sensitivity
        DAYLP = daylength(day, drv.LAT)
        DAYLP_t = _broadcast_to(DAYLP, p.shape, dtype=self.dtype, device=self.device)
        # Compute DVRED conditionally based on IDSL >= 1
        safe_den = p.DLO - p.DLC
        safe_den = safe_den.sign() * torch.maximum(torch.abs(safe_den), self._epsilon)
        dvred_active = torch.clamp((DAYLP_t - p.DLC) / safe_den, 0.0, 1.0)
        DVRED = torch.where(p.IDSL >= 1, dvred_active, self._ones)

        # Vernalisation factor - always compute if module exists
        VERNFAC = self._ones
        if self.vernalisation is not None:
            # Always call calc_rates (it handles stage internally now)
            self.vernalisation.calc_rates(day, drv)
            # Apply vernalisation only where IDSL >= 2 AND in vegetative stage
            is_vegetative = s.STAGE == 1
            VERNFAC = torch.where(
                (p.IDSL >= 2) & is_vegetative,
                self.kiosk["VERNFAC"],
                self._ones,
            )

        TEMP = _get_drv(drv.TEMP, p.shape, self.dtype, self.device)

        # Initialize all rate variables
        r.DTSUME = self._zeros
        r.DTSUM = self._zeros
        r.DVR = self._zeros

        # Compute rates for emerging stage (STAGE == 0)
        is_emerging = s.STAGE == 0
        if torch.any(is_emerging):
            temp_diff = TEMP - p.TBASEM
            # Ensure the maximum effective temperature difference is non-negative
            max_diff = torch.clamp(p.TEFFMX - p.TBASEM, min=0.0)
            dtsume_emerging = torch.clamp(temp_diff, min=0.0)
            dtsume_emerging = torch.minimum(dtsume_emerging, max_diff)
            safe_den = p.TSUMEM
            safe_den = safe_den.sign() * torch.maximum(torch.abs(safe_den), self._epsilon)
            dvr_emerging = 0.1 * dtsume_emerging / safe_den

            r.DTSUME = torch.where(is_emerging, dtsume_emerging, r.DTSUME)
            r.DVR = torch.where(is_emerging, dvr_emerging, r.DVR)

        # Compute rates for vegetative stage (STAGE == 1)
        is_vegetative = s.STAGE == 1
        if torch.any(is_vegetative):
            dtsum_vegetative = p.DTSMTB(TEMP) * VERNFAC * DVRED
            safe_den = p.TSUM1
            safe_den = safe_den.sign() * torch.maximum(torch.abs(safe_den), self._epsilon)
            dvr_vegetative = dtsum_vegetative / safe_den

            r.DTSUM = torch.where(is_vegetative, dtsum_vegetative, r.DTSUM)
            r.DVR = torch.where(is_vegetative, dvr_vegetative, r.DVR)

        # Compute rates for reproductive stage (STAGE == 2)
        is_reproductive = s.STAGE == 2
        if torch.any(is_reproductive):
            dtsum_reproductive = p.DTSMTB(TEMP)
            safe_den = p.TSUM2
            safe_den = safe_den.sign() * torch.maximum(torch.abs(safe_den), self._epsilon)
            dvr_reproductive = dtsum_reproductive / safe_den

            r.DTSUM = torch.where(is_reproductive, dtsum_reproductive, r.DTSUM)
            r.DVR = torch.where(is_reproductive, dvr_reproductive, r.DVR)

        # Mature stage (STAGE == 3) keeps zeros (already initialized)

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

        # Integrate phenologic states
        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        s.TSUM = s.TSUM + r.DTSUM

        day_ordinal = torch.tensor(day.toordinal(), dtype=self.dtype, device=self.device)

        # Check transitions for emerging -> vegetative (STAGE 0 -> 1)
        is_emerging = s.STAGE == 0
        should_emerge = is_emerging & (s.DVS >= 0.0)
        s.STAGE = torch.where(should_emerge, 1.0, s.STAGE)
        s.DOE = torch.where(should_emerge, day_ordinal, s.DOE)
        s.DVS = torch.where(should_emerge, torch.clamp(s.DVS, max=0.0), s.DVS)

        # Send signal if any crop emerged (only once per day)
        if torch.any(should_emerge):
            self._send_signal(signals.crop_emerged)

        # Check transitions for vegetative -> reproductive (STAGE 1 -> 2)
        is_vegetative = s.STAGE == 1
        should_flower = is_vegetative & (s.DVS >= 1.0)
        s.STAGE = torch.where(should_flower, 2.0, s.STAGE)
        s.DOA = torch.where(should_flower, day_ordinal, s.DOA)
        s.DVS = torch.where(should_flower, torch.clamp(s.DVS, max=1.0), s.DVS)

        # Check transitions for reproductive -> mature (STAGE 2 -> 3)
        is_reproductive = s.STAGE == 2
        should_mature = is_reproductive & (s.DVS >= p.DVSEND)
        s.STAGE = torch.where(should_mature, 3.0, s.STAGE)
        s.DOM = torch.where(should_mature, day_ordinal, s.DOM)
        s.DVS = torch.where(should_mature, torch.minimum(s.DVS, p.DVSEND), s.DVS)

        # Send crop_finish signal if maturity reached for one.
        # assumption is that all elements mature simultaneously
        # TODO: revisit this when fixing engine for agromanager
        if torch.any(should_mature) and p.CROP_END_TYPE in ["maturity", "earliest"]:
            self._send_signal(
                signal=signals.crop_finish,
                day=day,
                finish_type="maturity",
                crop_delete=True,
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
            Maturity-driven finish is triggered internally in _next_stage; this
            handler captures management-induced harvests.

        """
        if finish_type in ["harvest", "earliest"]:
            self._for_finalize["DOH"] = torch.full(
                self.params.shape, day.toordinal(), dtype=self.dtype, device=self.device
            )

    def get_variable(self, varname):
        # TODO: should be removed while fixing #49. this is needed because
        # conditions are applied on STAGE in pcse.crop.wofost72.py
        """Return the value of the specified state or rate variable.

        :param varname: Name of the variable.

        Note that the `get_variable()` will searches for `varname` exactly
        as specified (case sensitive).
        """
        if varname == "STAGE":
            # Return string representation of current stage
            stage_map = {
                0: "emerging",
                1: "vegetative",
                2: "reproductive",
                3: "mature",
            }
            stage_value = self.states.STAGE
            if stage_value.dim() != 0:
                stage_id = stage_value.flatten()[0].item()
            else:
                stage_id = stage_value.item()
            return stage_map[stage_id]

        # Search for variable in the current object, then traverse the hierarchy
        value = None
        if hasattr(self.states, varname):
            value = getattr(self.states, varname)
        elif hasattr(self.rates, varname):
            value = getattr(self.rates, varname)
        # Query individual sub-SimObject for existence of variable v
        else:
            for simobj in self.subSimObjects:
                value = simobj.get_variable(varname)
                if value is not None:
                    break
        return value
