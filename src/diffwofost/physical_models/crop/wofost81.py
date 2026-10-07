"""WOFOST 8.1 crop simulation with nitrogen stress and biomass reallocation.

The assimilate balance is the WOFOST 7.2 balance, with two additions taken
from PCSE's ``Wofost81``: reallocation of leaf and stem biomass to storage
organs after ``REALLOC_DVS``, and a nitrogen demand/uptake/stress loop.
Soil water and nitrogen are not part of this class. Pair it with
``SoilModuleWrapper_PP`` for potential production, or with
``SoilModuleWrapper_NWLP_MLWB_SNOMIN`` for the layered SNOMIN configuration.
"""

import datetime
import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.traitlets import Instance
from pcse.traitlets import Unicode
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.base.simulationobject import initialize_all_components
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.crop.assimilation import WOFOST81_Assimilation as Assimilation
from diffwofost.physical_models.crop.evapotranspiration import (
    EvapotranspirationWrapper as Evapotranspiration,
)
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics_N as Leaf_Dynamics
from diffwofost.physical_models.crop.n_dynamics import N_Crop_Dynamics as N_Crop
from diffwofost.physical_models.crop.nutrients.n_stress import N_Stress
from diffwofost.physical_models.crop.partitioning import DVS_Partitioning_N as Partitioning
from diffwofost.physical_models.crop.phenology import DVS_Phenology as Phenology
from diffwofost.physical_models.crop.respiration import (
    WOFOST_Maintenance_Respiration as MaintenanceRespiration,
)
from diffwofost.physical_models.crop.root_dynamics import WOFOST_Root_Dynamics as Root_Dynamics
from diffwofost.physical_models.crop.stem_dynamics import WOFOST_Stem_Dynamics as Stem_Dynamics
from diffwofost.physical_models.crop.storage_organ_dynamics import (
    WOFOST_Storage_Organ_Dynamics as Storage_Organ_Dynamics,
)
from diffwofost.physical_models.traitlets import Tensor


class Wofost81(SimulationObject):
    """Top-level WOFOST 8.1 crop simulation.

    The crop simulation organizes the different processes of the crop. It also
    holds the parameters, rate and state variables which are relevant at the
    level of the entire crop. The processes that are implemented as embedded
    simulation objects consist of:

        1. Phenology (self.pheno)
        2. Partitioning (self.part)
        3. Assimilation (self.assim)
        4. Maintenance respiration (self.mres)
        5. Evapotranspiration (self.evtra)
        6. Leaf dynamics (self.lv_dynamics)
        7. Stem dynamics (self.st_dynamics)
        8. Root dynamics (self.ro_dynamics)
        9. Storage organ dynamics (self.so_dynamics)
        10. N crop dynamics (self.n_crop_dynamics)
        11. N stress (self.n_stress)

    **Simulation parameters:**

    | Name    | Description                                               | Type | Unit |
    |---------|-----------------------------------------------------------|------|------|
    | CVL     | Conversion factor for assimilates to leaves               | SCr  | -    |
    | CVO     | Conversion factor for assimilates to storage organs       | SCr  | -    |
    | CVR     | Conversion factor for assimilates to roots                | SCr  | -    |
    | CVS     | Conversion factor for assimilates to stems                | SCr  | -    |
    | REALLOC_DVS | Development stage at which reallocation starts        | SCr  | -    |
    | REALLOC_STEM_FRACTION | Stem fraction available for reallocation    | SCr  | kg DM kg-1 DM |
    | REALLOC_LEAF_FRACTION | Leaf fraction available for reallocation    | SCr  | kg DM kg-1 DM |
    | REALLOC_STEM_RATE | Relative rate of reallocation from stems          | SCr  | d-1  |
    | REALLOC_LEAF_RATE | Relative rate of reallocation from leaves         | SCr  | d-1  |
    | REALLOC_EFFICIENCY | Efficiency of reallocation to storage organs   | SCr  | kg DM kg-1 DM |

    **State variables:**

    | Name    | Description                                               | Type | Unit |
    |---------|-----------------------------------------------------------|------|------|
    | TAGP    | Total above-ground Production                             | N    | kg ha-1 |
    | GASST   | Total gross assimilation                                  | N    | kg CH2O ha-1 |
    | MREST   | Total gross maintenance respiration                       | N    | kg CH2O ha-1 |
    | CTRAT   | Total crop transpiration accumulated over the crop cycle  | N    | cm |
    | CEVST   | Total soil evaporation accumulated over the crop cycle    | N    | cm |
    | HI      | Harvest Index (only calculated during `finalize()`)       | N    | - |
    | DOF     | Date representing the day of finish of the crop simulation | N    | - |
    | FINISH_TYPE | String representing the reason for finishing the simulation: | N    | - |
    |  |  maturity, harvest, leave death, etc. |   |  |
    | LV_REALLOCATED | Leaf biomass already reallocated to storage organs | N | kg ha-1 |
    | ST_REALLOCATED | Stem biomass already reallocated to storage organs | N | kg ha-1 |

    **Rate variables:**

    | Name    | Description                                             | Type | Unit |
    |---------|---------------------------------------------------------|------|------|
    | GASS    | Assimilation rate corrected for water stress            | N    | kg CH2O ha-1 d-1 |
    | MRES    | Actual maintenance respiration rate, ...                | N    | kg CH2O ha-1 d-1 |
    |     | ... taking into account that MRES <= GASS |   | |
    | ASRC    | Net available assimilates (GASS - MRES)                 | N    | kg CH2O ha-1 d-1 |
    | DMI     | Total dry matter increase, ...                          | Y    | kg ha-1 d-1 |
    |      | ... calculated as ASRC times a weighted conversion efficiency |    | |
    | ADMI    | Aboveground dry matter increase                         | Y    | kg ha-1 d-1 |
    | REALLOC_LV | Reallocation rate from leaves                        | Y    | kg ha-1 d-1 |
    | REALLOC_ST | Reallocation rate from stems                         | Y    | kg ha-1 d-1 |
    | REALLOC_SO | Reallocation rate into storage organs                | Y    | kg ha-1 d-1 |

    **Gradient mapping (which parameters have a gradient):**

    | Output      | Parameters influencing it                                      |
    |-------------|----------------------------------------------------------------|
    | DMI, TAGP   | CVL, CVO, CVR, CVS                                             |
    | REALLOC_LV  | REALLOC_LEAF_FRACTION, REALLOC_LEAF_RATE                      |
    | REALLOC_ST  | REALLOC_STEM_FRACTION, REALLOC_STEM_RATE                      |
    | REALLOC_SO  | REALLOC_LEAF_FRACTION, REALLOC_LEAF_RATE,                     |
    |             | REALLOC_STEM_FRACTION, REALLOC_STEM_RATE, REALLOC_EFFICIENCY  |

    [!NOTE]
    ``REALLOC_DVS`` is a hard development-stage switch, so its gradient is zero.
    The leaf and stem fractions still have a gradient once reallocation has started.
    """

    pheno = Instance(SimulationObject)
    part = Instance(SimulationObject)
    assim = Instance(SimulationObject)
    mres = Instance(SimulationObject)
    evtra = Instance(SimulationObject)
    lv_dynamics = Instance(SimulationObject)
    st_dynamics = Instance(SimulationObject)
    ro_dynamics = Instance(SimulationObject)
    so_dynamics = Instance(SimulationObject)
    n_crop_dynamics = Instance(SimulationObject)
    n_stress = Instance(SimulationObject)

    COMPONENT_SPECS = {
        "phenology": ("pheno", Phenology),
        "partitioning": ("part", Partitioning),
        "assimilation": ("assim", Assimilation),
        "maintenance_respiration": ("mres", MaintenanceRespiration),
        "evapotranspiration": ("evtra", Evapotranspiration),
        "root_dynamics": ("ro_dynamics", Root_Dynamics),
        "stem_dynamics": ("st_dynamics", Stem_Dynamics),
        "storage_organ_dynamics": ("so_dynamics", Storage_Organ_Dynamics),
        "leaf_dynamics": ("lv_dynamics", Leaf_Dynamics),
        "n_crop_dynamics": ("n_crop_dynamics", N_Crop),
        "n_stress": ("n_stress", N_Stress),
    }

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    class Parameters(TensorParamTemplate):
        CVL = Tensor(-99.0)
        CVO = Tensor(-99.0)
        CVR = Tensor(-99.0)
        CVS = Tensor(-99.0)
        REALLOC_DVS = Tensor(2.0)
        REALLOC_STEM_FRACTION = Tensor(0.0)
        REALLOC_LEAF_FRACTION = Tensor(0.0)
        REALLOC_STEM_RATE = Tensor(0.0)
        REALLOC_LEAF_RATE = Tensor(0.0)
        REALLOC_EFFICIENCY = Tensor(0.0)

    class StateVariables(TensorStatesTemplate):
        TAGP = Tensor(-99.0)
        GASST = Tensor(-99.0)
        MREST = Tensor(-99.0)
        CTRAT = Tensor(-99.0)
        CEVST = Tensor(-99.0)
        HI = Tensor(-99.0)
        DOF = Instance(datetime.date)
        FINISH_TYPE = Unicode(allow_none=True)
        LV_REALLOCATED = Tensor(0.0)
        ST_REALLOCATED = Tensor(0.0)

    class RateVariables(TensorRatesTemplate):
        GASS = Tensor(0.0)
        MRES = Tensor(0.0)
        ASRC = Tensor(0.0)
        DMI = Tensor(0.0)
        ADMI = Tensor(0.0)
        REALLOC_LV = Tensor(0.0)
        REALLOC_ST = Tensor(0.0)
        REALLOC_SO = Tensor(0.0)

    def initialize(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | torch.Size | None = None,
        component_overrides: dict | None = None,
    ) -> None:
        """Initialise crop components, including nitrogen book-keeping.

        Args:
            day: Start date of the simulation.
            kiosk: Variable kiosk used to read and publish crop state.
            parvalues: Parameter provider containing the physical-model
                parameters for the crop.
            shape: Target tensor shape for state and rate variables.
            component_overrides: Optional mapping used to replace one or more
                internal WOFOST components at construction time.
                The ``component_overrides`` is a dictionary containing:
                - "component_class": The class to use for the component.
                - "model": The model to use for the component, if specified in the override.
                - "kwargs": Any additional keyword arguments to pass to the component
                constructor.
        """
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(
            kiosk, publish=["DMI", "ADMI", "REALLOC_LV", "REALLOC_ST", "REALLOC_SO"], shape=shape
        )
        self.kiosk = kiosk

        # Initialize components of the crop
        # This will add attributes to self for each component, e.g. self.pheno, self.part, etc.
        initialize_all_components(
            self, day, kiosk, parvalues, shape=shape, component_overrides=component_overrides
        )

        # Initial total (living+dead) above-ground biomass of the crop
        tagp = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO
        self.states = self.StateVariables(
            kiosk,
            publish=["TAGP", "GASST", "MREST", "HI"],
            TAGP=tagp,
            GASST=0.0,
            MREST=0.0,
            CTRAT=0.0,
            CEVST=0.0,
            HI=0.0,
            DOF=None,
            FINISH_TYPE=None,
            LV_REALLOCATED=0.0,
            ST_REALLOCATED=0.0,
            shape=shape,
        )

        # Check partitioning of TDWI over plant organs
        checksum = parvalues["TDWI"] - self.states.TAGP - self.kiosk["TWRT"]
        if torch.any(torch.abs(checksum) > 0.0001):
            raise exc.PartitioningError("Error in partitioning of initial biomass (TDWI)!")

        # Placeholders for biomass available for reallocation
        self._WLV_REALLOC = None
        self._WST_REALLOC = None
        self._realloc_latched = None

        # assign handler for CROP_FINISH signal
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

    @staticmethod
    def _check_carbon_balance(day, dmi, gass, mres, cvf, pf):
        fr, fl, fs, fo = pf
        checksum = (gass - mres - (fr + (fl + fs + fo) * (1.0 - fr)) * dmi / cvf) / torch.clamp(
            gass, min=0.0001
        )
        if torch.any(torch.abs(checksum) >= 0.0001):
            msg = (
                f"Carbon flows not balanced on day {day}. "
                f"Checksum: {checksum.mean().item():f}, GASS: {gass.mean().item():f}"
            )
            raise exc.CarbonBalanceError(msg)

    def calc_rates(self, day: datetime.date, drv: dict) -> None:
        """Assimilation, reallocation, nitrogen stress and organ growth rates.

        Args:
            day (datetime.date): The current date of the simulation.
            drv (dict): A container holding weather data elements as key/value. The values are
                arrays or scalars.
        """
        params = self.params
        rates = self.rates
        kiosk = self.kiosk

        # Phenology
        self.pheno.calc_rates(day, drv)

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # TODO: revisit this when fixing #60
        if torch.all(self.pheno.states.STAGE == 0):
            return

        # Potential assimilation
        pgass = self.assim(day, drv)

        # (evapo)transpiration rates
        self.evtra(day, drv)

        # water stress reduction
        rates.GASS = pgass * kiosk.RFTRA

        # Respiration
        pmres = self.mres(day, drv)
        rates.MRES = torch.minimum(rates.GASS, pmres)

        # Net available assimilates
        rates.ASRC = rates.GASS - rates.MRES

        # DM partitioning factors (pf), conversion factor (CVF),
        # dry matter increase (DMI) and check on carbon balance
        pf = self.part.calc_rates(day, drv)
        cvf = 1.0 / (
            (pf.FL / params.CVL + pf.FS / params.CVS + pf.FO / params.CVO) * (1.0 - pf.FR)
            + pf.FR / params.CVR
        )
        rates.DMI = cvf * rates.ASRC
        self._check_carbon_balance(day, rates.DMI, rates.GASS, rates.MRES, cvf, pf)

        # Reallocation from stems/leaves
        self._reallocate(kiosk, params, rates)

        # Calculate N stress indices
        self.n_stress(day, drv)

        # distribution over plant organ

        # Below-ground dry matter increase and root dynamics
        self.ro_dynamics.calc_rates(day, drv)
        # Aboveground dry matter increase and distribution over stems,
        # leaves, organs
        rates.ADMI = (1.0 - pf.FR) * rates.DMI
        self.st_dynamics.calc_rates(day, drv)
        self.so_dynamics.calc_rates(day, drv)
        self.lv_dynamics.calc_rates(day, drv)

        self.n_crop_dynamics.calc_rates(day, drv)

    def _reallocate(self, kiosk, params, rates):
        """Move a fraction of leaf and stem biomass into storage organs.

        The amount that can be reallocated is latched on the first day
        development passes ``REALLOC_DVS``, matching PCSE.
        """
        # TODO: move reallocation to separate module that can be shared between 7.3 and 8.1
        started = kiosk.DVS >= params.REALLOC_DVS
        if self._WLV_REALLOC is None:
            self._WLV_REALLOC = torch.zeros_like(rates.DMI)
            self._WST_REALLOC = torch.zeros_like(rates.DMI)
            self._realloc_latched = torch.zeros_like(rates.DMI, dtype=torch.bool)

        # Start of reallocation: latch the maximum reallocatable biomass.
        just_started = started & ~self._realloc_latched
        self._WLV_REALLOC = torch.where(
            just_started, kiosk.WLV * params.REALLOC_LEAF_FRACTION, self._WLV_REALLOC
        )
        self._WST_REALLOC = torch.where(
            just_started, kiosk.WST * params.REALLOC_STEM_FRACTION, self._WST_REALLOC
        )
        self._realloc_latched = self._realloc_latched | started

        # Reallocation rate in terms of loss of stem/leaf dry matter.
        leaf_left = self._WLV_REALLOC - self.states.LV_REALLOCATED
        stem_left = self._WST_REALLOC - self.states.ST_REALLOCATED
        rates.REALLOC_LV = torch.where(
            started & (self.states.LV_REALLOCATED < self._WLV_REALLOC),
            torch.minimum(self._WLV_REALLOC * params.REALLOC_LEAF_RATE, leaf_left),
            torch.zeros_like(rates.DMI),
        )
        rates.REALLOC_ST = torch.where(
            started & (self.states.ST_REALLOCATED < self._WST_REALLOC),
            torch.minimum(self._WST_REALLOC * params.REALLOC_STEM_RATE, stem_left),
            torch.zeros_like(rates.DMI),
        )
        # Increase in storage organs, including conversion losses.
        rates.REALLOC_SO = (rates.REALLOC_LV + rates.REALLOC_ST) * params.REALLOC_EFFICIENCY

    def integrate(self, day: datetime.date, delt=1.0) -> None:
        """Integrate the state variables using the rates of change.

        Args:
            day (datetime.date): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        rates = self.rates
        states = self.states

        # Capture stage *before* integration (phenology will advance it).
        # STAGE == 0 means "emerging"; read directly from the tensor to avoid
        # a GPU→CPU sync from get_variable() / .item().
        crop_stage_before = self.pheno.states.STAGE.clone()

        # Phenology
        self.pheno.integrate(day, delt)

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # Just run a touch() to to ensure that all state variables are available
        # in the kiosk
        # TODO: revisit this when fixing #60
        if torch.all(crop_stage_before == 0):
            self.touch()
            return

        # Partitioning
        self.part.integrate(day, delt)

        # Integrate states on leaves, storage organs, stems and roots
        self.ro_dynamics.integrate(day, delt)
        self.so_dynamics.integrate(day, delt)
        self.st_dynamics.integrate(day, delt)
        self.lv_dynamics.integrate(day, delt)

        self.n_crop_dynamics.integrate(day, delt)

        # Integrate total (living+dead) above-ground biomass of the crop
        states.TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        # Keep track of amount of reallocated biomass
        states.LV_REALLOCATED = states.LV_REALLOCATED + rates.REALLOC_LV * delt
        states.ST_REALLOCATED = states.ST_REALLOCATED + rates.REALLOC_ST * delt

        # total gross assimilation and maintenance respiration
        states.GASST = states.GASST + rates.GASS * delt
        states.MREST = states.MREST + rates.MRES * delt

        # total crop transpiration and soil evaporation
        states.CTRAT = states.CTRAT + self.kiosk.TRA * delt
        states.CEVST = states.CEVST + self.kiosk.EVS * delt

    def finalize(self, day: datetime.date) -> None:
        """Finalize the crop simulation by computing the Harvest Index."""
        # Calculate Harvest Index
        tagp = self.states.TAGP
        if torch.any(tagp <= 0):
            self.logger.warning("Cannot calculate Harvest Index because TAGP=0")
        self.states.HI = torch.where(
            tagp > 0, self.kiosk.TWSO / torch.clamp(tagp, min=1e-10), torch.full_like(tagp, -1.0)
        )
        SimulationObject.finalize(self, day)

    def _on_CROP_FINISH(self, day, finish_type=None):
        """Handler for setting day of finish (DOF) and reason for crop finishing (FINISH)."""
        self._for_finalize["DOF"] = day
        self._for_finalize["FINISH_TYPE"] = finish_type
