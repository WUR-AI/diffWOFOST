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
from diffwofost.physical_models.crop.assimilation import WOFOST72_Assimilation as Assimilation
from diffwofost.physical_models.crop.evapotranspiration import (
    EvapotranspirationWrapper as Evapotranspiration,
)
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics as Leaf_Dynamics
from diffwofost.physical_models.crop.partitioning import DVS_Partitioning as Partitioning
from diffwofost.physical_models.crop.phenology import DVS_Phenology as Phenology
from diffwofost.physical_models.crop.respiration import (
    WOFOST_Maintenance_Respiration as MaintenanceRespiration,
)
from diffwofost.physical_models.crop.root_dynamics import WOFOST_Root_Dynamics as Root_Dynamics
from diffwofost.physical_models.crop.stem_dynamics import WOFOST_Stem_Dynamics as Stem_Dynamics
from diffwofost.physical_models.crop.storage_organ_dynamics import (
    WOFOST_Storage_Organ_Dynamics as Storage_Organ_Dynamics,
)
from diffwofost.physical_models.deferred import defer
from diffwofost.physical_models.kernels import kernel
from diffwofost.physical_models.traitlets import Tensor


class Wofost72(SimulationObject):
    """Top level object organizing the different components of WOFOST.

    The CropSimulation object organizes the different processes of the crop
    simulation. Moreover, it contains the parameters, rate and state variables
    which are relevant at the level of the entire crop. The processes that are
    implemented as embedded simulation objects consist of:

        1. Phenology (self.pheno)
        2. Partitioning (self.part)
        3. Assimilation (self.assim)
        4. Maintenance respiration (self.mres)
        5. Evapotranspiration (self.evtra)
        6. Leaf dynamics (self.lv_dynamics)
        7. Stem dynamics (self.st_dynamics)
        8. Root dynamics (self.ro_dynamics)
        9. Storage organ dynamics (self.so_dynamics)

    **Simulation parameters:**

    | Name    | Description                                               | Type | Unit |
    |---------|-----------------------------------------------------------|------|------|
    | CVL     | Conversion factor for assimilates to leaves               | SCr  | -    |
    | CVO     | Conversion factor for assimilates to storage organs       | SCr  | -    |
    | CVR     | Conversion factor for assimilates to roots                | SCr  | -    |
    | CVS     | Conversion factor for assimilates to stems                | SCr  | -    |


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

    """

    # sub-model components for crop simulation
    pheno = Instance(SimulationObject)
    part = Instance(SimulationObject)
    assim = Instance(SimulationObject)
    mres = Instance(SimulationObject)
    evtra = Instance(SimulationObject)
    lv_dynamics = Instance(SimulationObject)
    st_dynamics = Instance(SimulationObject)
    ro_dynamics = Instance(SimulationObject)
    so_dynamics = Instance(SimulationObject)

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
    }

    @property
    def device(self):
        """Get device from ComputeConfig."""
        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        return ComputeConfig.get_dtype()

    # Parameters, rates and states which are relevant at the main crop
    # simulation level
    class Parameters(TensorParamTemplate):
        CVL = Tensor(-99.0)
        CVO = Tensor(-99.0)
        CVR = Tensor(-99.0)
        CVS = Tensor(-99.0)

    class StateVariables(TensorStatesTemplate):
        TAGP = Tensor(-99.0)
        GASST = Tensor(-99.0)
        MREST = Tensor(-99.0)
        CTRAT = Tensor(-99.0)
        CEVST = Tensor(-99.0)
        HI = Tensor(-99.0)
        DOF = Instance(datetime.date)
        FINISH_TYPE = Unicode(allow_none=True)

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
        """Initialize the crop simulation and its embedded components.

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
            self,
            day,
            kiosk,
            parvalues,
            shape=shape,
            component_overrides=component_overrides,
        )

        # Initial total (living+dead) above-ground biomass of the crop
        TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO
        self.states = self.StateVariables(
            kiosk,
            publish=["TAGP", "GASST", "MREST", "HI"],
            TAGP=TAGP,
            GASST=0.0,
            MREST=0.0,
            CTRAT=0.0,
            CEVST=0.0,
            HI=0.0,
            DOF=None,
            FINISH_TYPE=None,
            shape=shape,
        )

        # Check partitioning of TDWI over plant organs
        checksum = parvalues["TDWI"] - self.states.TAGP - self.kiosk["TWRT"]
        if torch.any(torch.abs(checksum) > 0.0001):
            msg = "Error in partitioning of initial biomass (TDWI)!"
            raise exc.PartitioningError(msg)

        # whether all elements are still emerging, see integrate()
        self._all_emerging = None

        # assign handler for CROP_FINISH signal
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

    @staticmethod
    def _check_carbon_balance(day, DMI, GASS, MRES, CVF, pf):
        """Check that carbon flows are balanced on the current day.

        The check is evaluated at the end of the simulated day (see
        `diffwofost.physical_models.deferred`), so that it does not stall the device.
        """
        (FR, FL, FS, FO) = pf
        checksum = (
            (GASS - MRES - (FR + (FL + FS + FO) * (1.0 - FR)) * DMI / CVF)
            * 1.0
            / (torch.clamp(GASS, min=0.0001))
        )

        def fail():
            msg = f"Carbon flows not balanced on day {day}\n"
            msg += (
                f"Checksum: {checksum.mean().item():f}, GASS: {GASS.mean().item():f},"
                f" MRES: {MRES.mean().item():f}\n"
            )
            msg += (
                f"FR,L,S,O: {FR.mean().item():5.3f},{FL.mean().item():5.3f},"
                f"{FS.mean().item():5.3f},{FO.mean().item():5.3f},"
                f" DMI: {DMI.mean().item():f}, CVF: {CVF.mean().item():f}\n"
            )
            raise exc.CarbonBalanceError(msg)

        defer(torch.any(torch.abs(checksum) >= 0.0001), fail)

    def calc_rates(self, day: datetime.date, drv: dict) -> None:
        """Calculate the rates of change of the state variables.

        Args:
            day (datetime.date): The current date of the simulation.
            drv (dict): A container holding weather data elements as key/value. The values are
                arrays or scalars.
        """
        p = self.params
        r = self.rates
        k = self.kiosk

        # Phenology
        self.pheno.calc_rates(day, drv)

        # The rates are also calculated when all elements are still emerging: the
        # components mask the elements that have not emerged, and testing for that
        # case here would make the host wait for the device every day. The rates are
        # only integrated after an element has emerged (see integrate()).

        # Potential assimilation
        PGASS = self.assim(day, drv)

        # (evapo)transpiration rates
        self.evtra(day, drv)

        # Respiration
        PMRES = self.mres(day, drv)

        # DM partitioning factors (pf), conversion factor (CVF),
        # dry matter increase (DMI) and check on carbon balance
        pf = self.part.calc_rates(day, drv)
        r.GASS, r.MRES, r.ASRC, CVF, r.DMI, r.ADMI = _dry_matter_increase(
            PGASS, k.RFTRA, PMRES, pf.FR, pf.FL, pf.FS, pf.FO, p.CVL, p.CVS, p.CVO, p.CVR
        )
        self._check_carbon_balance(day, r.DMI, r.GASS, r.MRES, CVF, pf)

        # distribution over plant organ

        # Reallocation from stems/leaves not applicable in WOFOST72
        r.REALLOC_LV = torch.zeros_like(r.DMI)
        r.REALLOC_ST = torch.zeros_like(r.DMI)
        r.REALLOC_SO = torch.zeros_like(r.DMI)

        # Below-ground dry matter increase and root dynamics
        self.ro_dynamics.calc_rates(day, drv)
        # Aboveground dry matter increase (r.ADMI above) and distribution over stems,
        # leaves, organs
        self.st_dynamics.calc_rates(day, drv)
        self.so_dynamics.calc_rates(day, drv)
        self.lv_dynamics.calc_rates(day, drv)

    def integrate(self, day: datetime.date, delt=1.0) -> None:
        """Integrate the state variables using the rates of change.

        Args:
            day (datetime.date): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        rates = self.rates
        states = self.states

        # Whether all elements were still emerging *before* this integration
        # (phenology will advance the stage); evaluated at the end of the previous day.
        if self._all_emerging is None:
            self._all_emerging = defer(self.pheno.states.STAGE == 0)
        all_emerging = self._all_emerging.get()

        # Phenology
        self.pheno.integrate(day, delt)
        self._all_emerging = defer(self.pheno.states.STAGE == 0)

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # Just run a touch() to to ensure that all state variables are available
        # in the kiosk
        # TODO: revisit this when fixing #60
        if all_emerging:
            self.touch()
            return

        # Partitioning
        self.part.integrate(day, delt)

        # Integrate states on leaves, storage organs, stems and roots
        self.ro_dynamics.integrate(day, delt)
        self.so_dynamics.integrate(day, delt)
        self.st_dynamics.integrate(day, delt)
        self.lv_dynamics.integrate(day, delt)

        k = self.kiosk
        states.TAGP, states.GASST, states.MREST, states.CTRAT, states.CEVST = _crop_totals(
            k.TWLV, k.TWST, k.TWSO, states.GASST, states.MREST, states.CTRAT, states.CEVST,
            rates.GASS, rates.MRES, k.TRA, k.EVS,
        )  # fmt: skip

    def finalize(self, day: datetime.date) -> None:
        """Finalize the crop simulation by computing the Harvest Index."""
        # Calculate Harvest Index
        TAGP = self.states.TAGP
        if torch.any(TAGP <= 0):
            msg = "Cannot calculate Harvest Index because TAGP=0"
            self.logger.warning(msg)
        self.states.HI = torch.where(
            TAGP > 0, self.kiosk.TWSO / torch.clamp(TAGP, min=1e-10), torch.full_like(TAGP, -1.0)
        )

        SimulationObject.finalize(self, day)

    def _on_CROP_FINISH(self, day, finish_type=None):
        """Handler for setting day of finish (DOF) and reason for crop finishing (FINISH)."""
        self._for_finalize["DOF"] = day
        self._for_finalize["FINISH_TYPE"] = finish_type


@kernel
def _dry_matter_increase(PGASS, RFTRA, PMRES, FR, FL, FS, FO, CVL, CVS, CVO, CVR):
    """Assimilation, respiration and dry matter increase, see `Wofost72.calc_rates`."""
    # water stress reduction
    GASS = PGASS * RFTRA

    # Respiration
    MRES = torch.minimum(GASS, PMRES)

    # Net available assimilates
    ASRC = GASS - MRES

    # conversion factor (CVF) and dry matter increase (DMI)
    CVF = 1.0 / ((FL / CVL + FS / CVS + FO / CVO) * (1.0 - FR) + FR / CVR)
    DMI = CVF * ASRC

    # Aboveground dry matter increase
    ADMI = (1.0 - FR) * DMI
    return GASS, MRES, ASRC, CVF, DMI, ADMI


@kernel
def _crop_totals(TWLV, TWST, TWSO, GASST, MREST, CTRAT, CEVST, GASS, MRES, TRA, EVS):
    """Crop totals after integration, see `Wofost72.integrate`."""
    # Integrate total (living+dead) above-ground biomass of the crop
    TAGP = TWLV + TWST + TWSO

    # total gross assimilation and maintenance respiration
    GASST = GASST + GASS
    MREST = MREST + MRES

    # total crop transpiration and soil evaporation
    CTRAT = CTRAT + TRA
    CEVST = CEVST + EVS
    return TAGP, GASST, MREST, CTRAT, CEVST
