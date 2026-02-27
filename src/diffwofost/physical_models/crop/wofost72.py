import datetime
import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.crop.storage_organ_dynamics import WOFOST_Storage_Organ_Dynamics as Storage_Organ_Dynamics
from pcse.decorators import prepare_rates
from pcse.decorators import prepare_states
from pcse.traitlets import Instance
from pcse.traitlets import Unicode
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.traitlets import Tensor
from .assimilation import WOFOST72_Assimilation as Assimilation
from .evapotranspiration import EvapotranspirationWrapper as Evapotranspiration
from .leaf_dynamics import WOFOST_Leaf_Dynamics as Leaf_Dynamics
from .partitioning import DVS_Partitioning as Partitioning
from .phenology import DVS_Phenology as Phenology
from .respiration import WOFOST_Maintenance_Respiration as MaintenanceRespiration
from .root_dynamics import WOFOST_Root_Dynamics as Root_Dynamics
from .stem_dynamics import WOFOST_Stem_Dynamics as Stem_Dynamics


class Wofost72(SimulationObject):
    """Top level object organizing the different components of the WOFOST crop
    simulation.

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

    ======== =============================================== =======  ==========
     Name     Description                                     Type     Unit
    ======== =============================================== =======  ==========
    CVL      Conversion factor for assimilates to leaves       SCr     -
    CVO      Conversion factor for assimilates to storage      SCr     -
             organs.
    CVR      Conversion factor for assimilates to roots        SCr     -
    CVS      Conversion factor for assimilates to stems        SCr     -
    ======== =============================================== =======  ==========


    **State variables:**

    =========== ================================================= ==== ===============
     Name        Description                                      Pbl      Unit
    =========== ================================================= ==== ===============
    TAGP        Total above-ground Production                      N    |kg ha-1|
    GASST       Total gross assimilation                           N    |kg CH2O ha-1|
    MREST       Total gross maintenance respiration                N    |kg CH2O ha-1|
    CTRAT       Total crop transpiration accumulated over the
                crop cycle                                         N    cm
    CEVST       Total soil evaporation accumulated over the
                crop cycle                                         N    cm
    HI          Harvest Index (only calculated during              N    -
                `finalize()`)
    DOF         Date representing the day of finish of the crop    N    -
                simulation.
    FINISH_TYPE String representing the reason for finishing the   N    -
                simulation: maturity, harvest, leave death, etc.
    =========== ================================================= ==== ===============


     **Rate variables:**

    =======  ================================================ ==== =============
     Name     Description                                      Pbl      Unit
    =======  ================================================ ==== =============
    GASS     Assimilation rate corrected for water stress       N  |kg CH2O ha-1 d-1|
    MRES     Actual maintenance respiration rate, taking into
             account that MRES <= GASS.                         N  |kg CH2O ha-1 d-1|
    ASRC     Net available assimilates (GASS - MRES)            N  |kg CH2O ha-1 d-1|
    DMI      Total dry matter increase, calculated as ASRC
             times a weighted conversion efficiency.            Y  |kg ha-1 d-1|
    ADMI     Aboveground dry matter increase                    Y  |kg ha-1 d-1|
    =======  ================================================ ==== =============

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
    ) -> None:
        """:param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        :param shape: Target shape for the state and rate variables.
        """
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(
            kiosk, publish=["DMI", "ADMI", "REALLOC_LV", "REALLOC_ST", "REALLOC_SO"], shape=shape
        )
        self.kiosk = kiosk

        # Initialize components of the crop
        self.pheno = Phenology(day, kiosk, parvalues, shape=shape)
        self.part = Partitioning(day, kiosk, parvalues, shape=shape)
        self.assim = Assimilation(day, kiosk, parvalues, shape=shape)
        self.mres = MaintenanceRespiration(day, kiosk, parvalues, shape=shape)
        self.evtra = Evapotranspiration(day, kiosk, parvalues, shape=shape)
        self.ro_dynamics = Root_Dynamics(day, kiosk, parvalues, shape=shape)
        self.st_dynamics = Stem_Dynamics(day, kiosk, parvalues, shape=shape)
        self.so_dynamics = Storage_Organ_Dynamics(day, kiosk, parvalues, shape=shape)
        self.lv_dynamics = Leaf_Dynamics(day, kiosk, parvalues, shape=shape)

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

        # assign handler for CROP_FINISH signal
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

    @staticmethod
    def _check_carbon_balance(day, DMI, GASS, MRES, CVF, pf):
        (FR, FL, FS, FO) = pf
        checksum = (
            (GASS - MRES - (FR + (FL + FS + FO) * (1.0 - FR)) * DMI / CVF)
            * 1.0
            / (torch.clamp(GASS, min=0.0001))
        )
        if torch.any(torch.abs(checksum) >= 0.0001):
            msg = "Carbon flows not balanced on day %s\n" % day
            msg += "Checksum: %f, GASS: %f, MRES: %f\n" % (
                checksum.mean().item(),
                GASS.mean().item(),
                MRES.mean().item(),
            )
            msg += "FR,L,S,O: %5.3f,%5.3f,%5.3f,%5.3f, DMI: %f, CVF: %f\n" % (
                FR.mean().item(),
                FL.mean().item(),
                FS.mean().item(),
                FO.mean().item(),
                DMI.mean().item(),
                CVF.mean().item(),
            )
            raise exc.CarbonBalanceError(msg)

    @prepare_rates
    def calc_rates(self, day: datetime.date, drv: WeatherDataContainer) -> None:
        """Calculate the rates of change of the state variables.

        Args:
            day (datetime.date): The current date of the simulation.
            drv (WeatherDataContainer): A dictionary-like container holding
                weather data elements as key/value. The values are
                arrays or scalars. See PCSE documentation for details.
        """
        p = self.params
        r = self.rates
        k = self.kiosk

        # Phenology
        self.pheno.calc_rates(day, drv)
        crop_stage = self.pheno.get_variable("STAGE")

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # STAGE == 0 corresponds to "emerging" in the tensor encoding.
        if torch.all(crop_stage == 0):
            return

        # Potential assimilation
        PGASS = self.assim(day, drv)

        # (evapo)transpiration rates
        self.evtra(day, drv)

        # water stress reduction
        r.GASS = PGASS * k.RFTRA

        # Respiration
        PMRES = self.mres(day, drv)
        r.MRES = torch.minimum(r.GASS, PMRES)

        # Net available assimilates
        r.ASRC = r.GASS - r.MRES

        # DM partitioning factors (pf), conversion factor (CVF),
        # dry matter increase (DMI) and check on carbon balance
        pf = self.part.calc_rates(day, drv)
        CVF = 1.0 / (
            (pf.FL / p.CVL + pf.FS / p.CVS + pf.FO / p.CVO) * (1.0 - pf.FR) + pf.FR / p.CVR
        )
        r.DMI = CVF * r.ASRC
        self._check_carbon_balance(day, r.DMI, r.GASS, r.MRES, CVF, pf)

        # distribution over plant organ

        # Reallocation from stems/leaves not applicable in WOFOST72
        r.REALLOC_LV = torch.zeros_like(r.DMI)
        r.REALLOC_ST = torch.zeros_like(r.DMI)
        r.REALLOC_SO = torch.zeros_like(r.DMI)

        # Below-ground dry matter increase and root dynamics
        self.ro_dynamics.calc_rates(day, drv)
        # Aboveground dry matter increase and distribution over stems,
        # leaves, organs
        r.ADMI = (1.0 - pf.FR) * r.DMI
        self.st_dynamics.calc_rates(day, drv)
        self.so_dynamics.calc_rates(day, drv)
        self.lv_dynamics.calc_rates(day, drv)

    @prepare_states
    def integrate(self, day: datetime.date, delt=1.0) -> None:
        """Integrate the state variables using the rates of change.

        Args:
            day (datetime.date): The current date of the simulation.
            delt (float, optional): The time step for integration. Defaults to 1.0.
        """
        rates = self.rates
        states = self.states

        # crop stage before integration
        crop_stage = self.pheno.get_variable("STAGE")

        # Phenology
        self.pheno.integrate(day, delt)

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # Just run a touch() to to ensure that all state variables are available
        # in the kiosk
        # STAGE == 0 corresponds to "emerging" in the tensor encoding.
        if torch.all(crop_stage == 0):
            self.touch()
            return

        # Partitioning
        self.part.integrate(day, delt)

        # Integrate states on leaves, storage organs, stems and roots
        self.ro_dynamics.integrate(day, delt)
        self.so_dynamics.integrate(day, delt)
        self.st_dynamics.integrate(day, delt)
        self.lv_dynamics.integrate(day, delt)

        # Integrate total (living+dead) above-ground biomass of the crop
        states.TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        # total gross assimilation and maintenance respiration
        states.GASST = states.GASST + rates.GASS
        states.MREST = states.MREST + rates.MRES

        # total crop transpiration and soil evaporation
        states.CTRAT = states.CTRAT + self.kiosk.TRA
        states.CEVST = states.CEVST + self.kiosk.EVS

    @prepare_states
    def finalize(self, day: datetime.date) -> None:
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
        """Handler for setting day of finish (DOF) and reason for
        crop finishing (FINISH).
        """
        self._for_finalize["DOF"] = day
        self._for_finalize["FINISH_TYPE"] = finish_type
