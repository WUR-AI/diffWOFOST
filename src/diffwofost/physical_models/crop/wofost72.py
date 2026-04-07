import datetime
import torch
from pcse import exceptions as exc
from pcse import signals
from pcse.base import SimulationObject
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.traitlets import Instance
from pcse.traitlets import Unicode
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
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
    COMPONENT_OVERRIDE_META_KEYS = frozenset({"class", "model", "kwargs"})

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

        The ``component_overrides`` mapping must use the canonical component
        names from ``COMPONENT_SPECS``, such as ``partitioning``,
        ``phenology``, ``assimilation``, ``maintenance_respiration``,
        ``evapotranspiration``, ``root_dynamics``, ``stem_dynamics``,
        ``storage_organ_dynamics``, and ``leaf_dynamics``.

        Each override entry may be one of the following:

        - ``None``: keep the default component class with no extra arguments.
        - a ``SimulationObject`` subclass: replace only the component class.
        - a dict containing reserved keys:
          ``class`` for the replacement component class and ``model`` for an
          optional ML model object.
        - any additional keys in that dict are forwarded as keyword arguments
          to the component constructor. A nested ``kwargs`` dict is also
          accepted for backward-compatible explicit constructor kwargs.

        ML-backed overrides are supported by passing a ``model`` object in the
        override entry. When no model is provided, the component is constructed
        with ``(day, kiosk, parvalues, shape=..., **kwargs)`` so the component
        reads crop parameters from the ``ParameterProvider`` as usual. When a
        model is provided, the component is constructed with
        ``(day, kiosk, model, shape=..., **kwargs)`` instead. This allows a
        replacement component such as a neural partitioning module to consume a
        trained or trainable PyTorch model while the rest of WOFOST remains
        unchanged.
        """
        self.params = self.Parameters(parvalues, shape=shape)
        self.rates = self.RateVariables(
            kiosk, publish=["DMI", "ADMI", "REALLOC_LV", "REALLOC_ST", "REALLOC_SO"], shape=shape
        )
        self.kiosk = kiosk
        component_overrides = self._normalize_component_overrides(component_overrides)

        # Initialize components of the crop
        for component_name, (attribute_name, _) in self.COMPONENT_SPECS.items():
            setattr(
                self,
                attribute_name,
                self._initialize_component(
                    component_name,
                    day,
                    kiosk,
                    parvalues,
                    shape=shape,
                    component_overrides=component_overrides,
                ),
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

        # assign handler for CROP_FINISH signal
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)

    def _normalize_component_overrides(
        self,
        component_overrides: dict | None = None,
    ) -> dict:
        """Normalize override definitions to a single internal structure.

        Args:
            component_overrides: Raw override mapping passed to
                :meth:`initialize`.

        Returns:
            A dictionary keyed by canonical component names. Each value is a
            compact override dictionary containing ``class``, ``model``, and
            optionally ``kwargs`` for constructor keyword arguments.

        Notes:
            This method allows a concise override syntax for ML-enabled
            components. For example, callers can pass
            ``{"partitioning": {"class": MyPartitioningWrapper,
            "model": my_torch_model, "dropout": 0.0}}`` and the extra
            ``dropout`` key will be moved into ``kwargs`` automatically.
        """
        normalized_overrides = {}
        for component_name, override in (component_overrides or {}).items():
            if component_name not in self.COMPONENT_SPECS:
                msg = f"Unknown Wofost72 component override: {component_name}"
                raise KeyError(msg)
            if override is None:
                normalized_overrides[component_name] = {}
            elif isinstance(override, dict):
                override_dict = dict(override)
                explicit_kwargs = override_dict.pop("kwargs", None)
                constructor_kwargs = {
                    key: value
                    for key, value in override_dict.items()
                    if key not in self.COMPONENT_OVERRIDE_META_KEYS
                }
                normalized_override = {
                    key: value
                    for key, value in override_dict.items()
                    if key in self.COMPONENT_OVERRIDE_META_KEYS - {"kwargs"}
                }
                if explicit_kwargs is not None:
                    constructor_kwargs = {**dict(explicit_kwargs), **constructor_kwargs}
                if constructor_kwargs:
                    normalized_override["kwargs"] = constructor_kwargs
                normalized_overrides[component_name] = normalized_override
            else:
                normalized_overrides[component_name] = {"class": override}

        return normalized_overrides

    def _initialize_component(
        self,
        component_name: str,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: ParameterProvider,
        shape: tuple | torch.Size | None = None,
        component_overrides: dict | None = None,
    ) -> SimulationObject:
        """Build one embedded WOFOST component from the override definition.

        Args:
            component_name: Canonical component name to instantiate.
            day: Current simulation day.
            kiosk: Variable kiosk shared across crop components.
            parvalues: Physical-model parameter provider.
            shape: Optional tensor broadcast shape for the component.
            component_overrides: Normalized component override mapping.

        Returns:
            The instantiated simulation component.

        The constructor call depends on whether the override provides a
        ``model``. Default physical components expect the parameter provider as
        their third positional argument, whereas ML-backed wrappers typically
        expect a model object there instead. This method centralizes that
        dispatch so callers only need to describe the override declaratively.
        """
        _, default_component_class = self.COMPONENT_SPECS[component_name]
        override = (
            {} if component_overrides is None else component_overrides.get(component_name, {})
        )

        component_class = override.get("class", default_component_class)
        component_kwargs = dict(override.get("kwargs", {}))
        component_model = override.get("model")

        if component_model is None:
            return component_class(day, kiosk, parvalues, shape=shape, **component_kwargs)

        return component_class(day, kiosk, component_model, shape=shape, **component_kwargs)

    @staticmethod
    def _check_carbon_balance(day, DMI, GASS, MRES, CVF, pf):
        """Check that carbon flows are balanced on the current day."""
        # [!] This check runs every day and can be costly on GPU, so we can think about removing it
        # and sanitize the parameters instead.
        (FR, FL, FS, FO) = pf
        checksum = (
            (GASS - MRES - (FR + (FL + FS + FO) * (1.0 - FR)) * DMI / CVF)
            * 1.0
            / (torch.clamp(GASS, min=0.0001))
        )
        if torch.any(torch.abs(checksum) >= 0.0001):
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

        # if before emergence there is no need to continue
        # because only the phenology is running.
        # TODO: revisit this when fixing #60
        if torch.all(self.pheno.states.STAGE == 0):
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

        # Integrate total (living+dead) above-ground biomass of the crop
        states.TAGP = self.kiosk.TWLV + self.kiosk.TWST + self.kiosk.TWSO

        # total gross assimilation and maintenance respiration
        states.GASST = states.GASST + rates.GASS
        states.MREST = states.MREST + rates.MRES

        # total crop transpiration and soil evaporation
        states.CTRAT = states.CTRAT + self.kiosk.TRA
        states.CEVST = states.CEVST + self.kiosk.EVS

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
