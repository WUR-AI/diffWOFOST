import datetime
from pcse.base import SimulationObject
from pcse.traitlets import Instance
from diffwofost.physical_models.base.simulationobject import initialize_components
from diffwofost.physical_models.crop.partitioning import DVS_Partitioning as Partitioning
from diffwofost.physical_models.crop.phenology import DVS_Phenology as Phenology
from diffwofost.physical_models.override import ComponentOverride
from diffwofost.physical_models.variablekiosk import VariableKiosk


class TestSimulationObject(SimulationObject):
    pheno = Instance(SimulationObject)
    part = Instance(SimulationObject)
    COMPONENT_SPECS = {
        "phenology": ("pheno", Phenology),
        "partitioning": ("part", Partitioning),
    }

    def initialize(self, day, kiosk):
        pass


class TestPartioning(SimulationObject):
    def initialize(self, day, kiosk, parvalues, test_kwarg="default_value"):
        self._test_kwarg = test_kwarg


def test_initialize_components():
    day = datetime.date(2000, 1, 1)
    kiosk = VariableKiosk(
        [
            {"DAY": day, "DVS": 0.1},
        ]
    )
    parvalues = {
        # Phenology
        "TSUMEM": 50.0,
        "TBASEM": 0.0,
        "TEFFMX": 35.0,
        "TSUM1": 500.0,
        "TSUM2": 600.0,
        "IDSL": 0.5,
        "DLO": 0.5,
        "DLC": 0.5,
        "DVSI": 0.0,
        "DVSEND": 1.95,
        "DTSMTB": [0.0, 0.0, 35.0, 35.0, 45.0, 35.0],
        "CROP_START_TYPE": "sowing",
        "CROP_END_TYPE": "maturity",
        # Partitioning
        "FRTB": [[0.0, 0.3, 2.0, 0.1]],
        "FLTB": [[0.0, 0.85, 1.0, 0.5, 1.3, 0.05, 1.57, 0.05, 1.92, 0.05, 2.0, 0.05]],
        "FSTB": [[0.0, 0.15, 1.0, 0.5, 1.3, 0.10, 1.57, 0.10, 1.92, 0.05, 2.0, 0.05]],
        "FOTB": [[0.0, 0.00, 1.0, 0.0, 1.3, 0.85, 1.57, 0.85, 1.92, 0.90, 2.0, 0.90]],
    }

    simulation_object = TestSimulationObject(day, kiosk)

    initialize_components(
        simulation_object=simulation_object,
        day=day,
        kiosk=kiosk,
        parvalues=parvalues,
        shape=(1,),
    )

    assert simulation_object is simulation_object
    assert hasattr(simulation_object, "pheno")
    assert isinstance(simulation_object.pheno, Phenology)
    assert hasattr(simulation_object, "part")
    assert isinstance(simulation_object.part, Partitioning)


def test_initialize_components_with_overrides():
    day = datetime.date(2000, 1, 1)
    kiosk = VariableKiosk(
        [
            {"DAY": day, "DVS": 0.1},
        ]
    )
    parvalues = {
        # Phenology
        "TSUMEM": 50.0,
        "TBASEM": 0.0,
        "TEFFMX": 35.0,
        "TSUM1": 500.0,
        "TSUM2": 600.0,
        "IDSL": 0.5,
        "DLO": 0.5,
        "DLC": 0.5,
        "DVSI": 0.0,
        "DVSEND": 1.95,
        "DTSMTB": [0.0, 0.0, 35.0, 35.0, 45.0, 35.0],
        "CROP_START_TYPE": "sowing",
        "CROP_END_TYPE": "maturity",
    }

    simulation_object = TestSimulationObject(day, kiosk)
    component_overrides = {
        "phenology": ComponentOverride(
            component_class=Phenology,
            model=None,
            kwargs=None,
        ),
        "partitioning": ComponentOverride(
            component_class=TestPartioning,
            model=None,
            kwargs={"test_kwarg": "overridden_value"},
        ),
    }

    initialize_components(
        simulation_object=simulation_object,
        day=day,
        kiosk=kiosk,
        parvalues=parvalues,
        component_overrides=component_overrides,
    )

    assert simulation_object.part._test_kwarg == "overridden_value"
