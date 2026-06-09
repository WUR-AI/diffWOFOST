import torch
from pcse.base import SimulationObject
from diffwofost.physical_models.crop.assimilation import WOFOST72_Assimilation as Assimilation
from diffwofost.physical_models.crop.partitioning import DVS_Partitioning as Partitioning
from diffwofost.physical_models.crop.phenology import DVS_Phenology as Phenology
from diffwofost.physical_models.override import normalize_components


class TestPhenology(SimulationObject):
    def initialize(self, day, kiosk, parvalues):
        pass


class Testpartitioning(SimulationObject):
    def initialize(self, day, kiosk, parvalues):
        pass


class TestPhenologyModel(torch.nn.Module):
    def __init__(self, test_kwarg="default_value"):
        super().__init__()
        self.test_kwarg = test_kwarg


def test_normalize_components():
    crop_component_specs = {
        "phenology": ("pheno", Phenology),
        "partitioning": ("part", Partitioning),
        "assimilation": ("assim", Assimilation),
    }
    crop_components = {
        "phenology": {
            "class": TestPhenology,
            "model": TestPhenologyModel,
            "kwargs": {"test_kwarg": "test_value"},
        },
        "partitioning": Testpartitioning,
    }

    normalized_components = normalize_components(crop_components, crop_component_specs)

    assert isinstance(normalized_components, dict)
    assert "phenology" in normalized_components
    assert "partitioning" in normalized_components
    assert "assimilation" in normalized_components
    assert normalized_components["phenology"].component_class == TestPhenology
    assert normalized_components["phenology"].model == TestPhenologyModel
    assert normalized_components["phenology"].kwargs["test_kwarg"] == "test_value"

    assert normalized_components["partitioning"].component_class == Testpartitioning
    assert normalized_components["partitioning"].model is None
    assert normalized_components["partitioning"].kwargs is None

    assert normalized_components["assimilation"].component_class == Assimilation
    assert normalized_components["assimilation"].model is None
    assert normalized_components["assimilation"].kwargs is None
