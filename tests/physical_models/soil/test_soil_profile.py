"""Soil-profile moisture points and rooting weights against PCSE."""

import torch
from pcse.soil.soil_profile import SoilProfile as PcseSoilProfile
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.soil.soil_profile import SoilProfile
from .test_wofost81_snomin import _example_soil


def _close(actual, expected):
    torch.testing.assert_close(
        torch.as_tensor(actual, dtype=torch.float64),
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-7,
        atol=1e-9,
    )


def test_moisture_points_match_pcse_example_profile():
    """Field capacity, wilting point and saturation follow the PCSE example profile."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    soil = _example_soil()
    reference = PcseSoilProfile(soil)
    profile = SoilProfile(soil)
    assert len(profile) == len(reference)
    for layer, pcse_layer in zip(profile, reference, strict=True):
        for name in ("Thickness", "SM0", "SMFCF", "SMW", "WCFC"):
            _close(getattr(layer, name), getattr(pcse_layer, name))


def test_rooting_weight_matches_pcse_at_a_layer_boundary():
    """A root front on a boundary is fully in the layer above, as in PCSE."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    soil = _example_soil()
    maximum = sum(layer["Thickness"] for layer in soil["SoilProfileDescription"]["SoilLayers"])
    for depth in (10.0, 15.0):
        reference = PcseSoilProfile(soil)
        profile = SoilProfile(soil)
        reference.determine_rooting_status(depth, maximum)
        profile.determine_rooting_status(depth, maximum)
        for layer, pcse_layer in zip(profile, reference, strict=True):
            _close(layer.Wtop, pcse_layer.Wtop)
            _close(layer.Wpot, pcse_layer.Wpot)
