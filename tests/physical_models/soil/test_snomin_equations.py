"""Equation checks of SNOMIN rates against PCSE's own functions.

The 31-day trajectory can agree while two wrong responses cancel. These tests
call the PCSE rate functions and the torch ports on the same layers, then step
the pools and call them again.
"""

import numpy as np
import torch
from pcse.soil.snomin import SNOMIN
from diffwofost.physical_models.soil.snomin import _age_increase
from diffwofost.physical_models.soil.snomin import _application_mineral
from diffwofost.physical_models.soil.snomin import _application_organic
from diffwofost.physical_models.soil.snomin import _available_nitrogen
from diffwofost.physical_models.soil.snomin import _denitrification
from diffwofost.physical_models.soil.snomin import _deposition
from diffwofost.physical_models.soil.snomin import _dissimilation
from diffwofost.physical_models.soil.snomin import _limit_immobilisation
from diffwofost.physical_models.soil.snomin import _nitrification
from diffwofost.physical_models.soil.snomin import _solute_flow
from diffwofost.physical_models.soil.snomin import _uptake

_KSORP = 1.4
_KNIT = 0.25
_KDENIT = 0.06
_MRCDIS = 0.002
_WFPS_CRIT = 0.8
_FASDIS = 0.45
_CN_BIO = 8.0


class _Layer:
    """The attributes PCSE's SNOMIN rate functions read from a soil layer."""

    def __init__(self, thickness_cm, bulk_density, saturated, ph):
        self.Thickness = float(thickness_cm)
        self.Thickness_m = float(thickness_cm) * 0.01
        self.RHOD_kg_per_m3 = float(bulk_density)
        self.SM0 = float(saturated)
        self.Soil_pH = float(ph)


def _profile():
    return [
        _Layer(10, 1400, 0.45, 6.5),
        _Layer(20, 1500, 0.42, 5.2),
        _Layer(15, 1600, 0.40, 7.1),
    ]


def _same(actual, expected):
    got = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    want = torch.as_tensor(np.asarray(expected, dtype=np.float64).reshape(-1), dtype=torch.float64)
    # Numpy and torch evaluate exp and pow a few ulps apart.
    torch.testing.assert_close(got, want, rtol=1e-7, atol=1e-9)


def _ammonium():
    return SNOMIN.SoilInorganicNModel.SoilAmmoniumNModel()


def _nitrate():
    return SNOMIN.SoilInorganicNModel.SoilNNitrateModel()


def _organic():
    return SNOMIN.SoilOrganicNModel()


def test_nitrification_temperature_response():
    """The logistic temperature factor is inside the nitrification rate."""
    layer = _profile()[0]
    ammonium = 0.0015
    moisture = 0.22
    reference = _ammonium()
    for temperature in (-5.0, 0.0, 10.0, 17.0, 25.0, 35.0, 45.0):
        expected = reference.calculate_nitrification_rate(
            _KNIT,
            _KSORP,
            layer.Thickness_m,
            ammonium,
            layer.RHOD_kg_per_m3,
            moisture,
            layer.SM0,
            temperature,
        )
        port = _nitrification(
            [layer],
            torch.tensor(_KNIT),
            torch.tensor(_KSORP),
            torch.tensor([ammonium]),
            torch.tensor([moisture]),
            torch.tensor(temperature),
        )
        _same(port, expected)


def test_nitrification_moisture_response():
    """Dry, intermediate and near-saturated water-filled pore space."""
    layer = _profile()[1]
    ammonium = 0.002
    temperature = 18.0
    reference = _ammonium()
    for moisture in (0.05, 0.15, 0.25, 0.35, 0.41):
        expected = reference.calculate_nitrification_rate(
            _KNIT,
            _KSORP,
            layer.Thickness_m,
            ammonium,
            layer.RHOD_kg_per_m3,
            moisture,
            layer.SM0,
            temperature,
        )
        port = _nitrification(
            [layer],
            torch.tensor(_KNIT),
            torch.tensor(_KSORP),
            torch.tensor([ammonium]),
            torch.tensor([moisture]),
            torch.tensor(temperature),
        )
        _same(port, expected)


def test_denitrification_switches_at_critical_water_content():
    """No gaseous loss below WFPS_CRIT, and the quadratic above it."""
    layer = _profile()[2]
    nitrate = 0.0012
    carbon = 0.0004
    temperature = 16.0
    reference = _nitrate()
    saturated = layer.SM0
    for fraction in (0.5, 0.8, 0.9, 0.99):
        moisture = fraction * saturated
        expected = reference.calculate_denitrification_rate(
            layer.Thickness_m,
            nitrate,
            _KDENIT,
            _MRCDIS,
            carbon,
            moisture,
            saturated,
            temperature,
            _WFPS_CRIT,
        )
        port = _denitrification(
            [layer],
            torch.tensor(_KDENIT),
            torch.tensor(_MRCDIS),
            torch.tensor(_WFPS_CRIT),
            torch.tensor([nitrate]),
            torch.tensor([carbon]),
            torch.tensor([moisture]),
            torch.tensor(temperature),
        )
        _same(port, expected)


def test_nitrified_ammonium_is_the_nitrate_source():
    """PCSE copies the nitrification rate into the nitrate pool."""
    profile = _profile()
    ammonium = np.array([0.001, 0.0016, 0.0004])
    nitrate = np.array([0.0008, 0.0011, 0.0003])
    moisture = np.array([0.12, 0.28, 0.36])
    carbon = np.array([[0.0002, 0.0005, 0.0001], [0.0001, 0.0, 0.0003]])
    inorganic = SNOMIN.SoilInorganicNModel()
    mineralization, nitrification, nitrate_from_ammonium, denitrification = (
        inorganic.calculate_reaction_rates(
            profile,
            _KDENIT,
            _KNIT,
            _KSORP,
            _MRCDIS,
            ammonium,
            nitrate,
            carbon,
            carbon,
            moisture,
            18.0,
            _WFPS_CRIT,
        )
    )
    _same(nitrate_from_ammonium, nitrification)
    port_nitrification = _nitrification(
        profile,
        torch.tensor(_KNIT),
        torch.tensor(_KSORP),
        torch.tensor(ammonium),
        torch.tensor(moisture),
        torch.tensor(18.0),
    )
    port_denitrification = _denitrification(
        profile,
        torch.tensor(_KDENIT),
        torch.tensor(_MRCDIS),
        torch.tensor(_WFPS_CRIT),
        torch.tensor(nitrate),
        torch.tensor(carbon.sum(axis=0)),
        torch.tensor(moisture),
        torch.tensor(18.0),
    )
    _same(port_nitrification, nitrification)
    _same(port_denitrification, denitrification)
    _same(mineralization, carbon.sum(axis=0))


def test_dissimilation_matches_janssen_and_limits_immobilisation():
    """Organic loss follows Janssen, and a short ammonium pool rescales it."""
    profile = _profile()
    age = np.array([[400.0, 1200.0, 2500.0], [80.0, 900.0, 40.0]])
    organic_matter = np.array([[1.2, 0.4, 0.0], [0.2, 0.8, 0.3]])
    organic_nitrogen = np.array([[0.06, 0.02, 0.0], [0.01, 0.05, 0.015]])
    pf = np.array([2.0, 3.4, 4.5])
    ph = np.array([layer.Soil_pH for layer in profile])
    temperature = 14.0
    reference = _organic().calculate_dissimilation_rates(
        age, _CN_BIO, _FASDIS, organic_nitrogen, organic_matter, pf, ph, temperature
    )
    port = _dissimilation(
        torch.tensor(age),
        torch.tensor(organic_matter),
        torch.tensor(organic_nitrogen),
        torch.tensor(_CN_BIO),
        torch.tensor(_FASDIS),
        torch.tensor(pf),
        torch.tensor(ph),
        torch.tensor(temperature),
    )
    for got, want in zip(port, reference, strict=True):
        _same(got, want)

    ammonium_left = np.array([0.001, 0.02, 0.0003])
    # Layer 0 has a positive amendment sum and still trips the PCSE test,
    # because nitrification exceeds the ammonium left after that sum.
    nitrification = np.array([0.004, 0.001, 0.0001])
    mineralization = reference[2].sum(axis=0)
    expected_rate = mineralization.copy()
    expected_organic = reference[2].copy()
    for layer in range(3):
        if ammonium_left[layer] + (mineralization[layer] - nitrification[layer]) < 0:
            expected_rate[layer] = ammonium_left[layer] - nitrification[layer]
            total = reference[2][:, layer].sum()
            expected_organic[:, layer] = (expected_rate[layer] / total) * reference[2][:, layer]
    limited, scaled = _limit_immobilisation(
        torch.tensor(ammonium_left),
        torch.tensor(mineralization),
        torch.tensor(nitrification),
        torch.tensor(reference[2]),
        1.0,
    )
    _same(limited, expected_rate)
    _same(scaled, expected_organic)


def test_fertiliser_is_split_by_application_depth():
    """Mineral and organic additions use the same depth fractions as PCSE."""
    profile = _profile()
    amount = 120.0
    depth = 25.0
    inorganic = SNOMIN.SoilInorganicNModel()
    organic = _organic()
    expected_nh4, expected_no3 = inorganic.calculate_N_application_amounts(
        profile, amount, depth, 0.4, 0.3
    )
    expected_organic, expected_carbon, expected_nitrogen = organic.calculate_application_rates(
        profile, amount, depth, 12.0, 0.7
    )
    nh4, no3 = _application_mineral(
        profile, torch.tensor(amount), torch.tensor(depth), torch.tensor(0.4), torch.tensor(0.3)
    )
    org, carbon, nitrogen = _application_organic(
        profile, torch.tensor(amount), torch.tensor(depth), torch.tensor(12.0), torch.tensor(0.7)
    )
    _same(nh4, expected_nh4)
    _same(no3, expected_no3)
    _same(org, expected_organic)
    _same(carbon, expected_carbon)
    _same(nitrogen, expected_nitrogen)
    none = _application_organic(
        profile, torch.tensor(amount), torch.tensor(depth), torch.tensor(0.0), torch.tensor(0.7)
    )
    _same(none[2], np.zeros(len(profile)))


def test_available_nitrogen_sums_only_the_rooted_part_of_each_layer():
    """NAVAIL is the rooted share of soluble ammonium plus nitrate."""
    profile = _profile()
    ammonium = np.array([0.002, 0.0015, 0.0008])
    nitrate = np.array([0.001, 0.0007, 0.0002])
    moisture = np.array([0.18, 0.25, 0.30])
    for rooting_depth in (0.0, 0.10, 0.15, 0.30, 0.50):
        expected = SNOMIN.SoilInorganicNModel().calculate_NAVAIL(
            profile, _KSORP, ammonium, nitrate, rooting_depth, moisture
        )
        port = _available_nitrogen(
            profile,
            torch.tensor(_KSORP),
            torch.tensor(ammonium),
            torch.tensor(nitrate),
            torch.tensor(rooting_depth),
            torch.tensor(moisture),
        )
        _same(port, expected)


def test_next_day_uses_the_updated_layer_state():
    """Uptake, reactions, deposition and flow update the pools the next day sees."""
    profile = _profile()
    inorganic = SNOMIN.SoilInorganicNModel()
    organic_model = _organic()
    ammonium = np.array([0.003, 0.002, 0.001])
    nitrate = np.array([0.0015, 0.001, 0.0005])
    moisture = np.array([0.16, 0.27, 0.34])
    flow = np.array([0.002, 0.004, -0.001, 0.0015])
    infiltration = 0.003
    demand = 0.001
    rooting = 0.18
    age = np.array([[500.0, 1500.0, 80.0]])
    organic_matter = np.array([[0.9, 0.5, 0.2]])
    organic_nitrogen = np.array([[0.04, 0.02, 0.01]])
    pf = np.array([2.2, 3.1, 4.4])
    ph = np.array([layer.Soil_pH for layer in profile])
    carbon_dissimilation = np.zeros((1, 3))

    for _day in range(2):
        uptake_nh4, uptake_no3 = inorganic.calculate_N_uptake_rates(
            profile, 1.0, _KSORP, demand, ammonium, nitrate, rooting, moisture
        )
        port_uptake = _uptake(
            profile,
            1.0,
            torch.tensor(_KSORP),
            torch.tensor(demand),
            torch.tensor(ammonium),
            torch.tensor(nitrate),
            torch.tensor(rooting),
            torch.tensor(moisture),
        )
        _same(port_uptake[0], uptake_nh4)
        _same(port_uptake[1], uptake_no3)

        ammonium_left = ammonium - uptake_nh4
        nitrate_left = nitrate - uptake_no3
        mineralization, nitrification, nitrate_gain, denitrification = (
            inorganic.calculate_reaction_rates(
                profile,
                _KDENIT,
                _KNIT,
                _KSORP,
                _MRCDIS,
                ammonium_left,
                nitrate_left,
                carbon_dissimilation,
                np.zeros_like(carbon_dissimilation),
                moisture,
                15.0,
                _WFPS_CRIT,
            )
        )
        _same(
            _nitrification(
                profile,
                torch.tensor(_KNIT),
                torch.tensor(_KSORP),
                torch.tensor(ammonium_left),
                torch.tensor(moisture),
                torch.tensor(15.0),
            ),
            nitrification,
        )
        _same(
            _denitrification(
                profile,
                torch.tensor(_KDENIT),
                torch.tensor(_MRCDIS),
                torch.tensor(_WFPS_CRIT),
                torch.tensor(nitrate_left),
                torch.tensor(carbon_dissimilation.sum(axis=0)),
                torch.tensor(moisture),
                torch.tensor(15.0),
            ),
            denitrification,
        )
        deposition_nh4, deposition_no3 = inorganic.calculate_deposition_rates(
            profile, infiltration, ammonium, 0.8, nitrate, 1.2
        )
        port_deposition = _deposition(
            torch.tensor(infiltration), torch.tensor(0.8), torch.tensor(1.2), torch.tensor(ammonium)
        )
        _same(port_deposition[0], deposition_nh4)
        _same(port_deposition[1], deposition_no3)

        ammonium_after = ammonium_left + mineralization + deposition_nh4 - nitrification
        nitrate_after = nitrate_left + nitrate_gain + deposition_no3 - denitrification
        inflow_nh4, outflow_nh4, inflow_no3, outflow_no3 = inorganic.calculate_flow_rates(
            profile, flow, _KSORP, ammonium_after, nitrate_after, moisture
        )
        thickness = np.array([layer.Thickness_m for layer in profile])
        bulk_density = np.array([layer.RHOD_kg_per_m3 for layer in profile])
        port_flow_nh4 = _solute_flow(
            torch.tensor(flow),
            torch.tensor(ammonium_after / ((_KSORP * bulk_density + moisture) * thickness)),
        )
        port_flow_no3 = _solute_flow(
            torch.tensor(flow), torch.tensor(nitrate_after / (thickness * moisture))
        )
        _same(port_flow_nh4[0], inflow_nh4)
        _same(port_flow_nh4[1], outflow_nh4)
        _same(port_flow_no3[0], inflow_no3)
        _same(port_flow_no3[1], outflow_no3)

        rate_nh4 = (
            mineralization + deposition_nh4 - nitrification - uptake_nh4 + inflow_nh4 - outflow_nh4
        )
        rate_no3 = (
            nitrate_gain + deposition_no3 - denitrification - uptake_no3 + inflow_no3 - outflow_no3
        )
        ammonium = ammonium + rate_nh4
        nitrate = nitrate + rate_no3

        dissimilation = organic_model.calculate_dissimilation_rates(
            age, _CN_BIO, _FASDIS, organic_nitrogen, organic_matter, pf, ph, 15.0
        )
        port_dissimilation = _dissimilation(
            torch.tensor(age),
            torch.tensor(organic_matter),
            torch.tensor(organic_nitrogen),
            torch.tensor(_CN_BIO),
            torch.tensor(_FASDIS),
            torch.tensor(pf),
            torch.tensor(ph),
            torch.tensor(15.0),
        )
        ageing = organic_model.calculate_apparent_age_increase_rate(age, 1.0, pf, ph, 15.0)
        port_ageing = _age_increase(
            torch.tensor(age), 1.0, torch.tensor(pf), torch.tensor(ph), torch.tensor(15.0)
        )
        for got, want in zip(port_dissimilation, dissimilation, strict=True):
            _same(got, want)
        _same(port_ageing, ageing)
        organic_matter = organic_matter - dissimilation[0]
        organic_nitrogen = organic_nitrogen - dissimilation[2]
        age = age + ageing
        carbon_dissimilation = dissimilation[1]
