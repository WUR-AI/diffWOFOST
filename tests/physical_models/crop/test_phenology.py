import warnings
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.phenology import DVS_Phenology
from diffwofost.physical_models.test import EngineTestHelper
from diffwofost.physical_models.test import calculate_numerical_grad
from diffwofost.physical_models.test import get_test_data
from diffwofost.physical_models.test import prepare_engine_input
from .. import phy_data_folder

phenology_config = Configuration(
    CROP=DVS_Phenology,
    OUTPUT_VARS=["DVR", "DVS", "TSUM", "TSUME", "VERN"],
)

phenology_config_with_activity = Configuration(
    CROP=DVS_Phenology,
    OUTPUT_VARS=["DVR", "DVS", "TSUM", "TSUME", "VERN", "IS_ACTIVE", "STAGE", "DOS"],
)

# Phenology-related crop model parameters
CROP_MODEL_PARAMS = [
    "TSUMEM",
    "TBASEM",
    "TEFFMX",
    "TSUM1",
    "TSUM2",
    "IDSL",
    "DLO",
    "DLC",
    "DVSI",
    "DVSEND",
    "DTSMTB",
    "VERNRTB",
    "VERNSAT",
    "VERNBASE",
    "VERNDVS",
]


class DVS_PhenologyForPCSE(DVS_Phenology):
    """DVS_Phenology subclass that adds a get_variable() override required by
    PCSE's wofost72, which calls self.pheno.get_variable("STAGE") and expects
    a string result.  Only used in tests that patch pcse.crop.wofost72.Phenology.
    """

    def get_variable(self, varname):
        if varname == "STAGE":
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
        return super().get_variable(varname)


def assert_reference_match(reference, model, expected_precision):
    assert reference["DAY"] == model["day"]
    for var, precision in expected_precision.items():
        # [!] These are not 'State variables' and are not stored in model output
        if var in ["VERNFAC", "VERNR"]:
            continue

        # for some data tests, both reference and model can have None values
        if reference[var] is None and model[var] is None:
            continue
        ref_t = torch.as_tensor(reference[var])
        model_v = model[var]
        model_t = (
            model_v.detach().cpu()
            if isinstance(model_v, torch.Tensor)
            else torch.as_tensor(model_v)
        )
        assert torch.all(torch.abs(ref_t - model_t) < precision)


def get_test_diff_phenology_model():
    test_data_url = f"{phy_data_folder}/test_phenology_wofost72_05.yaml"
    test_data = get_test_data(test_data_url)
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
        prepare_engine_input(test_data, CROP_MODEL_PARAMS)
    )
    return DiffPhenologyDynamics(
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        phenology_config,
    )


def _prepare_batch_phenology_input(n, tsum1=100.0, tsum2=100.0, tsumem=30.0):
    """Prepare an n-element phenology batch that matures quickly.

    IDSL is disabled (no daylength/vernalisation gating) and TSUMEM/TSUM1/TSUM2
    are set to small values so the whole crop cycle completes in a few weeks, well within the
    available weather data window even with a staggered CROP_START_DATE.
    """
    test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
    test_data = get_test_data(test_data_url)

    (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        _,
    ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS, meteo_range_checks=False)

    for param in CROP_MODEL_PARAMS:
        if param == "DTSMTB":
            repeated = crop_model_params_provider[param].repeat(n, 1)
        else:
            repeated = crop_model_params_provider[param].repeat(n)
        crop_model_params_provider.set_override(param, repeated, check=False)

    dtype = crop_model_params_provider["TSUM1"].dtype
    crop_model_params_provider.set_override("IDSL", torch.zeros(n, dtype=dtype), check=False)
    crop_model_params_provider.set_override(
        "TSUMEM", torch.full((n,), tsumem, dtype=dtype), check=False
    )
    crop_model_params_provider.set_override(
        "TSUM1", torch.full((n,), tsum1, dtype=dtype), check=False
    )
    crop_model_params_provider.set_override(
        "TSUM2", torch.full((n,), tsum2, dtype=dtype), check=False
    )

    day0 = weather_data_provider[0]["DAY"].toordinal()
    return crop_model_params_provider, weather_data_provider, agro_management_inputs, day0


class DiffPhenologyDynamics(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config,
    ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config = config
        self.engine = EngineTestHelper(config=self.config)

    def forward(self, params_dict):
        # pass new value of parameters to the model
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = self.engine.setup(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        # Collect phenology outputs analogous to leaf dynamics test
        output_vars = ["DVS", "TSUM", "TSUME"]
        return {var: torch.stack([item[var] for item in results]) for var in output_vars}


@pytest.mark.usefixtures("fast_mode")
class TestPhenologyDynamics:
    phenology_data_urls = [
        f"{phy_data_folder}/test_phenology_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # assume 44 test files
    ]
    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    @pytest.mark.parametrize("test_data_url", phenology_data_urls)
    def test_phenology_with_testengine(self, test_data_url, device):
        test_data = get_test_data(test_data_url)

        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS)

        engine = EngineTestHelper(config=phenology_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            for var in expected_precision.keys():
                value = model.get(var)
                if isinstance(value, torch.Tensor):
                    assert value.device.type == device, f"{var} should be on {device}"
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert_reference_match(reference, model_cpu, expected_precision)

    @pytest.mark.parametrize(
        "param",
        [
            "TSUMEM",
            "TBASEM",
            "TEFFMX",
            "TSUM1",
            "TSUM2",
            "IDSL",
            "DLO",
            "DLC",
            "DVSI",
            "DVSEND",
            "DTSMTB",
            "VERNSAT",
            "VERNBASE",
            "VERNDVS",
            "TEMP",
        ],
    )
    def test_phenology_with_one_parameter_vector(self, param, device):
        # pick a test case with vernalisation to have all the parameters
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS, meteo_range_checks=False)

        if param == "TEMP":
            if device == "cuda":
                pytest.skip("Weather parameter vector tests are CPU-only")
            shape = (10,)

            def broadcast(wdp):
                for weather_data in wdp:
                    out = {}
                    for k, v in weather_data.items():
                        if isinstance(v, torch.Tensor):
                            out[k] = torch.broadcast_to(v, shape)
                        else:
                            out[k] = v
                    yield out

            weather_data_provider = broadcast(weather_data_provider)
        elif param == "DTSMTB":
            repeated = crop_model_params_provider[param].repeat(10, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(config=phenology_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)

    @pytest.mark.parametrize(
        "param,delta",
        [
            ("TSUMEM", 1.0),
            ("TBASEM", 1.0),
            ("TEFFMX", 1.0),
            ("TSUM1", 1.0),
            ("TSUM2", 1.0),
            ("DVSI", 0.1),
            ("DTSMTB", 1.0),
            ("VERNSAT", 1.0),
            ("VERNBASE", 0.5),
            ("VERNDVS", 0.1),
        ],
    )
    def test_phenology_with_different_parameter_values(self, param, delta, device):
        # we dont test IDSL,DLO, DLC, DVSEND because these paramaters controls the
        # simulation duration
        # TODO: revisit this choice when Engine is fixed
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)

        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS)

        test_value = crop_model_params_provider[param]
        if param == "DTSMTB":
            # AfgenTrait parameters need to have shape (N, M)
            # DTSMTB is increase in tempearture, so avoid negative values
            non_zeros_mask = test_value != 0
            param_vec = torch.stack([test_value + non_zeros_mask * delta, test_value])
        else:
            param_vec = torch.stack([test_value - delta, test_value + delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(config=phenology_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            # keep original special case using last element
            for var, precision in expected_precision.items():
                # [!] These are not 'State variables' and are not stored in model output
                if var in ["VERNFAC", "VERNR"]:
                    continue

                # for some data tests, both reference and model can have None values
                if reference[var] is None and model[var] is None:
                    continue
                assert torch.all(torch.abs(reference[var] - model[var][-1]) < precision)

    def test_phenology_with_multiple_parameter_vectors(self, device):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)

        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS)

        for param in CROP_MODEL_PARAMS:
            if param in ("DTSMTB", "VERNRTB"):
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(config=phenology_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)

    def test_phenology_with_multiple_parameter_arrays(self, device):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS, meteo_range_checks=False)

        for param in CROP_MODEL_PARAMS:
            if param in ("DTSMTB", "VERNRTB"):
                repeated = crop_model_params_provider[param].repeat(30, 5, 1)
            else:
                repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(config=phenology_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)
            assert all(
                model[var].shape == (30, 5)
                for var in expected_precision.keys()
                if var not in ["VERNFAC", "VERNR"]
            )

    def test_phenology_with_incompatible_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)

        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS)

        crop_model_params_provider.set_override(
            "TSUM1", crop_model_params_provider["TSUM1"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "TSUM2", crop_model_params_provider["TSUM2"].repeat(5), check=False
        )

        engine = EngineTestHelper(config=phenology_config)
        with pytest.raises(ValueError):
            engine.setup(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
            )

    def test_phenology_with_incompatible_weather_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)

        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS, meteo_range_checks=False)

        crop_model_params_provider.set_override(
            "TSUM1", crop_model_params_provider["TSUM1"].repeat(10), check=False
        )

        # Broadcast weather variables to a shape that does not match the parameters
        shape = (5,)

        def broadcast(wdp):
            for weather_data in wdp:
                out = {}
                for k, v in weather_data.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = torch.broadcast_to(v, shape)
                    else:
                        out[k] = v
                yield out

        weather_data_provider = broadcast(weather_data_provider)

        engine = EngineTestHelper(config=phenology_config)
        with pytest.raises(ValueError):
            engine.setup(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_phenology(self, test_data_url, monkeypatch):
        test_data = get_test_data(test_data_url)

        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, CROP_MODEL_PARAMS, return_weather_data_provider=True)
        )
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        # Keep this integration test on CPU.
        monkeypatch.setattr(DVS_Phenology, "device", "cpu")
        monkeypatch.setattr(DVS_Phenology, "dtype", torch.float64)

        with patch("pcse.crop.wofost72.Phenology", DVS_PhenologyForPCSE):
            model = Wofost72_PP(
                crop_model_params_provider, weather_data_provider, agro_management_inputs
            )
            model.run_till_terminate()
            actual_results = model.get_output()

            assert len(actual_results) == len(expected_results)
            for reference, model_day in zip(expected_results, actual_results, strict=False):
                assert_reference_match(reference, model_day, expected_precision)

    def test_crop_start_date_default_is_backward_compatible(self, device):
        # CROP_START_DATE is never overridden here: relies entirely on the
        # injected -1 default, so the run must reproduce the reference exactly,
        # with IS_ACTIVE True for every output day.
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = prepare_engine_input(test_data, CROP_MODEL_PARAMS)

        engine = EngineTestHelper(config=phenology_config_with_activity)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)
            assert bool(model["IS_ACTIVE"])

    def test_phenology_staggered_crop_start_date(self, device):
        n = 3
        offsets = [0, 5, 15]  # offsets for start date of each element
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            day0,
        ) = _prepare_batch_phenology_input(n)

        start_dates = torch.tensor(
            [day0 + offset for offset in offsets], dtype=torch.int64, device=device
        )
        crop_model_params_provider.set_override("CROP_START_DATE", start_dates, check=False)

        engine = EngineTestHelper(config=phenology_config_with_activity)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        engine.run_till_terminate()
        results = engine.get_output()

        is_active = torch.stack([r["IS_ACTIVE"] for r in results]).cpu()
        dvs = torch.stack([r["DVS"] for r in results]).cpu()
        dos = torch.stack([r["DOS"] for r in results]).cpu()

        # element 0: active from the very first day (offset = 0)
        assert torch.all(is_active[:, 0])
        assert not torch.any(torch.isnan(dvs[:, 0]))

        # element 1 and 2: inactive (DVS NaN) until its own start day, then active
        for el in (1, 2):
            assert not torch.any(is_active[: offsets[el], el])
            assert torch.all(torch.isnan(dvs[: offsets[el], el]))
            assert torch.all(is_active[offsets[el] :, el])
            assert not torch.any(torch.isnan(dvs[offsets[el] :, el]))
            assert dos[offsets[el], el] == day0 + offsets[el]

    def test_phenology_matured_element_keeps_reporting_frozen_values(self, device):
        n = 2
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _,
        ) = _prepare_batch_phenology_input(n)

        # element 0 matures much sooner than element 1
        tsum1 = crop_model_params_provider["TSUM1"].clone()
        tsum2 = crop_model_params_provider["TSUM2"].clone()
        tsum1[0], tsum2[0] = 10.0, 10.0
        tsum1[1], tsum2[1] = 300.0, 300.0
        crop_model_params_provider.set_override("TSUM1", tsum1, check=False)
        crop_model_params_provider.set_override("TSUM2", tsum2, check=False)

        engine = EngineTestHelper(config=phenology_config_with_activity)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        engine.run_till_terminate()
        results = engine.get_output()

        stage = torch.stack([r["STAGE"] for r in results]).cpu()
        is_active = torch.stack([r["IS_ACTIVE"] for r in results]).cpu()
        dvs = torch.stack([r["DVS"] for r in results]).cpu()

        # identify first day of maturity of element zero
        mature_days = (stage[:, 0] == 3).nonzero(as_tuple=True)[0]
        assert mature_days.numel() > 0
        first_mature_day = mature_days[0].item()

        # element 1 has not matured yet at that point
        assert stage[first_mature_day, 1] < 3

        # element 0 stays active for the remaining part of the simulation, and keeps reporting
        # its frozen final DVS (2.0) from the maturity day onwards
        assert torch.all(is_active[first_mature_day:, 0])
        assert torch.all(dvs[first_mature_day:, 0] == 2)

    def test_phenology_no_early_termination_for_pending_element(self, device):
        n = 2
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            day0,
        ) = _prepare_batch_phenology_input(n, tsum1=10.0, tsum2=10.0)

        # element 1 starts 30 days after element 0
        start_dates = torch.tensor([day0, day0 + 30], dtype=torch.int64, device=device)
        crop_model_params_provider.set_override("CROP_START_DATE", start_dates, check=False)

        engine = EngineTestHelper(config=phenology_config_with_activity)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        engine.run_till_terminate()
        results = engine.get_output()

        stage = torch.stack([r["STAGE"] for r in results]).cpu()

        # element 0 matures quickly, well before element 1's own start day
        mature_day = (stage[:, 0] == 3).nonzero(as_tuple=True)[0][0].item()
        assert stage[mature_day, 1] == -1  # element 1 still inactive

        # the run only terminates once every started element has matured too
        assert stage[-2, 1] == 2  # on the second-last day, element 1 is not yet mature
        assert stage[-1, 0] == 3  # on the last day, both elements are mature
        assert stage[-1, 1] == 3


@pytest.mark.usefixtures("fast_mode")
class TestDiffPhenologyDynamicsGradients:
    """Parametrized tests for gradient calculations in phenology dynamics."""

    # Check if they contribute to gradients of outputs
    param_names = [
        "TSUMEM",
        "TBASEM",
        "TEFFMX",
        "TSUM1",
        "TSUM2",
        "DLO",
        "DLC",
        "DVSEND",
        "DTSMTB",
    ]
    output_names = ["DVS", "TSUM", "TSUME"]

    param_configs = {
        "single": {
            "TSUMEM": (50.0, torch.float64),
            "TBASEM": (0.0, torch.float64),
            "TEFFMX": (35.0, torch.float64),
            "TSUM1": (500.0, torch.float64),
            "TSUM2": (600.0, torch.float64),
            "DLO": (0.5, torch.float64),
            "DLC": (0.5, torch.float64),
            "DVSEND": (2.0, torch.float64),
            "DTSMTB": ([0.0, 0.0, 35.0, 35.0, 45.0, 35.0], torch.float64),
            "VERNSAT": (15.0, torch.float64),
            "VERNBASE": (5.0, torch.float64),
            "VERNDVS": (0.5, torch.float64),
        },
        "tensor": {
            "TSUMEM": ([45.0, 50.0, 55.0], torch.float64),
            "TBASEM": ([-2.0, 0.0, 2.0], torch.float64),
            "TEFFMX": ([32.0, 35.0, 38.0], torch.float64),
            "TSUM1": ([450.0, 500.0, 550.0], torch.float64),
            "TSUM2": ([550.0, 600.0, 650.0], torch.float64),
            "DLO": ([0.4, 0.5, 0.6], torch.float64),
            "DLC": ([0.4, 0.5, 0.6], torch.float64),
            "DVSEND": ([1.9, 2.0, 2.1], torch.float64),
            "DTSMTB": (
                [
                    [0, 0, 15, 8, 30, 18],
                    [0, 0, 5, 9, 10, 19],
                    [0, 0, 25, 1, 30, 20],
                ],
                torch.float64,
            ),
            "VERNSAT": ([14.0, 15.0, 16.0], torch.float64),
            "VERNBASE": ([4.0, 5.0, 6.0], torch.float64),
            "VERNDVS": ([0.4, 0.5, 0.6], torch.float64),
        },
    }
    gradient_mapping = {
        "TSUMEM": ["DVS"],
        "TBASEM": ["DVS", "TSUME"],
        "TEFFMX": ["DVS", "TSUME"],
        "TSUM1": ["DVS"],
        "TSUM2": ["DVS"],
        "DLO": ["DVS"],
        "DLC": ["DVS"],
        "DVSI": ["DVS", "TSUM"],
        "DVSEND": ["DVS"],
        "DTSMTB": ["DVS", "TSUM"],
        "VERNSAT": ["DVS", "TSUM"],
        "VERNBASE": ["DVS", "TSUM"],
        "VERNDVS": ["DVS", "TSUM"],
    }

    gradient_params = []
    no_gradient_params = []
    for pname in param_names:
        for oname in output_names:
            if oname in gradient_mapping.get(pname, []):
                gradient_params.append((pname, oname))
            else:
                no_gradient_params.append((pname, oname))

    @pytest.mark.parametrize("param_name,output_name", no_gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_no_gradients(self, param_name, output_name, config_type, device):
        model = get_test_diff_phenology_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()
        if not loss.requires_grad:
            return
        grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all((grads == 0) | torch.isnan(grads)), (
                f"Gradient for {param_name} w.r.t. {output_name} should be zero or NaN"
            )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type, device):
        model = get_test_diff_phenology_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None
        param.grad = None
        loss.backward()
        grad_backward = param.grad
        assert grad_backward is not None
        assert torch.allclose(grad_backward, grads)

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type, device):
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device=device))
        numerical_grad = calculate_numerical_grad(
            lambda: get_test_diff_phenology_model(),
            param_name,
            param.data,
            output_name,
        )
        model = get_test_diff_phenology_model()
        output = model({param_name: param})
        loss = output[output_name].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        # here tol is relaxed due to approximations
        torch.testing.assert_close(
            numerical_grad,
            grads,
            rtol=1e-2,
            atol=1e-2,
        )
        if torch.all(grads == 0):
            warnings.warn(
                (
                    f"Gradient for par '{param_name}' wrt out '{output_name}' is zero: "
                    f"{grads.data.detach().cpu().numpy()}"
                ),
                UserWarning,
            )

    def test_gradients_mixed_active_inactive_batch(self, device):
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            day0,
        ) = _prepare_batch_phenology_input(2, tsum1=50.0, tsum2=50.0)

        start_dates = torch.tensor([-1, day0 + 5], dtype=torch.int64, device=device)
        crop_model_params_provider.set_override("CROP_START_DATE", start_dates, check=False)
        tsum1_value = crop_model_params_provider["TSUM1"].clone()

        def model(params_dict):
            for name, value in params_dict.items():
                crop_model_params_provider.set_override(name, value, check=False)
            engine = EngineTestHelper(config=phenology_config_with_activity)
            engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
            engine.run_till_terminate()
            results = engine.get_output()
            is_active = torch.stack([r["IS_ACTIVE"] for r in results])
            dvs = torch.stack([r["DVS"] for r in results])
            # mask the inactive elements with a plain zero constant
            return {"DVS": torch.where(is_active, dvs, torch.zeros_like(dvs))}

        numerical_grad = calculate_numerical_grad(lambda: model, "TSUM1", tsum1_value, "DVS")

        param = torch.nn.Parameter(tsum1_value.clone())
        grad = torch.autograd.grad(model({"TSUM1": param})["DVS"].sum(), param)[0]
        assert torch.all(torch.isfinite(grad))

        torch.testing.assert_close(grad, numerical_grad, rtol=1e-2, atol=1e-2)
