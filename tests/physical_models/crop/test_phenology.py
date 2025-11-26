import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from numpy.testing import assert_array_almost_equal
from pcse.engine import Engine
from pcse.models import Wofost72_PP
from diffwofost.physical_models.crop.phenology import DVS_Phenology
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

# Ignore deprecation warnings from pcse.base.simulationobject
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pcse.base.simulationobject")


def assert_reference_match(reference, model, expected_precision):
    assert reference["DAY"] == model["day"]
    for var, precision in expected_precision.items():
        if var == "VERNFAC" or var == "VERNR":
            # [!] These are not 'State variables' and are not stored in model output
            continue
        ref_val = reference[var]
        model_val = model[var]
        if ref_val is None or model_val is None:
            assert ref_val is None and model_val is None
            continue
        if torch.is_tensor(model_val):
            assert torch.all(torch.abs(ref_val - model_val) < precision)
        else:
            assert abs(ref_val - model_val) < precision


def get_test_diff_phenology_model():
    test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    # Phenology-related crop model parameters
    crop_model_params = [
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
    ]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params)
    )
    config_path = str(phy_data_folder / "WOFOST_Phenology.conf")
    return DiffPhenologyDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        config_path,
        copy.deepcopy(external_states),
    )


class DiffPhenologyDynamics(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config_path,
        external_states,
    ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config_path = config_path
        self.external_states = external_states

    def forward(self, params_dict):
        # pass new value of parameters to the model
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = EngineTestHelper(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.config_path,
            self.external_states,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        # Collect phenology outputs analogous to leaf dynamics test
        return {var: torch.stack([item[var] for item in results]) for var in ["DVS", "TSUM"]}


class TestPhenologyDynamics:
    phenology_data_urls = [
        f"{phy_data_folder}/test_phenology_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # assume 44 test files
        # for i in range(17, 18)  # assume 44 test files
    ]
    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    @pytest.mark.parametrize("test_data_url", phenology_data_urls)
    def test_phenology_with_testengine(self, test_data_url):
        """EngineTestHelper because it allows to specify `external_states`."""
        test_data = get_test_data(test_data_url)
        crop_model_params = [
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
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)

    def test_phenology_with_engine(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
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
        ]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        Engine(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
        )

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
    def test_phenology_with_one_parameter_vector(self, param):
        # pick a test case with vernalisation to have all the parameters
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
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
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        if param == "TEMP":
            for (_, _), wdc in weather_data_provider.store.items():
                wdc.TEMP = torch.ones(10, dtype=torch.float64) * wdc.TEMP
        elif param == "DTSMTB":
            repeated = crop_model_params_provider[param].repeat(10, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        if param == "TEMP":
            with pytest.raises(ValueError):
                engine = EngineTestHelper(
                    crop_model_params_provider,
                    weather_data_provider,
                    agro_management_inputs,
                    config_path,
                    external_states,
                )
                engine.run_till_terminate()
                _ = engine.get_output()
        else:
            engine = EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
                external_states,
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
            # ("TSUMEM", 1.0),
            ("TBASEM", 0.10),
            # ("TEFFMX", 1.0),
            # ("TSUM1", 1.0),
            # ("TSUM2", 1.0),
            # ("IDSL", 1.0),
            # ("DLO", 1.0),
            # ("DLC", 1.0),
            # ("DVSI", 0.1),
            # ("DVSEND", 0.1),
            # ("DTSMTB", 1.0),
            # ("VERNSAT", 1.0),
            # ("VERNBASE", 0.5),
            # ("VERNDVS", 0.1),
        ],
    )
    def test_phenology_with_different_parameter_values(self, param, delta):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_17.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
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
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        test_value = crop_model_params_provider[param]
        if param == "DTSMTB":
            # Clean trailing (0,0) pairs that are left in the test data
            tv = test_value.clone()
            n_pairs = tv.shape[0] // 2
            valid_n = n_pairs
            for i in range(n_pairs - 1, 0, -1):
                if tv[2 * i] == 0 and tv[2 * i + 1] == 0:
                    valid_n = i
                else:
                    break
            tv = tv[: 2 * valid_n]
            # Only modify y-values (odd indices) to maintain x-values ascending order
            param_vec_list = []
            for delta_factor in [-1, 1, 0]:  # subtract, add, original
                modified = tv.clone()
                modified[1::2] = modified[1::2] + delta_factor * delta
                param_vec_list.append(modified)
            param_vec = torch.stack(param_vec_list)
        else:
            param_vec = torch.tensor([test_value - delta, test_value + delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            # keep original special case using last element
            for var, precision in expected_precision.items():
                if var == "VERNFAC" or var == "VERNR":
                    # [!] These are not 'State variables' and are not stored in model output
                    continue
                ref_val = reference[var]
                model_val = model[var]
                if ref_val is None or model_val is None:
                    assert ref_val is None and model_val is None
                    continue
                # Use last element for comparison with vector parameters
                print(f"\nThis is day {reference['DAY']} and all the model data are {model_val}")
                print(f"Checking param {param}, var {var}, ref {ref_val}, model {model_val[-1]}")
                assert abs(ref_val - model_val[-1]) < precision

    def test_phenology_with_multiple_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "DVSEND", "DTSMTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        for param in crop_model_params:
            if param == "DTSMTB":
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)

    def test_phenology_with_multiple_parameter_arrays(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "DVSEND", "DTSMTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        for param in ("TSUM1", "TSUM2", "TSUMEM", "TBASEM", "TEFFMX", "DVSEND", "DTSMTB"):
            if param == "DTSMTB":
                repeated = crop_model_params_provider[param].repeat(30, 5, 1)
            else:
                repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones((30, 5), dtype=torch.float64) * wdc.TEMP

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model, expected_precision)
            assert all(model[var].shape == (30, 5) for var in expected_precision.keys())

    def test_phenology_with_incompatible_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "DVSEND", "DTSMTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        crop_model_params_provider.set_override(
            "TSUM1", crop_model_params_provider["TSUM1"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "TSUM2", crop_model_params_provider["TSUM2"].repeat(5), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
                external_states,
            )

    def test_phenology_with_incompatible_weather_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "DVSEND", "DTSMTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        crop_model_params_provider.set_override(
            "TSUM1", crop_model_params_provider["TSUM1"].repeat(10), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones(5, dtype=torch.float64) * wdc.TEMP

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
                external_states,
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_phenology(self, test_data_url):
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "DVSEND", "DTSMTB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Phenology", DVS_Phenology):
            model = Wofost72_PP(
                crop_model_params_provider, weather_data_provider, agro_management_inputs
            )
            model.run_till_terminate()
            actual_results = model.get_output()

            assert len(actual_results) == len(expected_results)
            for reference, model_day in zip(expected_results, actual_results, strict=False):
                assert_reference_match(reference, model_day, expected_precision)

    @pytest.mark.parametrize("test_data_url", phenology_data_urls)
    def test_phenology_with_sigmoid_approx(self, test_data_url):
        """Test if calculation with parameter gradients matches expected phenology output."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "DVSEND", "DTSMTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        crop_model_params_provider["TSUM1"].requires_grad = True
        config_path = str(phy_data_folder / "WOFOST_Phenology.conf")

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model_day in zip(expected_results, actual_results, strict=False):
            assert_reference_match(reference, model_day, expected_precision)


class TestDiffPhenologyDynamicsGradients:
    """Parametrized tests for gradient calculations in phenology dynamics."""

    param_names = ["TSUMEM", "TSUM1", "TSUM2", "TBASEM", "TEFFMX", "DVSEND", "DTSMTB"]
    output_names = ["DVS", "TSUM"]

    param_configs = {
        "single": {
            "TSUMEM": (50.0, torch.float64),
            "TSUM1": (500.0, torch.float64),
            "TSUM2": (600.0, torch.float64),
            "TBASEM": (0.0, torch.float64),
            "TEFFMX": (35.0, torch.float64),
            "DVSEND": (2.0, torch.float64),
            "DTSMTB": ([[0, 0], [10, 5], [20, 15], [30, 20]], torch.float64),
        },
        "tensor": {
            "TSUMEM": ([45.0, 50.0, 55.0], torch.float64),
            "TSUM1": ([450.0, 500.0, 550.0], torch.float64),
            "TSUM2": ([550.0, 600.0, 650.0], torch.float64),
            "TBASEM": ([-2.0, 0.0, 2.0], torch.float64),
            "TEFFMX": ([32.0, 35.0, 38.0], torch.float64),
            "DVSEND": ([1.9, 2.0, 2.1], torch.float64),
            "DTSMTB": (
                [
                    [0, 0, 15, 8, 30, 18],
                    [0, 0, 15, 9, 30, 19],
                    [0, 0, 15, 10, 30, 20],
                ],
                torch.float64,
            ),
        },
    }

    gradient_mapping = {
        "TSUMEM": ["DVS", "TSUM"],
        "TSUM1": ["DVS", "TSUM"],
        "TSUM2": ["DVS", "TSUM"],
        "TBASEM": ["DVS", "TSUM"],
        "TEFFMX": ["DVS", "TSUM"],
        "DTSMTB": ["DVS", "TSUM"],
        "DVSEND": [],  # acts as cap; treat as no gradient target
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
    def test_no_gradients(self, param_name, output_name, config_type):
        model = get_test_diff_phenology_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output[output_name].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all((grads == 0) | torch.isnan(grads)), (
                f"Gradient for {param_name} w.r.t. {output_name} should be zero or NaN"
            )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type):
        model = get_test_diff_phenology_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output[output_name].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None
        param.grad = None
        loss.backward()
        grad_backward = param.grad
        assert grad_backward is not None
        assert torch.all(grad_backward == grads)

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type):
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        numerical_grad = calculate_numerical_grad(
            get_test_diff_phenology_model, param_name, param.data, output_name
        )
        model = get_test_diff_phenology_model()
        output = model({param_name: param})
        loss = output[output_name].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert_array_almost_equal(numerical_grad, grads.data, decimal=3)
        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for par '{param_name}' wrt out '{output_name}' is zero: {grads.data}",
                UserWarning,
            )
