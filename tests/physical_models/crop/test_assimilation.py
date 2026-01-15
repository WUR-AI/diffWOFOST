import copy
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.assimilation import WOFOST72_Assimilation
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import _afgen_y_mask
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

assimilation_config = Configuration(
    CROP=WOFOST72_Assimilation,
    OUTPUT_VARS=["PGASS"],
)


def get_test_diff_assimilation_model(device: str = "cpu"):
    test_data_url = f"{phy_data_folder}/test_assimilation_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params, device=device)
    )
    return DiffAssimilation(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        assimilation_config,
        copy.deepcopy(external_states),
        device=device,
    )


class DiffAssimilation(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config,
        external_states,
        device: str = "cpu",
    ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config = config
        self.external_states = external_states
        self.device = device

    def forward(self, params_dict):
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = EngineTestHelper(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.config,
            self.external_states,
            device=self.device,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return {"PGASS": torch.stack([item["PGASS"] for item in results])}


class TestAssimilation:
    assimilation_data_urls = [
        f"{phy_data_folder}/test_assimilation_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    @pytest.mark.parametrize("test_data_url", assimilation_data_urls)
    def test_assimilation_with_testengine(self, test_data_url, device):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            assimilation_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            for var in expected_precision.keys():
                assert model[var].device.type == device, f"{var} should be on {device}"
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                abs(reference[var] - model_cpu[var]) < precision
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize("param", ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"])
    def test_assimilation_with_one_parameter_vector(self, param, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        repeated = crop_model_params_provider[param].repeat(10, 1)
        crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            assimilation_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            for var in expected_precision.keys():
                assert model[var].device.type == device, f"{var} should be on {device}"
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                all(abs(reference[var] - model_cpu[var]) < precision)
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize(
        "param,delta",
        [
            ("AMAXTB", 0.1),
            ("EFFTB", 1e-6),
            ("KDIFTB", 0.01),
            ("TMPFTB", 0.01),
            ("TMNFTB", 0.01),
        ],
    )
    def test_assimilation_with_different_parameter_values(self, param, delta, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        test_value = crop_model_params_provider[param]
        ymask = _afgen_y_mask(test_value)
        param_vec = torch.stack([test_value + ymask * delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            assimilation_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            for var in expected_precision.keys():
                assert model[var].device.type == device, f"{var} should be on {device}"
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                abs(reference[var] - model_cpu[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_assimilation_with_multiple_parameter_vectors(self, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        for param in crop_model_params:
            repeated = crop_model_params_provider[param].repeat(10, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            assimilation_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                all(abs(reference[var] - model_cpu[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_assimilation_with_multiple_parameter_arrays(self, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        for param in crop_model_params:
            repeated = crop_model_params_provider[param].repeat(30, 5, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)

        # Make weather drivers match (30, 5) so _get_drv validates/broadcasts.
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.IRRAD = torch.ones((30, 5), dtype=torch.float64) * wdc.IRRAD
            wdc.TEMP = torch.ones((30, 5), dtype=torch.float64) * wdc.TEMP
            wdc.TMIN = torch.ones((30, 5), dtype=torch.float64) * wdc.TMIN

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            assimilation_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                torch.all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )
            assert all(model[var].shape == (30, 5) for var in expected_precision.keys())

    def test_assimilation_with_incompatible_parameter_vectors(self):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        crop_model_params_provider.set_override(
            "AMAXTB", crop_model_params_provider["AMAXTB"].repeat(10, 1), check=False
        )
        crop_model_params_provider.set_override(
            "EFFTB", crop_model_params_provider["EFFTB"].repeat(5, 1), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                assimilation_config,
                external_states,
                device="cpu",
            )

    def test_assimilation_with_incompatible_weather_parameter_vectors(self):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        crop_model_params_provider.set_override(
            "AMAXTB", crop_model_params_provider["AMAXTB"].repeat(10, 1), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones(5, dtype=torch.float64) * wdc.TEMP

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                assimilation_config,
                external_states,
                device="cpu",
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_assimilation(self, test_data_url):
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Assimilation", WOFOST72_Assimilation):
            model = Wofost72_PP(
                crop_model_params_provider, weather_data_provider, agro_management_inputs
            )
            model.run_till_terminate()
            actual_results = model.get_output()

            assert len(actual_results) == len(expected_results)

            for reference, model in zip(expected_results, actual_results, strict=False):
                assert reference["DAY"] == model["day"]
                assert all(
                    abs(reference[var] - model[var]) < precision
                    for var, precision in expected_precision.items()
                )


class TestDiffAssimilationGradients:
    """Parametrized tests for gradient calculations in assimilation."""

    param_names = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
    output_names = ["PGASS"]

    param_configs = {
        "single": {
            "AMAXTB": ([0.0, 30.0, 2.0, 30.0], torch.float64),
            "EFFTB": ([0.0, 0.0005, 40.0, 0.0005], torch.float64),
            "KDIFTB": ([0.0, 0.7, 2.0, 0.7], torch.float64),
            "TMPFTB": ([0.0, 1.0, 40.0, 1.0], torch.float64),
            "TMNFTB": ([-10.0, 0.0, 0.0, 1.0, 10.0, 1.0], torch.float64),
        },
        "tensor": {
            "AMAXTB": (
                [[0.0, 28.0, 2.0, 28.0], [0.0, 30.0, 2.0, 30.0], [0.0, 32.0, 2.0, 32.0]],
                torch.float64,
            ),
            "EFFTB": (
                [
                    [0.0, 0.00045, 40.0, 0.00045],
                    [0.0, 0.00050, 40.0, 0.00050],
                    [0.0, 0.00055, 40.0, 0.00055],
                ],
                torch.float64,
            ),
            "KDIFTB": (
                [[0.0, 0.6, 2.0, 0.6], [0.0, 0.7, 2.0, 0.7], [0.0, 0.8, 2.0, 0.8]],
                torch.float64,
            ),
            "TMPFTB": (
                [[0.0, 0.9, 40.0, 0.9], [0.0, 1.0, 40.0, 1.0], [0.0, 1.1, 40.0, 1.1]],
                torch.float64,
            ),
            "TMNFTB": (
                [
                    [-10.0, 0.0, 0.0, 0.9, 10.0, 0.9],
                    [-10.0, 0.0, 0.0, 1.0, 10.0, 1.0],
                    [-10.0, 0.0, 0.0, 1.1, 10.0, 1.1],
                ],
                torch.float64,
            ),
        },
    }

    gradient_mapping = {
        "AMAXTB": ["PGASS"],
        "EFFTB": ["PGASS"],
        "KDIFTB": ["PGASS"],
        "TMPFTB": ["PGASS"],
        "TMNFTB": ["PGASS"],
    }

    gradient_params = []
    no_gradient_params = []
    for param_name in param_names:
        for output_name in output_names:
            if output_name in gradient_mapping.get(param_name, []):
                gradient_params.append((param_name, output_name))
            else:
                no_gradient_params.append((param_name, output_name))

    @pytest.mark.parametrize("param_name,output_name", no_gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_no_gradients(self, param_name, output_name, config_type, device):
        model = get_test_diff_assimilation_model(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all(grads == 0)

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type, device):
        model = get_test_diff_assimilation_model(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None, f"Gradients for {param_name} should not be None"

        param.grad = None
        loss.backward()
        grad_backward = param.grad

        assert grad_backward is not None, f"Backward gradients for {param_name} should not be None"
        assert torch.all(grad_backward == grads), (
            f"Forward and backward gradients for {param_name} should match"
        )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type, device):
        value, _ = self.param_configs[config_type][param_name]
        param_value = torch.tensor(value, dtype=torch.float64, device=device)

        def get_model_fn():
            return get_test_diff_assimilation_model(device=device)

        grad_num = calculate_numerical_grad(get_model_fn, param_name, param_value, output_name)

        param = torch.nn.Parameter(param_value.clone())
        output = get_model_fn()({param_name: param})
        loss = output[output_name].sum()
        grad_autograd = torch.autograd.grad(loss, param)[0]

        assert torch.allclose(grad_autograd, grad_num, atol=1e-4, rtol=1e-4)
