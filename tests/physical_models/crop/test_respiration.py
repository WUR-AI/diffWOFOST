import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.respiration import WOFOST_Maintenance_Respiration
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

respiration_config = Configuration(
    CROP=WOFOST_Maintenance_Respiration,
    OUTPUT_VARS=["PMRES"],
)


def get_test_diff_respiration_model():
    test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
    (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    ) = prepare_engine_input(test_data, crop_model_params)
    return DiffRespiration(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        respiration_config,
        copy.deepcopy(external_states),
    )


class DiffRespiration(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config,
        external_states,
    ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config = config
        self.external_states = external_states

    def forward(self, params_dict):
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = EngineTestHelper(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.config,
            self.external_states,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return {"PMRES": torch.stack([item["PMRES"] for item in results])}


class TestRespiration:
    respiration_data_urls = [
        f"{phy_data_folder}/test_respiration_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", respiration_data_urls)
    def test_respiration_with_testengine(self, test_data_url, device):
        """EngineTestHelper (not Engine) allows forcing `external_states` from YAML."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
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
            respiration_config,
            external_states,
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

    @pytest.mark.parametrize("param", ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB", "TEMP"])
    def test_respiration_with_one_parameter_vector(self, param, device):
        test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        if param == "TEMP":
            for (_, _), wdc in weather_data_provider.store.items():
                wdc.TEMP = torch.ones(10, dtype=torch.float64) * wdc.TEMP
            with pytest.raises(ValueError):
                engine = EngineTestHelper(
                    crop_model_params_provider,
                    weather_data_provider,
                    agro_management_inputs,
                    respiration_config,
                    external_states,
                )
                engine.run_till_terminate()
                _ = engine.get_output()
            return

        if param == "RFSETB":
            repeated = crop_model_params_provider[param].repeat(10, 1)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
        crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            respiration_config,
            external_states,
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

    @pytest.mark.parametrize(
        "param,delta",
        [
            ("Q10", 0.2),
            ("RMR", 0.002),
            ("RML", 0.002),
            ("RMS", 0.002),
            ("RMO", 0.002),
        ],
    )
    def test_respiration_with_different_parameter_values(self, param, delta, device):
        test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        test_value = crop_model_params_provider[param]
        param_vec = torch.tensor([test_value - delta, test_value + delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            respiration_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                abs(reference[var] - model_cpu[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_respiration_with_multiple_parameter_vectors(self, device):
        test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        for param in ("Q10", "RMR", "RML", "RMS", "RMO"):
            repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)
        crop_model_params_provider.set_override(
            "RFSETB", crop_model_params_provider["RFSETB"].repeat(10, 1), check=False
        )

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            respiration_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]
        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_respiration_with_multiple_parameter_arrays(self, device):
        test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        for param in ("Q10", "RMR", "RML", "RMS", "RMO"):
            repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)
        crop_model_params_provider.set_override(
            "RFSETB", crop_model_params_provider["RFSETB"].repeat(30, 5, 1), check=False
        )

        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones((30, 5), dtype=torch.float64) * wdc.TEMP

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            respiration_config,
            external_states,
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

    def test_respiration_with_incompatible_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        crop_model_params_provider.set_override(
            "RMR", crop_model_params_provider["RMR"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "RML", crop_model_params_provider["RML"].repeat(5), check=False
        )

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                respiration_config,
                external_states,
            )

    def test_respiration_with_incompatible_weather_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_respiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        crop_model_params_provider.set_override(
            "RMR", crop_model_params_provider["RMR"].repeat(10), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones(5, dtype=torch.float64) * wdc.TEMP

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                respiration_config,
                external_states,
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_respiration(self, test_data_url):
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["Q10", "RMR", "RML", "RMS", "RMO", "RFSETB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.MaintenanceRespiration", WOFOST_Maintenance_Respiration):
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


class TestDiffRespirationGradients:
    """Parametrized tests for gradient calculations in maintenance respiration."""

    param_configs = {
        "single": {
            "Q10": (2.0, torch.float64),
            "RMR": (0.015, torch.float64),
            "RML": (0.03, torch.float64),
            "RMS": (0.02, torch.float64),
            "RMO": (0.01, torch.float64),
        },
        "tensor": {
            "Q10": ([1.5, 2.0, 2.5], torch.float64),
            "RMR": ([0.01, 0.015, 0.02], torch.float64),
            "RML": ([0.02, 0.03, 0.04], torch.float64),
            "RMS": ([0.01, 0.02, 0.03], torch.float64),
            "RMO": ([0.005, 0.01, 0.02], torch.float64),
        },
    }

    @pytest.mark.parametrize("param_name", ["Q10", "RMR", "RML", "RMS", "RMO"])
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, config_type, device):
        model = get_test_diff_respiration_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output["PMRES"].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None, f"Gradients for {param_name} should not be None"

        param.grad = None
        loss.backward()
        grad_backward = param.grad
        assert grad_backward is not None, f"Backward gradients for {param_name} should not be None"
        assert torch.all(grad_backward == grads), (
            f"Forward and backward gradients for {param_name} should match"
        )

    @pytest.mark.parametrize("param_name", ["Q10", "RMR", "RML", "RMS", "RMO"])
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, config_type, device):
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))

        numerical_grad = calculate_numerical_grad(
            lambda: get_test_diff_respiration_model(),
            param_name,
            param,
            "PMRES",
        )

        model = get_test_diff_respiration_model()
        output = model({param_name: param})
        loss = output["PMRES"].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        torch.testing.assert_close(
            numerical_grad.detach().cpu(),
            grads.detach().cpu(),
            rtol=1e-3,
            atol=1e-3,
        )

        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for parameter '{param_name}' with"
                + f"respect to output 'PMRES' is zero: {grads.data}",
                UserWarning,
            )
