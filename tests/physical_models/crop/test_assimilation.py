from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.assimilation import WOFOST72_Assimilation
from diffwofost.physical_models.test import EngineTestHelper
from diffwofost.physical_models.test import calculate_numerical_grad
from diffwofost.physical_models.test import get_test_data
from diffwofost.physical_models.test import prepare_engine_input
from diffwofost.physical_models.utils import _afgen_y_mask
from .. import phy_data_folder

assimilation_config = Configuration(
    CROP=WOFOST72_Assimilation,
    OUTPUT_VARS=["PGASS"],
)


def get_test_diff_assimilation_model():
    test_data_url = f"{phy_data_folder}/test_assimilation_wofost72_05.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params)
    )
    return DiffAssimilation(
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        assimilation_config,
        external_states,
    )


class DiffAssimilation(torch.nn.Module):
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
        self.engine = EngineTestHelper(config=self.config)

    def forward(self, params_dict):
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = self.engine.setup(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.external_states,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return {"PGASS": torch.stack([item["PGASS"] for item in results])}


@pytest.mark.usefixtures("fast_mode")
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

        engine = EngineTestHelper(config=assimilation_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
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

    @pytest.mark.parametrize("param", ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"])
    def test_assimilation_with_one_parameter_vector(self, param, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_05.yaml"
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

        engine = EngineTestHelper(config=assimilation_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
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
        test_data_url = phy_data_folder / "test_assimilation_wofost72_05.yaml"
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

        engine = EngineTestHelper(config=assimilation_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
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
                abs(reference[var] - model_cpu[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_assimilation_with_multiple_parameter_vectors(self, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_05.yaml"
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

        engine = EngineTestHelper(config=assimilation_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
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

    def test_assimilation_with_multiple_parameter_arrays(self, device):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_05.yaml"
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

        engine = EngineTestHelper(config=assimilation_config)
        engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
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

    def test_assimilation_with_incompatible_parameter_vectors(self):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_05.yaml"
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

        engine = EngineTestHelper(config=assimilation_config)
        with pytest.raises(ValueError):
            engine.setup(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                external_states,
            )

    def test_assimilation_with_incompatible_weather_parameter_vectors(self):
        test_data_url = phy_data_folder / "test_assimilation_wofost72_05.yaml"
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

        broadcasted = broadcast(weather_data_provider)

        engine = EngineTestHelper(config=assimilation_config)
        with pytest.raises(ValueError):
            engine.setup(
                crop_model_params_provider,
                broadcasted,
                agro_management_inputs,
                external_states,
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_assimilation(self, test_data_url):
        test_data = get_test_data(test_data_url)
        crop_model_params = ["AMAXTB", "EFFTB", "KDIFTB", "TMPFTB", "TMNFTB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params, return_weather_data_provider=True)
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


@pytest.mark.usefixtures("fast_mode")
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
        model = get_test_diff_assimilation_model()
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
        model = get_test_diff_assimilation_model()
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
            return get_test_diff_assimilation_model()

        grad_num = calculate_numerical_grad(get_model_fn, param_name, param_value, output_name)

        param = torch.nn.Parameter(param_value.clone())
        output = get_model_fn()({param_name: param})
        loss = output[output_name].sum()
        grad_autograd = torch.autograd.grad(loss, param)[0]

        assert torch.allclose(grad_autograd, grad_num, atol=1e-4, rtol=1e-4)


def test_wofost81_assimilation_matches_pcse():
    """WOFOST 8.1 gross assimilation follows PCSE below and above the LAI threshold
    for the SLN profile."""
    import datetime
    from types import SimpleNamespace
    from pcse.base.parameter_providers import ParameterProvider
    from pcse.base.variablekiosk import VariableKiosk
    from pcse.crop.assimilation import WOFOST81_Assimilation as PcseAssimilation
    from diffwofost.physical_models.config import ComputeConfig
    from diffwofost.physical_models.crop.assimilation import WOFOST81_Assimilation

    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    day = datetime.date(2010, 6, 1)
    tables = {
        "AMAX_LNB": 0.5,
        "AMAX_REF": 40.0,
        "AMAX_SLP": 20.0,
        "KN": 0.5,
        "CO2": 360.0,
        "EFFTB": [0.0, 0.45, 40.0, 0.45],
        "KDIFTB": [0.0, 0.6, 2.0, 0.6],
        "TMPFTB": [0.0, 1.0, 40.0, 1.0],
        "TMNFTB": [0.0, 1.0, 40.0, 1.0],
        "CO2AMAXTB": [40.0, 0.0, 360.0, 1.0, 720.0, 1.2],
        "CO2EFFTB": [40.0, 0.0, 360.0, 1.0, 720.0, 1.2],
    }
    weather = {"IRRAD": 15e6, "TEMP": 18.0, "DTEMP": 20.0, "TMIN": 10.0, "LAT": 52.0}
    for lai in (0.005, 2.0):
        states = {"DVS": 1.0, "LAI": lai, "NamountLV": 40.0}
        kiosk = VariableKiosk()
        for name, value in states.items():
            kiosk.register_variable(0, name, type="S", publish=True)
            kiosk.set_variable(0, name, value)
        pcse = PcseAssimilation(day, kiosk, ParameterProvider(cropdata=tables))
        reference = pcse(day, SimpleNamespace(**weather))
        tensor_kiosk = VariableKiosk()
        for name, value in states.items():
            tensor_kiosk.register_variable(0, name, type="S", publish=True)
            tensor_kiosk.set_variable(0, name, torch.tensor(value))
        diff = WOFOST81_Assimilation(day, tensor_kiosk, ParameterProvider(cropdata=tables))
        got = diff(day, weather)
        torch.testing.assert_close(
            got, torch.tensor(reference, dtype=got.dtype), rtol=1e-6, atol=1e-8
        )


def _assimilation_pgass(overrides, lai, n_amount_lv):
    """One day of WOFOST 8.1 assimilation with the given parameter overrides."""
    import datetime
    from pcse.base.parameter_providers import ParameterProvider
    from pcse.base.variablekiosk import VariableKiosk
    from diffwofost.physical_models.crop.assimilation import WOFOST81_Assimilation

    day = datetime.date(2010, 6, 1)
    crop = {
        "AMAX_LNB": 0.5,
        "AMAX_REF": 40.0,
        "AMAX_SLP": 20.0,
        "KN": 0.5,
        "CO2": 360.0,
        "EFFTB": [0.0, 0.45, 40.0, 0.45],
        "KDIFTB": [0.0, 0.6, 2.0, 0.6],
        "TMPFTB": [0.0, 1.0, 40.0, 1.0],
        "TMNFTB": [0.0, 1.0, 40.0, 1.0],
        "CO2AMAXTB": [40.0, 0.0, 360.0, 1.0, 720.0, 1.2],
        "CO2EFFTB": [40.0, 0.0, 360.0, 1.0, 720.0, 1.2],
    }
    provider = ParameterProvider(cropdata=crop)
    for name, value in overrides.items():
        provider.set_override(name, value, check=False)
    kiosk = VariableKiosk()
    states = {"DVS": 1.0, "LAI": lai, "NamountLV": n_amount_lv}
    for name, value in states.items():
        kiosk.register_variable(0, name, type="S", publish=True)
        kiosk.set_variable(0, name, torch.tensor(value, dtype=torch.float64))
    weather = {"IRRAD": 15e6, "TEMP": 18.0, "DTEMP": 20.0, "TMIN": 10.0, "LAT": 52.0}
    return WOFOST81_Assimilation(day, kiosk, provider)(day, weather)


def _assert_scalar_gradient(evaluate, name, value):
    """Central differences agree with autograd for one scalar parameter."""
    from diffwofost.physical_models.config import ComputeConfig

    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
    loss = evaluate({name: param}).sum()
    autograd = torch.autograd.grad(loss, param)[0]
    delta = 1e-6
    with torch.no_grad():
        plus = evaluate({name: param.detach() + delta}).sum()
        minus = evaluate({name: param.detach() - delta}).sum()
    numerical = (plus - minus) / (2 * delta)
    assert torch.isfinite(autograd).all()
    assert autograd.item() != 0
    torch.testing.assert_close(autograd, numerical, rtol=1e-3, atol=1e-3)


def test_amax_ref_gradient_matches_numerical():
    """AMAX_REF changes gross assimilation once the leaf response is at the cap."""
    _assert_scalar_gradient(
        lambda overrides: _assimilation_pgass(overrides, 2.0, 40.0), "AMAX_REF", 40.0
    )


def test_amax_slp_gradient_matches_numerical():
    """AMAX_SLP changes gross assimilation while the leaf response is below the cap."""
    _assert_scalar_gradient(
        lambda overrides: _assimilation_pgass(overrides, 2.0, 2.0), "AMAX_SLP", 20.0
    )
