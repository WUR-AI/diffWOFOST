import copy
import datetime
import warnings
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.evapotranspiration import Evapotranspiration
from diffwofost.physical_models.crop.evapotranspiration import EvapotranspirationCO2
from diffwofost.physical_models.crop.evapotranspiration import EvapotranspirationCO2Layered
from diffwofost.physical_models.crop.evapotranspiration import EvapotranspirationWrapper
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

evapotranspiration_config = Configuration(
    CROP=EvapotranspirationWrapper,
    OUTPUT_VARS=["EVSMX", "EVWMX", "TRAMX", "TRA"],
)


def get_test_diff_evapotranspiration_model(device: str = "cpu"):
    test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = [
        "CFET",
        "DEPNR",
        "KDIFTB",
        "IAIRDU",
        "IOX",
        "CRAIRC",
        "SM0",
        "SMW",
        "SMFCF",
    ]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False, device=device)
    )
    return DiffEvapotranspiration(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        evapotranspiration_config,
        copy.deepcopy(external_states),
        device=device,
    )


class DiffEvapotranspiration(torch.nn.Module):
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

    def forward(self, params_dict: dict[str, torch.Tensor]):
        for name, value in params_dict.items():
            if isinstance(value, torch.Tensor) and value.device.type != self.device:
                value = value.to(self.device)
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
        return {
            var: torch.stack([item[var] for item in results])
            for var in ["EVSMX", "EVWMX", "TRAMX", "TRA"]
        }


class TestEvapotranspiration:
    transpiration_data_urls = [
        f"{phy_data_folder}/test_transpiration_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    @pytest.mark.parametrize("test_data_url", transpiration_data_urls)
    def test_evapotranspiration_with_testengine(self, test_data_url, device):
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(
            test_data, crop_model_params, meteo_range_checks=False, device=device
        )

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            evapotranspiration_config,
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

    @pytest.mark.parametrize(
        "param",
        [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
            "ET0",
        ],
    )
    def test_evapotranspiration_with_one_parameter_vector(self, param, device):
        test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(
            test_data, crop_model_params, meteo_range_checks=False, device=device
        )

        if param == "ET0":
            for (_, _), wdc in weather_data_provider.store.items():
                wdc.ET0 = torch.ones(10, dtype=torch.float64, device=wdc.ET0.device) * wdc.ET0
            with pytest.raises(ValueError):
                EngineTestHelper(
                    crop_model_params_provider,
                    weather_data_provider,
                    agro_management_inputs,
                    evapotranspiration_config,
                    external_states,
                    device=device,
                )
            return

        if param == "KDIFTB":
            repeated = crop_model_params_provider[param].repeat(10, 1)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
        crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            evapotranspiration_config,
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
            ("CFET", 0.1),
            ("DEPNR", 1.0),
            ("KDIFTB", 0.05),
            ("SMW", 0.01),
            ("SMFCF", 0.01),
            ("SM0", 0.01),
        ],
    )
    def test_evapotranspiration_with_different_parameter_values(self, param, delta, device):
        test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(
            test_data, crop_model_params, meteo_range_checks=False, device=device
        )

        test_value = crop_model_params_provider[param]
        if param == "KDIFTB":
            non_zeros_mask = test_value != 0
            param_vec = torch.stack([test_value + non_zeros_mask * delta, test_value])
        else:
            param_vec = torch.tensor(
                [test_value - delta, test_value + delta, test_value], device=device
            )
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            evapotranspiration_config,
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

    def test_evapotranspiration_with_multiple_parameter_vectors(self, device):
        test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(
            test_data, crop_model_params, meteo_range_checks=False, device=device
        )

        for param in crop_model_params:
            if param == "KDIFTB":
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            evapotranspiration_config,
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
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_evapotranspiration_with_multiple_parameter_arrays(self, device):
        test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(
            test_data, crop_model_params, meteo_range_checks=False, device=device
        )

        # Use an arbitrary batched shape and keep weather vars consistent.
        batch_shape = (30, 5)
        for param in ("CFET", "DEPNR", "KDIFTB"):
            if param == "KDIFTB":
                repeated = crop_model_params_provider[param].repeat(*batch_shape, 1)
            else:
                repeated = crop_model_params_provider[param].broadcast_to(batch_shape)
            crop_model_params_provider.set_override(param, repeated, check=False)

        for (_, _), wdc in weather_data_provider.store.items():
            wdc.ET0 = torch.ones(batch_shape, dtype=torch.float64, device=wdc.ET0.device) * wdc.ET0
            wdc.E0 = torch.ones(batch_shape, dtype=torch.float64, device=wdc.E0.device) * wdc.E0
            wdc.ES0 = torch.ones(batch_shape, dtype=torch.float64, device=wdc.ES0.device) * wdc.ES0

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            evapotranspiration_config,
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
            assert all(model[var].shape == batch_shape for var in expected_precision.keys())

    def test_evapotranspiration_with_incompatible_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        crop_model_params_provider.set_override(
            "CFET", crop_model_params_provider["CFET"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "DEPNR", crop_model_params_provider["DEPNR"].repeat(5), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                evapotranspiration_config,
                external_states,
                device="cpu",
            )

    def test_evapotranspiration_with_incompatible_weather_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_transpiration_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        crop_model_params_provider.set_override(
            "CFET", crop_model_params_provider["CFET"].repeat(10), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.ET0 = torch.ones(5, dtype=torch.float64, device=wdc.ET0.device) * wdc.ET0

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                evapotranspiration_config,
                external_states,
                device="cpu",
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls[:1])
    def test_wofost_pp_with_evapotranspiration(self, test_data_url):
        test_data = get_test_data(test_data_url)
        crop_model_params = [
            "CFET",
            "DEPNR",
            "KDIFTB",
            "IAIRDU",
            "IOX",
            "CRAIRC",
            "SM0",
            "SMW",
            "SMFCF",
        ]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)
        )

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Evapotranspiration", EvapotranspirationWrapper):
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


def _minimal_parvalues(device: str, *, include_co2: bool = False, include_layers: bool = False):
    dtype = torch.float64
    pars: dict[str, object] = {
        "CFET": torch.tensor(1.0, dtype=dtype, device=device),
        "DEPNR": torch.tensor(2.0, dtype=dtype, device=device),
        "KDIFTB": torch.tensor([0.0, 0.69, 2.0, 0.69], dtype=dtype, device=device),
        "IAIRDU": torch.tensor(0.0, dtype=dtype, device=device),
        "IOX": torch.tensor(0.0, dtype=dtype, device=device),
        "CRAIRC": torch.tensor(0.06, dtype=dtype, device=device),
        "SM0": torch.tensor(0.40, dtype=dtype, device=device),
        "SMW": torch.tensor(0.15, dtype=dtype, device=device),
        "SMFCF": torch.tensor(0.29, dtype=dtype, device=device),
    }

    if include_co2:
        pars.update(
            {
                "CO2": torch.tensor(700.0, dtype=dtype, device=device),
                "CO2TRATB": torch.tensor([0.0, 1.0, 1000.0, 0.5], dtype=dtype, device=device),
            }
        )

    if include_layers:
        soil_profile = [
            SimpleNamespace(SMW=0.15, SMFCF=0.29, SM0=0.40, CRAIRC=0.06, Thickness=10.0),
            SimpleNamespace(SMW=0.16, SMFCF=0.30, SM0=0.41, CRAIRC=0.06, Thickness=20.0),
        ]
        pars["soil_profile"] = soil_profile

    return ParameterProvider(cropdata=pars)


class TestEvapotranspirationVariants:
    def test_wrapper_selects_base(self, device):
        parvalues = _minimal_parvalues(device)
        kiosk = VariableKiosk()
        wrapper = EvapotranspirationWrapper(datetime.date(2000, 1, 1), kiosk, parvalues)
        assert isinstance(wrapper.etmodule, Evapotranspiration)

    def test_wrapper_selects_co2(self, device):
        parvalues = _minimal_parvalues(device, include_co2=True)
        kiosk = VariableKiosk()
        wrapper = EvapotranspirationWrapper(datetime.date(2000, 1, 1), kiosk, parvalues)
        assert isinstance(wrapper.etmodule, EvapotranspirationCO2)

    def test_wrapper_selects_layered(self, device):
        parvalues = _minimal_parvalues(device, include_co2=True, include_layers=True)
        kiosk = VariableKiosk()
        wrapper = EvapotranspirationWrapper(datetime.date(2000, 1, 1), kiosk, parvalues)
        assert isinstance(wrapper.etmodule, EvapotranspirationCO2Layered)

    def test_co2_reduces_tramx(self, device):
        def _kiosk_with_states():
            kiosk = VariableKiosk()
            oid = 0
            for name in ("DVS", "LAI", "SM"):
                kiosk.register_variable(oid, name, type="S", publish=True)
            kiosk.set_variable(oid, "DVS", torch.tensor(1.0, dtype=torch.float64, device=device))
            kiosk.set_variable(oid, "LAI", torch.tensor(3.0, dtype=torch.float64, device=device))
            kiosk.set_variable(oid, "SM", torch.tensor(0.25, dtype=torch.float64, device=device))
            return kiosk

        drv = SimpleNamespace(
            ET0=torch.tensor(0.5, dtype=torch.float64, device=device),
            E0=torch.tensor(0.6, dtype=torch.float64, device=device),
            ES0=torch.tensor(0.55, dtype=torch.float64, device=device),
            CO2=torch.tensor(700.0, dtype=torch.float64, device=device),
        )

        p_base = _minimal_parvalues(device)
        kiosk_base = _kiosk_with_states()
        base = Evapotranspiration(datetime.date(2000, 1, 1), kiosk_base, p_base)
        base.calc_rates(datetime.date(2000, 1, 2), drv)
        tramx_base = base.rates.TRAMX

        p_co2 = _minimal_parvalues(device, include_co2=True)
        kiosk_co2 = _kiosk_with_states()
        co2 = EvapotranspirationCO2(datetime.date(2000, 1, 1), kiosk_co2, p_co2)
        co2.calc_rates(datetime.date(2000, 1, 2), drv)
        tramx_co2 = co2.rates.TRAMX

        assert torch.all(tramx_co2 <= tramx_base)


class TestDiffEvapotranspirationGradients:
    param_names = ["CFET", "DEPNR", "KDIFTB"]
    output_names = ["EVWMX", "EVSMX", "TRAMX", "TRA"]

    param_configs = {
        "single": {
            "CFET": (1.0, torch.float64),
            "DEPNR": (2.0, torch.float64),
            "KDIFTB": ([[0.0, 0.69, 2.0, 0.69]], torch.float64),
        },
        "tensor": {
            "CFET": ([0.8, 1.0, 1.2], torch.float64),
            "DEPNR": ([1.0, 2.0, 3.0], torch.float64),
            "KDIFTB": (
                [[0.0, 0.60, 2.0, 0.60], [0.0, 0.69, 2.0, 0.69], [0.0, 0.78, 2.0, 0.78]],
                torch.float64,
            ),
        },
    }

    gradient_mapping = {
        "CFET": ["TRAMX", "TRA"],
        "DEPNR": ["TRA"],
        "KDIFTB": ["EVWMX", "EVSMX", "TRAMX", "TRA"],
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
        model = get_test_diff_evapotranspiration_model(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        if not loss.requires_grad:
            grads = None
        else:
            grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all((grads == 0) | torch.isnan(grads)), (
                f"Gradient for {param_name} w.r.t. {output_name} should be zero or NaN"
            )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type, device):
        model = get_test_diff_evapotranspiration_model(device=device)
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
        assert torch.all(grad_backward == grads), "Forward and backward gradients should match"

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type, device):
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device=device))

        numerical_grad = calculate_numerical_grad(
            lambda: get_test_diff_evapotranspiration_model(device=device),
            param_name,
            param,
            output_name,
        )

        model = get_test_diff_evapotranspiration_model(device=device)
        output = model({param_name: param})
        loss = output[output_name].sum()
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        torch.testing.assert_close(
            numerical_grad.detach().cpu(),
            grads.detach().cpu(),
            rtol=1e-3,
            atol=1e-3,
        )

        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for parameter '{param_name}'"
                + f" w.r.t '{output_name}' is zero: {grads.data}",
                UserWarning,
            )
