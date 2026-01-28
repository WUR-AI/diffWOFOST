import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.storage_organ_dynamics import WOFOST_Storage_Organ_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

storage_dynamics_config = Configuration(
    CROP=WOFOST_Storage_Organ_Dynamics,
    OUTPUT_VARS=["PAI", "TWSO", "WSO", "DWSO"],
)

# [!] Notice that the storage organ module does not have dedicated test data.
# This means that we can only test the execution of the module,
# but not the correctness of its results (except when used within Wofost72_PP).


def _prepare_common_storage_inputs(test_data_url, device, meteo_range_checks=True):
    # prepare model input
    test_data = get_test_data(test_data_url)
    crop_model_params = ["TDWI", "SPA"]
    (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    ) = prepare_engine_input(
        test_data, crop_model_params, meteo_range_checks=meteo_range_checks, device=device
    )

    # Patch missing states
    for state in external_states:
        if "FO" not in state:
            state["FO"] = 0.5
        if "FR" not in state:
            state["FR"] = 0.5
        if "ADMI" not in state:
            state["ADMI"] = 100.0
        # DVS is unused in storage organ dynamics but good to have if something changes
        if "DVS" not in state:
            state["DVS"] = 0.0

    # Patch missing parameters
    if "SPA" not in crop_model_params_provider:
        crop_model_params_provider.set_override(
            "SPA",
            torch.tensor(0.01, dtype=torch.float64, device=device),
            check=False,
        )
    if "TDWI" not in crop_model_params_provider:
        crop_model_params_provider.set_override(
            "TDWI", torch.tensor(20.0, dtype=torch.float64, device=device), check=False
        )

    return (
        test_data,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    )


def get_test_diff_storage_model(device: str = "cpu"):
    # [!] The storage organ module does not have dedicated test data.
    # We reuse the partitioning test data as they contain relevant parameters and states.
    test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"

    (
        _,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    ) = _prepare_common_storage_inputs(test_data_url, device=device)

    return DiffStorageDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        storage_dynamics_config,
        copy.deepcopy(external_states),
        device=device,
    )


class DiffStorageDynamics(torch.nn.Module):
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
        # pass new value of parameters to the model
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

        return {
            var: torch.stack([item[var] for item in results])
            for var in ["PAI", "TWSO", "WSO", "DWSO"]
        }


class TestStorageOrganDynamics:
    # [!] The storage module does not have dedicated test data.
    # We reuse the partitioning test data as they contain relevant parameters and states.
    storage_dynamics_data_urls = [
        f"{phy_data_folder}/test_partitioning_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", storage_dynamics_data_urls)
    def test_storage_dynamics_with_testengine(self, test_data_url, device):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_storage_inputs(test_data_url, device=device)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            storage_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    @pytest.mark.parametrize("param", ["TDWI", "SPA", "TEMP"])
    def test_storage_dynamics_with_one_parameter_vector(self, param, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_storage_inputs(test_data_url, device=device, meteo_range_checks=False)

        # Setting a vector (with one value) for the selected parameter
        if param == "TEMP":
            # Vectorize weather variable
            for (_, _), wdc in weather_data_provider.store.items():
                wdc.TEMP = torch.ones(10, dtype=torch.float64) * wdc.TEMP
        else:
            # Broadcast all parameters to match the batch size of 10
            for p_name in ["TDWI", "SPA"]:
                if p_name in crop_model_params_provider:
                    p_val = crop_model_params_provider[p_name]
                    if p_val.dim() == 0:  # scalar
                        crop_model_params_provider.set_override(
                            p_name, p_val.repeat(10), check=False
                        )
                    elif p_val.dim() == 2 and p_val.shape[0] == 1:  # table (1, M) -> (10, M)
                        crop_model_params_provider.set_override(
                            p_name, p_val.repeat(10, 1), check=False
                        )

        if param == "TEMP":
            # Vectorize weather variable
            # We expect the model to handle scalar parameters with vectorized weather
            # via implicit broadcasting or explicit checks passing.
            engine = EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                storage_dynamics_config,
                external_states,
                device=device,
            )
            engine.run_till_terminate()
            actual_results = engine.get_output()
        else:
            engine = EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                storage_dynamics_config,
                external_states,
                device=device,
            )
            engine.run_till_terminate()
            actual_results = engine.get_output()

            # get expected results from YAML test data
            expected_results = test_data["ModelResults"]

            # Assertions on values removed as test data is not appropriate for this module
            assert len(actual_results) == len(expected_results)

    @pytest.mark.parametrize(
        "param,delta",
        [
            ("TDWI", 0.1),
            ("SPA", 0.0001),
        ],
    )
    def test_storage_dynamics_with_different_parameter_values(self, param, delta, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_storage_inputs(test_data_url, device=device)

        # Setting a vector with multiple values for the selected parameter
        test_value = crop_model_params_provider[param]

        param_vec = torch.tensor(
            [test_value - delta, test_value + delta, test_value],
            device=device,
            dtype=torch.float64,
        )
        target_batch_size = 3
        crop_model_params_provider.set_override(param, param_vec, check=False)

        # Broadcast all other params
        for p_name in ["TDWI", "SPA"]:
            if p_name == param:
                continue
            if p_name not in crop_model_params_provider:
                continue

            p_val = crop_model_params_provider[p_name]
            if p_val.dim() == 0:
                crop_model_params_provider.set_override(
                    p_name, p_val.repeat(target_batch_size), check=False
                )

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            storage_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    def test_storage_dynamics_with_multiple_parameter_vectors(self, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_storage_inputs(test_data_url, device=device)

        # Setting a vector (with one value) for the TDWI and SPA parameters
        for param in ("TDWI", "SPA"):
            if param == "SPA" and crop_model_params_provider[param].dim() == 2:
                # In case SPA is treated as table somehow, though here it is scalar
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            storage_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    def test_storage_dynamics_with_multiple_parameter_arrays(self, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_storage_inputs(test_data_url, device=device, meteo_range_checks=False)

        # Setting an array with arbitrary shape (and one value)
        for param in ("TDWI", "SPA"):
            repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones((30, 5), dtype=torch.float64) * wdc.TEMP

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            storage_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    def test_storage_dynamics_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            _,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_storage_inputs(test_data_url, device="cpu")

        # Setting a vector (with one value) for the TDWI and SPA parameters,
        # but with different lengths
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "SPA", crop_model_params_provider["SPA"].repeat(5), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                storage_dynamics_config,
                external_states,
                device="cpu",
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_storage_dynamics(self, test_data_url):
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TDWI", "SPA"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Storage_Organ_Dynamics", WOFOST_Storage_Organ_Dynamics):
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


class TestDiffStorageDynamicsGradients:
    """Parametrized tests for gradient calculations in storage organ dynamics."""

    # Define parameters and outputs
    param_names = ["TDWI", "SPA"]
    output_names = ["PAI", "TWSO", "WSO"]

    # Define parameter configurations (value, dtype)
    param_configs = {
        "single": {
            "TDWI": (0.2, torch.float64),
            "SPA": (0.01, torch.float64),
        },
        "tensor": {
            "TDWI": ([0.1, 0.2, 0.3], torch.float64),
            "SPA": ([0.01, 0.02, 0.03], torch.float64),
        },
    }

    # Define which parameter-output pairs should have gradients
    # Format: {param_name: [list of outputs that should have gradients]}
    gradient_mapping = {
        "TDWI": ["PAI", "TWSO", "WSO", "DWSO"],
        "SPA": ["PAI"],
    }

    # Generate all combinations
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
        """Test cases where parameters should not have gradients for specific outputs."""
        model = get_test_diff_storage_model(device=device)

        if config_type == "tensor":
            for p_name, (p_val, p_dtype) in self.param_configs["tensor"].items():
                if p_name != param_name:
                    model.crop_model_params_provider.set_override(
                        p_name, torch.tensor(p_val, dtype=p_dtype, device=device), check=False
                    )

        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        if not loss.requires_grad:
            return

        try:
            grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        except RuntimeError as e:
            if "does not require grad" in str(e):
                return
            raise e

        if grads is not None:
            assert torch.all((grads == 0) | torch.isnan(grads)), (
                f"Gradient for {param_name} w.r.t. {output_name} should be zero or NaN"
            )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type, device):
        """Test that forward and backward gradients match for parameter-output pairs."""
        model = get_test_diff_storage_model(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))

        overrides = {param_name: param}
        if config_type == "tensor":
            for p_name, (p_val, p_dtype) in self.param_configs["tensor"].items():
                if p_name != param_name:
                    overrides[p_name] = torch.tensor(p_val, dtype=p_dtype, device=device)

        output = model(overrides)
        loss = output[output_name].sum()

        # this is ∂loss/∂param
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        assert grads is not None, f"Gradients for {param_name} should not be None"

        param.grad = None  # clear any existing gradient
        loss.backward()

        # this is ∂loss/∂param calculated using backpropagation
        grad_backward = param.grad

        assert grad_backward is not None, f"Backward gradients for {param_name} should not be None"
        assert torch.all(grad_backward == grads), (
            f"Forward and backward gradients for {param_name} should match"
        )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type, device):
        """Test that analytical gradients match numerical gradients."""
        value, _ = self.param_configs[config_type][param_name]

        # we pass `param` and not `param.data` because we want `requires_grad=True`
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device=device))

        def model_factory():
            m = get_test_diff_storage_model(device=device)
            if config_type == "tensor":
                for p_name, (p_val, p_dtype) in self.param_configs["tensor"].items():
                    if p_name != param_name:
                        m.crop_model_params_provider.set_override(
                            p_name, torch.tensor(p_val, dtype=p_dtype, device=device), check=False
                        )
            return m

        numerical_grad = calculate_numerical_grad(model_factory, param_name, param, output_name)

        model = model_factory()
        output = model({param_name: param})
        loss = output[output_name].sum()

        # this is ∂loss/∂param, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        torch.testing.assert_close(
            numerical_grad.detach().cpu(),
            grads.detach().cpu(),
            rtol=1e-3,
            atol=1e-3,
        )

        # Warn if gradient is zero (but this shouldn't happen for gradient_params)
        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for parameter '{param_name}' with respect to output "
                + f"'{output_name}' is zero: {grads.data}",
                UserWarning,
            )
