import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.stem_dynamics import WOFOST_Stem_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

stem_dynamics_config = Configuration(
    CROP=WOFOST_Stem_Dynamics,
    OUTPUT_VARS=["SAI", "TWST"],
)

# [!] Notice that the stem module does not have dedicated test data.
# This means that we can only test the execution of the module,
# but not the correctness of its results (except when used within Wofost72_PP).


def _prepare_common_stem_inputs(test_data_url, device, meteo_range_checks=True):
    # prepare model input
    test_data = get_test_data(test_data_url)
    crop_model_params = ["TDWI", "RDRSTB", "SSATB"]
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
        if "FS" not in state:
            state["FS"] = 0.5
        if "FR" not in state:
            state["FR"] = 0.5
        if "DVS" not in state:
            state["DVS"] = 0.0
        if "ADMI" not in state:
            state["ADMI"] = 100.0

    # Patch missing parameters
    if "RDRSTB" not in crop_model_params_provider:
        crop_model_params_provider.set_override(
            "RDRSTB",
            torch.tensor([[0.0, 0.0, 2.5, 0.0]], dtype=torch.float64, device=device),
            check=False,
        )
    if "SSATB" not in crop_model_params_provider:
        crop_model_params_provider.set_override(
            "SSATB",
            torch.tensor([[0.0, 0.0003, 2.5, 0.0003]], dtype=torch.float64, device=device),
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


def get_test_diff_stem_model(device: str = "cpu"):
    # [!] The stem module does not have dedicated test data.
    # We reuse the partitioning test data as they contain relevant parameters and states.
    test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"

    (
        _,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    ) = _prepare_common_stem_inputs(test_data_url, device=device)

    return DiffStemDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        stem_dynamics_config,
        copy.deepcopy(external_states),
        device=device,
    )


class DiffStemDynamics(torch.nn.Module):
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

        return {var: torch.stack([item[var] for item in results]) for var in ["SAI", "TWST"]}


class TestStemDynamics:
    # [!] The stem module does not have dedicated test data.
    # We reuse the partitioning test data as they contain relevant parameters and states.
    stemdynamics_data_urls = [
        f"{phy_data_folder}/test_partitioning_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", stemdynamics_data_urls)
    def test_stem_dynamics_with_testengine(self, test_data_url, device):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device=device)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            stem_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    @pytest.mark.parametrize("param", ["TDWI", "RDRSTB", "SSATB", "TEMP"])
    def test_stem_dynamics_with_one_parameter_vector(self, param, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device=device, meteo_range_checks=False)

        # Setting a vector (with one value) for the selected parameter
        if param == "TEMP":
            # Vectorize weather variable
            for (_, _), wdc in weather_data_provider.store.items():
                wdc.TEMP = torch.ones(10, dtype=torch.float64) * wdc.TEMP
        else:
            # Broadcast all parameters to match the batch size of 10
            # This ensures compatibility for all parameters including table traits
            for p_name in ["TDWI", "RDRSTB", "SSATB"]:
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
                stem_dynamics_config,
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
                stem_dynamics_config,
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
            ("RDRSTB", 0.001),
            ("SSATB", 0.0001),
        ],
    )
    def test_stem_dynamics_with_different_parameter_values(self, param, delta, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device=device)

        # Setting a vector with multiple values for the selected parameter
        test_value = crop_model_params_provider[param]
        # We set the value for which test data are available as the last element
        if param in {"RDRSTB", "SSATB"}:
            # AfgenTrait parameters need to have shape (N, M)
            non_zeros_mask = test_value != 0
            # Use cat to get (2, 4) instead of stack (2, 1, 4)
            param_vec = torch.cat([test_value + non_zeros_mask * delta, test_value], dim=0)
            target_batch_size = 2
        else:
            param_vec = torch.tensor(
                [test_value - delta, test_value + delta, test_value],
                device=device,
                dtype=torch.float64,
            )
            target_batch_size = 3
        crop_model_params_provider.set_override(param, param_vec, check=False)

        # Broadcast all other params
        for p_name in ["TDWI", "RDRSTB", "SSATB"]:
            if p_name == param:
                continue
            if p_name not in crop_model_params_provider:
                continue

            p_val = crop_model_params_provider[p_name]
            if p_val.dim() == 0:
                crop_model_params_provider.set_override(
                    p_name, p_val.repeat(target_batch_size), check=False
                )
            elif p_val.dim() == 2 and p_val.shape[0] == 1:
                crop_model_params_provider.set_override(
                    p_name, p_val.repeat(target_batch_size, 1), check=False
                )

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            stem_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    def test_stem_dynamics_with_multiple_parameter_vectors(self, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device=device)

        # Setting a vector (with one value) for the TDWI, RDRSTB and SSATB parameters
        for param in ("TDWI", "RDRSTB", "SSATB"):
            if param in ("RDRSTB", "SSATB"):
                # AfgenTrait parameters need to have shape (N, M)
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            stem_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    def test_stem_dynamics_with_multiple_parameter_arrays(self, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            test_data,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device=device, meteo_range_checks=False)

        # Setting an array with arbitrary shape (and one value)
        for param in ("RDRSTB", "SSATB"):
            if param in ("RDRSTB", "SSATB"):
                # AfgenTrait parameters need to have shape (N, M)
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
            stem_dynamics_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results = test_data["ModelResults"]

        # Assertions on values removed as test data is not appropriate for this module
        assert len(actual_results) == len(expected_results)

    def test_stem_dynamics_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            _,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device="cpu")

        # Setting a vector (with one value) for the TDWI and RDRSTB parameters,
        # but with different lengths
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "RDRSTB", crop_model_params_provider["RDRSTB"].repeat(5, 1), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                stem_dynamics_config,
                external_states,
                device="cpu",
            )

    def test_stem_dynamics_with_incompatible_weather_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        (
            _,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = _prepare_common_stem_inputs(test_data_url, device="cpu", meteo_range_checks=False)

        # Setting vectors with incompatible shapes: TDWI and TEMP
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones(5, dtype=torch.float64) * wdc.TEMP

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                stem_dynamics_config,
                external_states,
                device="cpu",
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_stem_dynamics(self, test_data_url):
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["TDWI", "RDRSTB", "SSATB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Stem_Dynamics", WOFOST_Stem_Dynamics):
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


class TestDiffStemDynamicsGradients:
    """Parametrized tests for gradient calculations in stem dynamics."""

    # Define parameters and outputs
    param_names = ["TDWI", "RDRSTB", "SSATB"]
    output_names = ["SAI", "TWST"]

    # Define parameter configurations (value, dtype)
    param_configs = {
        "single": {
            "TDWI": (0.2, torch.float64),
            "RDRSTB": ([[0.0, 0.0, 1.5, 0.025, 2.1, 0.05]], torch.float64),
            "SSATB": ([[0.0, 0.0003, 2.0, 0.0003]], torch.float64),
        },
        "tensor": {
            "TDWI": ([0.1, 0.2, 0.3], torch.float64),
            "RDRSTB": (
                [
                    [0.0, 0.0, 1.5, 0.020, 2.1, 0.045],
                    [0.0, 0.0, 1.5, 0.025, 2.1, 0.050],
                    [0.0, 0.0, 1.5, 0.030, 2.1, 0.055],
                ],
                torch.float64,
            ),
            "SSATB": (
                [
                    [0.0, 0.00025, 2.0, 0.00025],
                    [0.0, 0.00030, 2.0, 0.00030],
                    [0.0, 0.00035, 2.0, 0.00035],
                ],
                torch.float64,
            ),
        },
    }

    # Define which parameter-output pairs should have gradients
    # Format: {param_name: [list of outputs that should have gradients]}
    gradient_mapping = {
        "TDWI": ["SAI", "TWST"],
        "RDRSTB": ["TWST", "SAI"],
        "SSATB": ["SAI"],
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
        model = get_test_diff_stem_model(device=device)

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
        model = get_test_diff_stem_model(device=device)
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
            m = get_test_diff_stem_model(device=device)
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
