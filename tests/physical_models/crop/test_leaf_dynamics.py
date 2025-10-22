import copy
from unittest.mock import patch
import pytest
import torch
import torch.testing
from numpy.testing import assert_almost_equal
from pcse.engine import Engine
from pcse.models import Wofost72_PP
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder


def get_test_diff_leaf_model():
    test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
    crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data_path, crop_model_params)
    )
    config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")
    return DiffLeafDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        config_path,
        copy.deepcopy(external_states),
    )


class DiffLeafDynamics(torch.nn.Module):
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

        return torch.stack(
            [torch.stack([item["LAI"], item["TWLV"]]) for item in results]
        ).unsqueeze(0)  # shape: [1, time_steps, 2]


class TestLeafDynamics:
    def test_leaf_dynamics_with_testengine(self):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""

        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                abs(reference[var] - model[var]) < precision
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_engine(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path, crop_model_params)
        )

        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Engine does not allows to specify `external_states`
        with pytest.raises(ValueError):
            Engine(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
            )

    def test_leaf_dynamics_with_one_parameter_vector(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector (with one value) for the TDWI parameter
        param = "TDWI"
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

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_different_parameter_values(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector with multiple values for the TDWI parameter
        param = "TDWI"
        test_value = crop_model_params_provider[param]
        # We set the value for which test data are available as the last element
        param_vec = torch.tensor([test_value - 0.1, test_value + 0.1, test_value])
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

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                # The value for which test data are available is the last element
                abs(reference[var] - model[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_multiple_parameter_vectors(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector (with one value) for the TDWI and SPAN parameters
        for param in ("TDWI", "SPAN"):
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

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_multiple_parameter_arrays(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting an array with arbitrary shape (and one value) for the
        # TDWI and SPAN parameters
        for param in ("TDWI", "SPAN"):
            repeated = crop_model_params_provider[param].broadcast_to((30, 5))
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

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                torch.all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )
            assert all (
                model[var].shape == (30, 5)
                for var in expected_precision.keys()
            )  # check the output shapes

    def test_leaf_dynamics_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector (with one value) for the TDWI and SPAN parameters,
        # but with different lengths
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "SPAN", crop_model_params_provider["SPAN"].repeat(5), check=False
        )

        with pytest.raises(RuntimeError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
                external_states,
            )

    def test_wofost_pp_with_leaf_dynamics(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_potentialproduction_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        with patch("pcse.crop.wofost72.Leaf_Dynamics", WOFOST_Leaf_Dynamics):
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


class TestDiffLeafDynamicsTDWI:
    def test_gradients_tdwi_lai_leaf_dynamics(self):
        model = get_test_diff_leaf_model()
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        output = model({"TDWI": tdwi})
        lai = output[0, :, 0]
        loss = lai.sum()

        # this is ∂loss/∂tdwi without calling loss.backward().
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]
        assert grads is not None, "Gradients for TDWI should not be None"

        tdwi.grad = None  # clear any existing gradient
        loss.backward()
        # this is ∂loss/∂tdwi calculated using backpropagation
        grad_backward = tdwi.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"

    def test_gradients_tdwi_lai_leaf_dynamics_numerical(self):
        # first check if the numerical gradient isnot zero i.e. the parameter has an effect
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        output_index = 0  # LAI is at index 0
        numerical_grad = calculate_numerical_grad(
            get_test_diff_leaf_model, "TDWI", tdwi, output_index
        )  # this is Δloss/Δtdwi

        model = get_test_diff_leaf_model()
        output = model({"TDWI": tdwi})
        lai = output[0, :, output_index]
        loss = lai.sum()
        # this is ∂loss/∂tdwi, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]

        assert_almost_equal(numerical_grad, grads.item(), decimal=3)

    def test_gradients_tdwi_twlv_leaf_dynamics(self):
        # prepare model input
        model = get_test_diff_leaf_model()
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        output = model({"TDWI": tdwi})
        twlv = output[0, :, 1]
        loss = twlv.sum()

        # this is ∂loss/∂tdwi
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]
        assert grads is not None, "Gradients for TDWI should not be None"

        tdwi.grad = None  # clear any existing gradient
        loss.backward()
        # this is ∂loss/∂tdwi calculated using backpropagation
        grad_backward = tdwi.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"

    def test_gradients_tdwi_twlv_leaf_dynamics_numerical(self):
        # first check if the numerical gradient isnot zero i.e. the parameter has an effect
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        output_index = 1  # TWLV is at index 1
        numerical_grad = calculate_numerical_grad(
            get_test_diff_leaf_model, "TDWI", tdwi, output_index
        )  # this is Δloss/Δtdwi

        model = get_test_diff_leaf_model()
        output = model({"TDWI": tdwi})
        twlv = output[0, :, output_index]
        loss = twlv.sum()
        # this is ∂loss/∂tdwi, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]

        assert_almost_equal(numerical_grad, grads.item(), decimal=3)


class TestDiffLeafDynamicsSPAN:
    def test_gradients_span_lai_leaf_dynamics(self):
        # prepare model input
        model = get_test_diff_leaf_model()
        span = torch.nn.Parameter(torch.tensor(30, dtype=torch.float32))
        output = model({"SPAN": span})
        lai = output[0, :, 0]
        loss = lai.sum()

        # this is ∂loss/∂span
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, span, retain_graph=True)[0]
        assert grads is not None, "Gradients for SPAN should not be None"

        span.grad = None  # clear any existing gradient
        loss.backward()
        # this is ∂loss/∂span calculated using backpropagation
        grad_backward = span.grad

        assert grad_backward is not None, "Backward gradients for SPAN should not be None"
        assert grad_backward == grads, "Forward and backward gradients for SPAN should match"

    def test_gradients_span_lai_leaf_dynamics_numerical(self):
        # first check if the numerical gradient isnot zero i.e. the parameter has an effect
        span = torch.nn.Parameter(torch.tensor(30, dtype=torch.float64))
        output_index = 0  # LAI is at index 0
        numerical_grad = calculate_numerical_grad(
            get_test_diff_leaf_model, "SPAN", span, output_index
        )  # this is Δloss/Δspan

        model = get_test_diff_leaf_model()
        output = model({"SPAN": span})
        lai = output[0, :, output_index]
        loss = lai.sum()
        # this is ∂loss/∂tdwi, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, span, retain_graph=True)[0]

        assert_almost_equal(numerical_grad, grads.item(), decimal=3)

    def test_gradients_span_twlv_leaf_dynamics(self):
        # prepare model input
        model = get_test_diff_leaf_model()
        span = torch.nn.Parameter(torch.tensor(30, dtype=torch.float32))
        output = model({"SPAN": span})
        twlv = output[0, :, 1]
        loss = twlv.sum()

        # this is ∂loss/∂span
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, span, retain_graph=True)[0]
        assert grads is not None, "Gradients for SPAN should not be None"

        span.grad = None  # clear any existing gradient
        loss.backward()
        # this is ∂loss/∂span calculated using backpropagation
        grad_backward = span.grad

        assert grad_backward is not None, "Backward gradients for SPAN should not be None"
        assert grad_backward == grads, "Forward and backward gradients for SPAN should match"

    def test_gradients_span_twlv_leaf_dynamics_numerical(self):
        # first check if the numerical gradient isnot zero i.e. the parameter has an effect
        span = torch.nn.Parameter(torch.tensor(30, dtype=torch.float64))
        output_index = 1  # TWLV is at index 1
        numerical_grad = calculate_numerical_grad(
            get_test_diff_leaf_model, "SPAN", span, output_index
        )  # this is Δloss/Δspan

        model = get_test_diff_leaf_model()
        output = model({"SPAN": span})
        twlv = output[0, :, output_index]
        loss = twlv.sum()
        # this is ∂loss/∂tdwi, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, span, retain_graph=True)[0]

        assert numerical_grad == 0.0
        assert_almost_equal(numerical_grad, grads.item(), decimal=3)
