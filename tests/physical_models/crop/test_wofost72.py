import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.wofost72 import Wofost72
from diffwofost.physical_models.soil.classic_waterbalance import WaterbalancePP
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

wofost72_config = Configuration(
    CROP=Wofost72,
    SOIL=WaterbalancePP,
    OUTPUT_VARS=["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
)

# All output variables used in the differentiable model for gradient tests (mirrors OUTPUT_VARS)
GRAD_OUTPUT_VARS = ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"]


def get_test_diff_wofost72_model():
    test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = [
        # Leaf dynamics
        "SPAN",
        "TDWI",
        "TBASE",
        "PERDL",
        "RGRLAI",
        "KDIFTB",
        "SLATB",
        # Phenology
        "TSUMEM",
        "TBASEM",
        "TEFFMX",
        "TSUM1",
        "TSUM2",
        "DLO",
        "DLC",
        "DVSI",
        "DVSEND",
        "DTSMTB",
        # Assimilation (KDIFTB already included above)
        "AMAXTB",
        "EFFTB",
        "TMPFTB",
        "TMNFTB",
        # Respiration
        "Q10",
        "RMR",
        "RML",
        "RMS",
        "RMO",
        "RFSETB",
        # Evapotranspiration (KDIFTB already included, IAIRDU already in root)
        "CFET",
        "DEPNR",
        "IAIRDU",
        "IOX",
        "CRAIRC",
        "SM0",
        "SMW",
        "SMFCF",
        # Root dynamics (TDWI already included)
        "RDI",
        "RRI",
        "RDMCR",
        "RDMSOL",
        "RDRRTB",
        # Stem dynamics (TDWI already included)
        "RDRSTB",
        "SSATB",
        # Storage organ dynamics
        "SPA",
        # Partitioning
        "FRTB",
        "FLTB",
        "FSTB",
        "FOTB",
        # Wofost72 top-level conversion factors
        "CVL",
        "CVO",
        "CVR",
        "CVS",
    ]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params)
    )
    return DiffWofost72(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        wofost72_config,
        copy.deepcopy(external_states),
    )


class DiffWofost72(torch.nn.Module):
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
        # pass new value of parameters to the model
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

        return {var: torch.stack([item[var] for item in results]) for var in GRAD_OUTPUT_VARS}


class TestWofost72:
    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost72_with_testengine(self, test_data_url, device):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
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
            wofost72_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            # Verify output is on the correct device
            for var in expected_precision.keys():
                assert model[var].device.type == device, f"{var} should be on {device}"
            # Move to CPU for comparison if needed
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                abs(reference[var] - model_cpu[var]) < precision
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize(
        "param", ["TDWI", "SPAN", "RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB", "TEMP"]
    )
    def test_wofost72_with_one_parameter_vector(self, param, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        # Setting a vector (with one value) for the selected parameter
        if param == "TEMP":
            # Vectorize weather variable
            for (_, _), wdc in weather_data_provider.store.items():
                base = wdc.TEMP
                if isinstance(base, torch.Tensor):
                    ones = torch.ones(10, dtype=base.dtype, device=base.device)
                    wdc.TEMP = ones * base
                else:
                    wdc.TEMP = torch.ones(10, dtype=torch.float64) * base
        elif param in ["KDIFTB", "SLATB"]:
            # AfgenTrait parameters need to have shape (N, M)
            repeated = crop_model_params_provider[param].repeat(10, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        if param == "TEMP":
            # Expect error due to incompatible shapes
            # (By defaults parameters are not reshaped following weather variables)
            with pytest.raises(ValueError):
                engine = EngineTestHelper(
                    crop_model_params_provider,
                    weather_data_provider,
                    agro_management_inputs,
                    wofost72_config,
                    external_states,
                )
                engine.run_till_terminate()
                actual_results = engine.get_output()
        else:
            engine = EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                wofost72_config,
                external_states,
            )
            engine.run_till_terminate()
            actual_results = engine.get_output()

            # get expected results from YAML test data
            expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

            assert len(actual_results) == len(expected_results)

            for reference, model in zip(expected_results, actual_results, strict=False):
                assert reference["DAY"] == model["day"]
                # Verify output is on the correct device
                for var in expected_precision.keys():
                    assert model[var].device.type == device, f"{var} should be on {device}"
                # Move to CPU for comparison
                model_cpu = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()
                }
                assert all(
                    all(abs(reference[var] - model_cpu[var]) < precision)
                    for var, precision in expected_precision.items()
                )

    @pytest.mark.parametrize(
        "param,delta",
        [
            ("TDWI", 0.1),
            ("SPAN", 5),
            ("TBASE", 2.0),
            ("PERDL", 0.01),
            ("RGRLAI", 0.002),
            ("KDIFTB", 0.1),
            ("SLATB", 0.0005),
        ],
    )
    def test_wofost72_with_different_parameter_values(self, param, delta, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector with multiple values for the selected parameter
        test_value = crop_model_params_provider[param]
        # We set the value for which test data are available as the last element
        if param in {"KDIFTB", "SLATB"}:
            # AfgenTrait parameters need to have shape (N, M)
            non_zeros_mask = test_value != 0
            param_vec = torch.stack([test_value + non_zeros_mask * delta, test_value])
        else:
            param_vec = torch.stack([test_value - delta, test_value + delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            wofost72_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            # Verify output is on the correct device
            for var in expected_precision.keys():
                assert model[var].device.type == device, f"{var} should be on {device}"
            # Move to CPU for comparison
            model_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
            assert all(
                # The value for which test data are available is the last element
                abs(reference[var] - model_cpu[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_wofost72_with_multiple_parameter_vectors(self, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for multiple parameters simultaneously
        for param in ("TDWI", "SPAN", "RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB"):
            if param in ("KDIFTB", "SLATB"):
                # AfgenTrait parameters need to have shape (N, M)
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            wofost72_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_wofost72_with_multiple_parameter_arrays(self, device):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        # Setting an array with arbitrary shape (and one value)
        for param in ("RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB"):
            if param in ("KDIFTB", "SLATB"):
                # AfgenTrait parameters need to have shape (N, M)
                repeated = crop_model_params_provider[param].repeat(30, 5, 1)
            else:
                repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

        for (_, _), wdc in weather_data_provider.store.items():
            base = wdc.TEMP
            if isinstance(base, torch.Tensor):
                ones = torch.ones((30, 5), dtype=base.dtype, device=base.device)
                wdc.TEMP = ones * base
            else:
                wdc.TEMP = torch.ones((30, 5), dtype=torch.float64) * base

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            wofost72_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                torch.all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )
            assert all(
                model[var].shape == (30, 5) for var in expected_precision.keys()
            )  # check the output shapes

    def test_wofost72_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for the TDWI and SPAN parameters,
        # but with different lengths
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "SPAN", crop_model_params_provider["SPAN"].repeat(5), check=False
        )

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                wofost72_config,
                external_states,
            )

    def test_wofost72_with_incompatible_weather_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        # Setting vectors with incompatible shapes: TDWI and TEMP
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones(5, dtype=torch.float64) * wdc.TEMP

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                wofost72_config,
                external_states,
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost72_against_pcse_pp(self, test_data_url):
        """Test that diffWOFOST Wofost72 gives the same results as PCSE Wofost72_PP."""
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Wofost72", Wofost72):
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


class TestDiffWofost72Gradients:
    """Parametrized tests for gradient calculations in Wofost72.

    Tests ALL output variables (matching OUTPUT_VARS) against ALL differentiable
    crop parameters spanning every submodule (phenology, leaf dynamics, assimilation,
    respiration, evapotranspiration, root dynamics, stem dynamics, storage organ
    dynamics, partitioning, and top-level conversion factors).
    """

    # All output variables from wofost72_config (mirrors GRAD_OUTPUT_VARS)
    output_names = ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"]

    # All differentiable crop parameters across every submodule
    param_names = [
        # Leaf dynamics
        "TDWI",
        "SPAN",
        "RGRLAI",
        "TBASE",
        "PERDL",
        "KDIFTB",
        "SLATB",
        # Phenology
        "TSUMEM",
        "TBASEM",
        "TEFFMX",
        "TSUM1",
        "TSUM2",
        "DLO",
        "DLC",
        "DVSI",
        "DVSEND",
        "DTSMTB",
        # Assimilation (KDIFTB shared with leaf dynamics / evapotranspiration)
        "AMAXTB",
        "EFFTB",
        "TMPFTB",
        "TMNFTB",
        # Respiration
        "Q10",
        "RMR",
        "RML",
        "RMS",
        "RMO",
        "RFSETB",
        # Evapotranspiration
        "CFET",
        "DEPNR",
        "CRAIRC",
        "SMW",
        "SMFCF",
        "SM0",
        # Root dynamics
        "RDI",
        "RRI",
        "RDMCR",
        "RDMSOL",
        "RDRRTB",
        # Stem dynamics
        "RDRSTB",
        "SSATB",
        # Storage organ dynamics
        "SPA",
        # Partitioning
        "FRTB",
        "FLTB",
        "FSTB",
        "FOTB",
        # Wofost72 top-level conversion factors
        "CVL",
        "CVO",
        "CVR",
        "CVS",
    ]

    # Define parameter configurations (value, dtype)
    param_configs = {
        "single": {
            # Leaf dynamics
            "TDWI": (0.2, torch.float64),
            "SPAN": (30, torch.float64),
            "RGRLAI": (0.016, torch.float64),
            "TBASE": (3.0, torch.float64),
            "PERDL": (0.03, torch.float64),
            "KDIFTB": ([[0.0, 0.6, 2.0, 0.6]], torch.float64),
            "SLATB": ([[0.0, 0.002, 2.0, 0.002]], torch.float64),
            # Phenology
            "TSUMEM": (50.0, torch.float64),
            "TBASEM": (0.0, torch.float64),
            "TEFFMX": (35.0, torch.float64),
            "TSUM1": (500.0, torch.float64),
            "TSUM2": (600.0, torch.float64),
            "DLO": (0.5, torch.float64),
            "DLC": (0.5, torch.float64),
            "DVSI": (0.0, torch.float64),
            "DVSEND": (2.0, torch.float64),
            "DTSMTB": ([0.0, 0.0, 35.0, 35.0, 45.0, 35.0], torch.float64),
            # Assimilation
            "AMAXTB": ([[0.0, 30.0, 2.0, 30.0]], torch.float64),
            "EFFTB": ([[0.0, 0.0005, 40.0, 0.0005]], torch.float64),
            "TMPFTB": ([[0.0, 1.0, 40.0, 1.0]], torch.float64),
            "TMNFTB": ([[-10.0, 0.0, 0.0, 1.0, 10.0, 1.0]], torch.float64),
            # Respiration
            "Q10": (2.0, torch.float64),
            "RMR": (0.015, torch.float64),
            "RML": (0.03, torch.float64),
            "RMS": (0.02, torch.float64),
            "RMO": (0.01, torch.float64),
            "RFSETB": ([[-1.0, 1.0, 3.0, 0.75]], torch.float64),
            # Evapotranspiration
            "CFET": (1.0, torch.float64),
            "DEPNR": (2.0, torch.float64),
            "CRAIRC": (0.06, torch.float64),
            "SMW": (0.15, torch.float64),
            "SMFCF": (0.29, torch.float64),
            "SM0": (0.40, torch.float64),
            # Root dynamics
            "RDI": (10.0, torch.float64),
            "RRI": (2.25, torch.float64),
            "RDMCR": (121.0, torch.float64),
            "RDMSOL": (121.0, torch.float64),
            "RDRRTB": ([[0.0, 0.0, 1.5, 0.02]], torch.float64),
            # Stem dynamics
            "RDRSTB": ([[0.0, 0.0, 1.5, 0.025, 2.1, 0.05]], torch.float64),
            "SSATB": ([[0.0, 0.0003, 2.0, 0.0003]], torch.float64),
            # Storage organ dynamics
            "SPA": (0.01, torch.float64),
            # Partitioning
            # FLTB+FSTB+FOTB must sum to 1.0 at EVERY DVS for both the TDWI
            # conservation check (initialize) and the carbon-balance check (calc_rates)
            "FRTB": ([[0.0, 0.3, 2.0, 0.1]], torch.float64),
            "FLTB": (
                [[0.0, 0.85, 1.0, 0.5, 1.3, 0.05, 1.57, 0.05, 1.92, 0.05, 2.0, 0.05]],
                torch.float64,
            ),
            "FSTB": (
                [[0.0, 0.15, 1.0, 0.5, 1.3, 0.10, 1.57, 0.10, 1.92, 0.05, 2.0, 0.05]],
                torch.float64,
            ),
            "FOTB": (
                [[0.0, 0.00, 1.0, 0.0, 1.3, 0.85, 1.57, 0.85, 1.92, 0.90, 2.0, 0.90]],
                torch.float64,
            ),
            # Wofost72 top-level conversion factors
            "CVL": (0.72, torch.float64),
            "CVO": (0.80, torch.float64),
            "CVR": (0.72, torch.float64),
            "CVS": (0.69, torch.float64),
        },
        "tensor": {
            # Leaf dynamics
            "TDWI": ([0.1, 0.2, 0.3], torch.float64),
            "SPAN": ([25, 30, 35], torch.float64),
            "RGRLAI": ([-10, 0.08, 1], torch.float64),
            "TBASE": ([-5, 0, 10.0], torch.float64),
            "PERDL": ([-10, 0.1, 15], torch.float64),
            "KDIFTB": (
                [[0.0, 0.5, 10.0, 1.0], [0.0, 0.6, 12.0, 1.2], [0.0, 0.4, 8.0, 0.8]],
                torch.float64,
            ),
            "SLATB": (
                [
                    [0.0, 0.002031, 0.5, 0.002031, 2.0, 0.002031],
                    [0.0, 0.0025, 0.6, 0.0025, 2.5, 0.0025],
                    [0.0, 0.0015, 0.4, 0.0015, 1.5, 0.0015],
                ],
                torch.float64,
            ),
            # Phenology
            "TSUMEM": ([45.0, 50.0, 55.0], torch.float64),
            "TBASEM": ([-2.0, 0.0, 2.0], torch.float64),
            "TEFFMX": ([32.0, 35.0, 38.0], torch.float64),
            "TSUM1": ([450.0, 500.0, 550.0], torch.float64),
            "TSUM2": ([550.0, 600.0, 650.0], torch.float64),
            "DLO": ([0.4, 0.5, 0.6], torch.float64),
            "DLC": ([0.4, 0.5, 0.6], torch.float64),
            "DVSI": ([0.0, 0.0, 0.0], torch.float64),
            "DVSEND": ([1.9, 2.0, 2.1], torch.float64),
            "DTSMTB": (
                [
                    [0, 0, 15, 8, 30, 18],
                    [0, 0, 5, 9, 10, 19],
                    [0, 0, 25, 1, 30, 20],
                ],
                torch.float64,
            ),
            # Assimilation
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
            # Respiration
            "Q10": ([1.5, 2.0, 2.5], torch.float64),
            "RMR": ([0.01, 0.015, 0.02], torch.float64),
            "RML": ([0.02, 0.03, 0.04], torch.float64),
            "RMS": ([0.01, 0.02, 0.03], torch.float64),
            "RMO": ([0.005, 0.01, 0.02], torch.float64),
            "RFSETB": (
                [[-1.0, 1.0, 3.0, 0.70], [-1.0, 1.0, 3.0, 0.75], [-1.0, 1.0, 3.0, 0.80]],
                torch.float64,
            ),
            # Evapotranspiration
            "CFET": ([0.8, 1.0, 1.2], torch.float64),
            "DEPNR": ([1.0, 2.0, 4.0], torch.float64),
            "CRAIRC": ([0.04, 0.06, 0.08], torch.float64),
            "SMW": ([0.12, 0.15, 0.18], torch.float64),
            "SMFCF": ([0.26, 0.29, 0.32], torch.float64),
            "SM0": ([0.37, 0.40, 0.43], torch.float64),
            # Root dynamics
            "RDI": ([9.0, 10.0, 11.0], torch.float64),
            "RRI": ([2.0, 2.25, 2.5], torch.float64),
            "RDMCR": ([110.0, 121.0, 130.0], torch.float64),
            "RDMSOL": ([110.0, 121.0, 130.0], torch.float64),
            "RDRRTB": ([[0.0, 0.0, 1.5, 0.02], [0.0, 0.0, 1.6, 0.03]], torch.float64),
            # Stem dynamics
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
            # Storage organ dynamics
            "SPA": ([0.01, 0.02, 0.03], torch.float64),
            # Partitioning
            "FRTB": (
                [[0.0, 0.3, 2.0, 0.1], [0.0, 0.4, 2.0, 0.2], [0.0, 0.2, 2.0, 0.05]],
                torch.float64,
            ),
            # Each of FLTB, FSTB, FOTB must use the exact same knot structure as
            # the YAML file so FL+FS+FO=1 holds at every interpolated DVS during the
            # full simulation run
            "FLTB": (
                [
                    [0.0, 0.85, 1.0, 0.5, 1.3, 0.05, 1.57, 0.05, 1.92, 0.05, 2.0, 0.05],
                    [0.0, 0.85, 1.0, 0.5, 1.3, 0.05, 1.57, 0.05, 1.92, 0.05, 2.0, 0.05],
                    [0.0, 0.85, 1.0, 0.5, 1.3, 0.05, 1.57, 0.05, 1.92, 0.05, 2.0, 0.05],
                ],
                torch.float64,
            ),
            "FSTB": (
                [
                    [0.0, 0.15, 1.0, 0.5, 1.3, 0.10, 1.57, 0.10, 1.92, 0.05, 2.0, 0.05],
                    [0.0, 0.15, 1.0, 0.5, 1.3, 0.10, 1.57, 0.10, 1.92, 0.05, 2.0, 0.05],
                    [0.0, 0.15, 1.0, 0.5, 1.3, 0.10, 1.57, 0.10, 1.92, 0.05, 2.0, 0.05],
                ],
                torch.float64,
            ),
            "FOTB": (
                [
                    [0.0, 0.00, 1.0, 0.0, 1.3, 0.85, 1.57, 0.85, 1.92, 0.90, 2.0, 0.90],
                    [0.0, 0.00, 1.0, 0.0, 1.3, 0.85, 1.57, 0.85, 1.92, 0.90, 2.0, 0.90],
                    [0.0, 0.00, 1.0, 0.0, 1.3, 0.85, 1.57, 0.85, 1.92, 0.90, 2.0, 0.90],
                ],
                torch.float64,
            ),
            # Wofost72 top-level conversion factors
            "CVL": ([0.68, 0.72, 0.76], torch.float64),
            "CVO": ([0.76, 0.80, 0.84], torch.float64),
            "CVR": ([0.68, 0.72, 0.76], torch.float64),
            "CVS": ([0.65, 0.69, 0.73], torch.float64),
        },
    }

    # Define which parameter-output pairs should have gradients.
    # Derived from individual-module gradient tests, projected onto WOFOST72 outputs.
    # Format: {param_name: [list of outputs that should have gradients]}
    gradient_mapping = {
        # --- Leaf dynamics ---
        # TDWI sets initial DM for all organs → affects every biomass output
        "TDWI": ["LAI", "TWLV", "TAGP", "TWSO", "TWST", "TWRT", "TRA"],
        "SPAN": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],
        "RGRLAI": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],
        "TBASE": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],
        "PERDL": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],
        # KDIFTB used in assimilation (PGASS) and evapotranspiration (TRA)
        "KDIFTB": ["LAI", "TAGP", "TWLV", "TWSO", "TWST", "TWRT", "TRA"],
        "SLATB": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],
        # --- Phenology ---
        # Phenology params drive DVS via temperature-sum accumulation
        "TSUMEM": ["DVS", "LAI", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "TBASEM": ["DVS", "LAI", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "TEFFMX": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "TSUM1": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "TSUM2": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "DLO": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "DLC": ["DVS", "LAI", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "DVSI": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "DVSEND": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        "DTSMTB": ["DVS", "LAI", "RD", "TAGP", "TRA", "TWLV", "TWRT", "TWSO", "TWST"],
        # --- Assimilation ---
        # PGASS → GASS → ASRC → DMI distributed to all organs via partitioning
        "AMAXTB": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "EFFTB": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "TMPFTB": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "TMNFTB": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        # --- Respiration ---
        # MRES subtracted from GASS → ASRC → DMI to all organs; RFSETB scales MRES
        "Q10": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "RMR": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "RML": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "RMS": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "RMO": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        "RFSETB": ["LAI", "TRA", "TAGP", "TWLV", "TWSO", "TWST", "TWRT"],
        # --- Evapotranspiration ---
        # In potential production (WaterbalancePP) these govern TRA only
        "CFET": ["TRA"],
        "DEPNR": ["TRA"],
        "CRAIRC": ["TRA"],
        "SMW": ["TRA"],
        "SMFCF": ["TRA"],
        "SM0": ["TRA"],
        # --- Root dynamics ---
        "RDI": ["RD"],
        "RRI": ["RD"],
        "RDMCR": ["RD"],
        "RDMSOL": ["RD"],
        "RDRRTB": ["TWRT", "LAI", "TWLV", "TWST", "TWSO", "TAGP", "TRA"],
        # --- Stem dynamics ---
        "RDRSTB": ["TWST", "LAI", "TRA", "TWLV", "TWSO", "TWRT", "TAGP"],
        "SSATB": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],  # SAI is part of LAI
        # --- Storage organ dynamics ---
        "SPA": ["LAI", "TRA", "TWLV", "TWST", "TWSO", "TWRT", "TAGP"],  # PAI is part of LAI
        # --- Partitioning ---
        # Each table controls DM allocation to the respective organ
        # FR appears in ADMI=(1-FR)*DMI and CVF → all organs affected
        "FRTB": ["TWRT", "TWLV", "LAI", "TWST", "TWSO", "TAGP", "TRA"],
        # FL/FS/FO each appear in CVF → DMI changes affect all organs
        "FLTB": ["TWLV", "LAI", "TWST", "TWSO", "TWRT", "TAGP", "TRA"],
        "FSTB": ["TWST", "TWLV", "LAI", "TWSO", "TWRT", "TAGP", "TRA"],
        "FOTB": ["TWSO", "TAGP", "TWLV", "LAI", "TWST", "TWRT", "TRA"],
        # --- Wofost72 top-level conversion factors ---
        # CVx = assimilate→dry-matter conversion; affects the growth of each organ
        # Each CVx appears in CVF = 1/((FL/CVL+FS/CVS+FO/CVO)*(1-FR)+FR/CVR) → all organs
        "CVL": ["TWLV", "TAGP", "LAI", "TWST", "TWSO", "TWRT", "TRA"],
        "CVO": ["TWSO", "TAGP", "TWLV", "LAI", "TWST", "TWRT", "TRA"],
        "CVR": ["TWRT", "TWLV", "LAI", "TWST", "TWSO", "TAGP", "TRA"],
        "CVS": ["TWST", "TAGP", "TWLV", "LAI", "TWSO", "TWRT", "TRA"],
    }

    # Generate all combinations of (param, output) and classify them
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
        model = get_test_diff_wofost72_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()
        # print(f"Testing no gradients for param '{param_name}' w.r.t. output '{output_name}'")
        # print(f"Output value: {output[output_name].data}, Loss: {loss.data},"
        #       f" Requires grad: {loss.requires_grad}")

        # If the output is fully disconnected from `param`, `loss` will not
        # require gradients and `torch.autograd.grad()` would raise. In that
        # case the correct result is "no gradient".
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
        """Test that forward and backward gradients match for parameter-output pairs."""
        model = get_test_diff_wofost72_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
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
        # for parameter `SPAN`
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device=device))
        numerical_grad = calculate_numerical_grad(
            lambda: get_test_diff_wofost72_model(), param_name, param, output_name
        )

        model = get_test_diff_wofost72_model()
        output = model({param_name: param})
        loss = output[output_name].sum()

        # this is ∂loss/∂param, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        # for span, the numerical gradient can't be equal to the pytorch one
        # because we are using STE method
        if param_name != "SPAN":
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
