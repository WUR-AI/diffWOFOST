import numpy as np
import pytest
import torch
import yaml

from tests.physical_models.pcse_test_code import TestEngine, TestWeatherDataProvider
from pcse.base.parameter_providers import ParameterProvider
from pcse.models import Wofost72_PP
from pcse.engine import Engine
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
from diffwofost.physical_models.conf import phy_conf_folder
from .. import phy_data_folder
from unittest.mock import patch


def prepare_engine_input(file_path):
    inputs = yaml.safe_load(open(file_path))
    agro = inputs["AgroManagement"]
    cropd = inputs["ModelParameters"]

    wdp = TestWeatherDataProvider(inputs["WeatherVariables"])
    params = ParameterProvider(cropdata=cropd)
    external_states = inputs["ExternalStates"]
    return params, wdp, agro, external_states


def get_test_data(file_path):
    inputs = yaml.safe_load(open(file_path))
    return inputs["ModelResults"], inputs["Precision"]


class TestLeafDynamics:
    def test_leaf_dynamics_with_TestEngine(self):
        """TestEngine and not Engine because it allows to specify `external_states`."""

        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        params, wdp, agro, external_states = prepare_engine_input(test_data_path)

        config_path = str(phy_conf_folder / "WOFOST_Leaf_Dynamics.conf")

        # convert external states to tensors
        tensor_external_states = [
            {
                k: v if k == 'DAY' else torch.tensor(v, dtype=torch.float32)
                for k, v in item.items()
            }
            for item in external_states
        ]

        engine = TestEngine(params, wdp, agro, config_path, tensor_external_states)
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results):
            assert reference["DAY"] == model["day"]
            assert all(
                abs(reference[var] - model[var]) < precision
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_Engine(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        params, wdp, agro, _ = prepare_engine_input(test_data_path)

        config_path = str(phy_conf_folder / "WOFOST_Leaf_Dynamics.conf")

        # Engine does not allows to specify `external_states`
        with pytest.raises(ValueError):
            engine = Engine(params, wdp, agro, config_path)

    def test_wofost_pp_with_leaf_dynamics(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_potentialproduction_wofost72_01.yaml"
        params, wdp, agro, _ = prepare_engine_input(test_data_path)

        with patch(
            'pcse.crop.leaf_dynamics.WOFOST_Leaf_Dynamics',
            WOFOST_Leaf_Dynamics
            ):
            model = Wofost72_PP(params, wdp, agro)
            model.run_till_terminate()
            actual_results = model.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results):
            assert reference["DAY"] == model["day"]
            assert all(
                abs(reference[var] - model[var]) < precision
                for var, precision in expected_precision.items()
            )
