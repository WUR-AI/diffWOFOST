import pytest
import torch
from pcse.base import VariableKiosk
from diffwofost.physical_models.base import TensorParamTemplate
from diffwofost.physical_models.base import TensorRatesTemplate
from diffwofost.physical_models.base import TensorStatesTemplate
from diffwofost.physical_models.traitlets import Tensor
from diffwofost.physical_models.utils import AfgenTrait


class TestTensorParamTemplate:
    class Params(TensorParamTemplate):
        A = Tensor(0)
        B = Tensor(0, dtype=int)

    class ParamsWithAfgen(TensorParamTemplate):
        A = Tensor(0)
        B = AfgenTrait()

    def test_template_automatically_cast_to_tensor_with_correct_type(self):
        p = self.Params(dict(A=1, B=[1, 1, 1]))
        assert isinstance(p.A, torch.Tensor)
        assert p.A.dtype == torch.float64
        assert isinstance(p.B, torch.Tensor)
        assert p.B.dtype == torch.int64

    def test_template_can_infer_shape_from_parameters(self):
        p = self.Params(dict(A=1, B=[1, 1, 1]))
        assert all(param.shape == (3,) for param in (p.A, p.B))
        assert p.shape == (3,)

    def test_template_can_apply_shape_from_argument(self):
        shape = (3,)
        p = self.Params(dict(A=1, B=1), shape=shape)
        assert all(param.shape == shape for param in (p.A, p.B))
        assert p.shape == shape

    def test_template_checks_consistency_of_parameter_and_input_shapes(self):
        # Here the input shape is consistent with the parameters
        shape = (3,)
        p = self.Params(dict(A=1, B=[1, 1, 1]), shape=shape)
        assert all(param.shape == shape for param in (p.A, p.B))
        assert p.shape == shape

        # Here it is not
        with pytest.raises(ValueError):
            self.Params(dict(A=1, B=[1, 1, 1]), shape=(5,))

    def test_template_allows_to_skip_broadcasting_of_variables(self):
        shape = (5,)
        p = self.Params(dict(A=1, B=[1, 1, 1]), shape=shape, do_not_broadcast=["B"])
        assert p.A.shape == shape
        assert p.B.shape == (3,)

    def test_template_recognizes_shape_of_afgen_tables(self):
        p = self.ParamsWithAfgen(dict(A=1, B=[0, 0, 1, 1]))
        assert p.shape == ()
        p = self.ParamsWithAfgen(dict(A=1, B=[[0, 0, 1, 1], [0, 0, 2, 2]]))
        assert p.shape == (2,)


class TestTensorRatesTemplate:
    class Rates(TensorRatesTemplate):
        A = Tensor(0)
        B = Tensor(0, dtype=int)

    def test_template_automatically_cast_to_tensor_with_correct_type(self):
        r = self.Rates(kiosk=VariableKiosk())
        assert isinstance(r.A, torch.Tensor)
        assert r.A.dtype == torch.float64
        assert isinstance(r.B, torch.Tensor)
        assert r.B.dtype == torch.int64

    def test_template_can_apply_shape_from_argument(self):
        shape = (3,)
        r = self.Rates(kiosk=VariableKiosk(), shape=shape)
        assert all(rate.shape == shape for rate in (r.A, r.B))
        assert r.shape == shape

    def test_template_allows_to_skip_broadcasting_of_variables(self):
        shape = (5,)
        r = self.Rates(kiosk=VariableKiosk(), shape=shape, do_not_broadcast=["B"])
        assert r.A.shape == shape
        assert r.B.shape == ()

    def test_template_allows_to_publish_in_kiosk(self):
        k = VariableKiosk()
        r = self.Rates(kiosk=k, publish=["A"])
        assert "A" in k
        assert k.A == 0.0
        r.A = torch.tensor(1.0)
        assert k.A == 1.0


class TestTensorStatesTemplate:
    class States(TensorStatesTemplate):
        A = Tensor(0)
        B = Tensor(0, dtype=int)

    def test_template_automatically_cast_to_tensor_with_correct_type(self):
        s = self.States(kiosk=VariableKiosk(), A=1, B=1)
        assert isinstance(s.A, torch.Tensor)
        assert s.A.dtype == torch.float64
        assert isinstance(s.B, torch.Tensor)
        assert s.B.dtype == torch.int64

    def test_template_can_apply_shape_from_argument(self):
        shape = (3,)
        s = self.States(kiosk=VariableKiosk(), shape=shape, A=1, B=1)
        assert all(state.shape == shape for state in (s.A, s.B))
        assert s.shape == shape

    def test_template_allows_to_skip_broadcasting_of_variables(self):
        shape = (5,)
        s = self.States(kiosk=VariableKiosk(), shape=shape, do_not_broadcast=["B"], A=1, B=1)
        assert s.A.shape == shape
        assert s.B.shape == ()

    def test_template_allows_to_publish_in_kiosk(self):
        k = VariableKiosk()
        s = self.States(kiosk=k, publish=["A"], A=0, B=0)
        assert "A" in k
        assert k.A == 0.0
        s.A = torch.tensor(1.0)
        assert k.A == 1.0
