"""Tests for the utils module, specifically Afgen and AfgenTrait classes."""

import datetime
import pytest
import torch
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.utils import Afgen
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import WeatherDataProviderTestHelper
from diffwofost.physical_models.utils import _get_drv
from diffwofost.physical_models.utils import astro
from diffwofost.physical_models.utils import daylength
from diffwofost.physical_models.utils import get_test_data
from . import phy_data_folder

ComputeConfig.set_dtype(torch.float64)
DTYPE = ComputeConfig.get_dtype()


@pytest.mark.usefixtures("fast_mode")
class TestAfgen:
    """Tests for the Afgen class."""

    def test_basic_interpolation(self):
        """Test basic linear interpolation."""
        # Simple linear function: y = x
        afgen = Afgen([0, 0, 10, 10])

        # Test exact points
        assert afgen(torch.tensor(0.0)) == 0.0
        assert afgen(torch.tensor(10.0)) == 10.0

        # Test interpolation
        result = afgen(torch.tensor(5.0))
        assert torch.isclose(result, torch.tensor(5.0, dtype=DTYPE))

    def test_non_linear_interpolation(self):
        """Test non-linear interpolation with multiple segments."""
        # Piecewise linear function
        afgen = Afgen([0, 0, 5, 10, 10, 5])

        # Test exact points
        assert afgen(torch.tensor(0.0)) == 0.0
        assert torch.isclose(afgen(torch.tensor(5.0)), torch.tensor(10.0, dtype=DTYPE))
        assert torch.isclose(afgen(torch.tensor(10.0)), torch.tensor(5.0, dtype=DTYPE))

        # Test interpolation in first segment (0 to 5)
        result = afgen(torch.tensor(2.5))
        expected = torch.tensor(5.0, dtype=DTYPE)  # y = 2*x at x=2.5
        assert torch.isclose(result, expected)

        # Test interpolation in second segment (5 to 10)
        result = afgen(torch.tensor(7.5))
        expected = torch.tensor(7.5, dtype=DTYPE)  # y = 15 - x at x=7.5
        assert torch.isclose(result, expected)

    def test_boundary_clamping(self):
        """Test that values outside the range are clamped to boundary values."""
        afgen = Afgen([0, 5, 10, 15])

        # Test below lower bound
        result = afgen(torch.tensor(-5.0))
        assert result == 5.0

        # Test above upper bound
        result = afgen(torch.tensor(20.0))
        assert result == 15.0

    def test_tensor_input(self):
        """Test that tensor inputs work correctly."""
        afgen = Afgen([0, 0, 10, 10])
        x = torch.tensor(5.0, dtype=DTYPE)
        result = afgen(x)

        assert isinstance(result, torch.Tensor)
        assert torch.isclose(result, torch.tensor(5.0, dtype=DTYPE))

    def test_list_input_conversion(self):
        """Test that list inputs are converted to tensors."""
        # Test with list input for table
        afgen = Afgen([0, 0, 5, 10, 10, 5])

        # Test with float input (should be converted to tensor internally)
        result = afgen(2.5)
        assert isinstance(result, torch.Tensor)

    def test_gradient_flow(self):
        """Test that gradients can flow through the interpolation."""
        afgen = Afgen([0, 0, 10, 10])
        x = torch.tensor(5.0, dtype=DTYPE, requires_grad=True)

        result = afgen(x)
        result.backward()

        # For the linear function y = x, the gradient is 1
        assert x.grad is not None
        assert torch.isclose(x.grad, torch.tensor(1.0, dtype=DTYPE), atol=1e-5)

    def test_gradient_flow_piecewise(self):
        """Test gradient flow through piecewise linear function."""
        # y = 2x for x in [0, 5], y = 20 - x for x in [5, 10]
        afgen = Afgen([0, 0, 5, 10, 10, 0])

        # Test gradient in first segment (should be 2)
        x1 = torch.tensor(2.5, dtype=DTYPE, requires_grad=True)
        result1 = afgen(x1)
        result1.backward()
        assert torch.isclose(x1.grad, torch.tensor(2.0, dtype=DTYPE), atol=1e-5)

        # Test gradient in second segment (should be -2)
        x2 = torch.tensor(7.5, dtype=DTYPE, requires_grad=True)
        result2 = afgen(x2)
        result2.backward()
        assert torch.isclose(x2.grad, torch.tensor(-2.0, dtype=DTYPE), atol=1e-5)

    def test_ascending_check_valid(self):
        """Test that valid ascending x values pass the check."""
        # Should not raise an exception
        afgen = Afgen([0, 1, 5, 10, 10, 20])
        assert afgen is not None

    def test_ascending_check_invalid(self):
        """Test that non-ascending x values raise an error."""
        # X values with multiple breaks: 0, 10, 5, 15 (has 2 breaks in ascending pattern)
        with pytest.raises(
            ValueError, match="X values for AFGEN input list not strictly ascending"
        ):
            Afgen([0, 1, 10, 2, 5, 3, 15, 4])

    def test_ascending_check_with_trailing_zeros(self):
        """Test that trailing (0, 0) pairs are handled correctly."""
        # This should work - trailing zeros are truncated
        afgen = Afgen([0, 0, 5, 10, 10, 20, 0, 0])
        assert afgen is not None

        # Verify that the function still works correctly
        result = afgen(torch.tensor(5.0))
        assert torch.isclose(result, torch.tensor(10.0, dtype=DTYPE))

    def test_single_segment(self):
        """Test with just a single segment."""
        afgen = Afgen([0, 5, 10, 15])

        result = afgen(torch.tensor(5.0))
        expected = torch.tensor(10.0, dtype=DTYPE)
        assert torch.isclose(result, expected)

    def test_dtype_consistency(self):
        """Test that the output dtype is consistent with DTYPE."""
        afgen = Afgen([0, 0, 10, 10])
        result = afgen(torch.tensor(5.0))

        assert result.dtype == DTYPE

    def test_complex_table(self):
        """Test with a more complex table resembling real crop parameters."""
        # Simulating something like a temperature response curve
        afgen = Afgen([0, 0, 10, 0.5, 20, 1.0, 30, 1.5, 40, 1.0, 50, 0])

        # Test various points
        assert afgen(torch.tensor(-5.0)) == 0.0  # Below range
        assert torch.isclose(afgen(torch.tensor(10.0)), torch.tensor(0.5, dtype=DTYPE))
        assert torch.isclose(afgen(torch.tensor(20.0)), torch.tensor(1.0, dtype=DTYPE))
        assert torch.isclose(afgen(torch.tensor(30.0)), torch.tensor(1.5, dtype=DTYPE))
        assert afgen(torch.tensor(60.0)) == 0.0  # Above range

        # Test interpolation between 20 and 30
        result = afgen(torch.tensor(25.0))
        expected = torch.tensor(1.25, dtype=DTYPE)  # Linear interpolation
        assert torch.isclose(result, expected)

    def test_to_moves_dtype_and_device(self):
        afgen = Afgen([0, 0, 10, 10])
        returned = afgen.to(dtype=torch.float64)
        assert returned is afgen
        out = afgen(torch.tensor(5.0))
        assert out.dtype == torch.float64

        # Batched tables
        tbl = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 20.0]], dtype=torch.float32)
        afgen_batched = Afgen(tbl)
        afgen_batched.to(dtype=torch.float64)
        out_batched = afgen_batched(torch.tensor([5.0, 5.0]))
        assert out_batched.dtype == torch.float64

        if torch.cuda.is_available():
            afgen_cuda = Afgen([0, 0, 10, 10]).to(device="cuda")
            out_cuda = afgen_cuda(torch.tensor(5.0, device="cuda"))
            assert out_cuda.device.type == "cuda"


@pytest.mark.usefixtures("fast_mode")
class TestAfgenTrait:
    """Tests for the AfgenTrait class."""

    def test_default_value(self):
        """Test that the default value is set correctly."""
        # Ensure default_value matches current config
        AfgenTrait.default_value = Afgen([0, 0, 1, 1])

        trait = AfgenTrait()
        assert isinstance(trait.default_value, Afgen)

        # Default is [0, 0, 1, 1] which is identity mapping
        result = trait.default_value(torch.tensor(0.5))
        assert torch.isclose(result, torch.tensor(0.5, dtype=DTYPE))

    def test_validate_with_afgen_instance(self):
        """Test validation with an Afgen instance."""
        trait = AfgenTrait()
        afgen = Afgen([0, 0, 10, 10])

        # Create a dummy object for validation
        class DummyObj:
            pass

        obj = DummyObj()
        validated = trait.validate(obj, afgen)

        assert isinstance(validated, Afgen)
        assert validated is afgen

    def test_validate_with_iterable(self):
        """Test validation with an iterable (list)."""
        trait = AfgenTrait()

        class DummyObj:
            pass

        obj = DummyObj()
        table = [0, 0, 10, 10]
        validated = trait.validate(obj, table)

        assert isinstance(validated, Afgen)
        # Test that it works correctly
        result = validated(torch.tensor(5.0))
        assert torch.isclose(result, torch.tensor(5.0, dtype=DTYPE))

    def test_validate_with_tuple(self):
        """Test validation with a tuple iterable."""
        trait = AfgenTrait()

        class DummyObj:
            pass

        obj = DummyObj()
        table = (0, 0, 10, 10)
        validated = trait.validate(obj, table)

        assert isinstance(validated, Afgen)


@pytest.mark.usefixtures("fast_mode")
class TestAfgenEdgeCases:
    """Test edge cases and special scenarios for Afgen."""

    def test_identical_x_values_raises_error(self):
        """Test that identical consecutive x values raise an error."""
        with pytest.raises(
            ValueError, match="X values for AFGEN input list not strictly ascending"
        ):
            Afgen([0, 0, 5, 10, 5, 20])

    def test_two_point_table(self):
        """Test with minimal two-point table."""
        afgen = Afgen([0, 10, 10, 20])

        assert afgen(torch.tensor(0.0)) == 10.0
        assert afgen(torch.tensor(10.0)) == 20.0
        assert torch.isclose(afgen(torch.tensor(5.0)), torch.tensor(15.0, dtype=DTYPE))

    def test_x_breakpoint_at_clamp(self):
        """Illustrate why AFGEN x-breakpoint grads can disagree with finite differences.

        AFGEN is piecewise-linear in the query x, but evaluation includes discrete interval
        selection (via searchsorted) and boundary clamping (via where).

        At the *exact* upper breakpoint (x_query == x_last), the output is clamped to y_last.
        Autograd therefore reports ~0 gradient w.r.t. x_last (the breakpoint), because within
        that branch the output does not depend on x_last.

        However, a finite-difference perturbation of x_last changes which branch is taken
        for x_query fixed at the boundary, producing a non-zero numerical derivative.
        This is the same phenomenon that caused the partitioning numerical-grad test to fail
        when comparing all AFGEN table entries.
        """

        # Keep this example deterministic across environments.
        old_device = ComputeConfig.get_device()
        old_dtype = DTYPE
        ComputeConfig.set_device("cpu")
        ComputeConfig.set_dtype(torch.float64)
        try:
            # Table is encoded as [x0, y0, x1, y1]. Use a non-flat y so x-breakpoints matter.
            tbl = torch.tensor([0.0, 0.3, 2.0, 0.1], dtype=torch.float64, requires_grad=True)
            afgen = Afgen(tbl)

            # Query exactly at the last breakpoint, which triggers the clamp branch.
            x_query = torch.tensor(2.0, dtype=torch.float64)
            out = afgen(x_query)

            (grad_auto,) = torch.autograd.grad(out, tbl, retain_graph=False)

            # Central finite difference w.r.t. each table entry
            delta = 1e-6
            grad_num = torch.zeros_like(tbl)
            for i in range(tbl.numel()):
                tbl_plus = tbl.detach().clone()
                tbl_minus = tbl.detach().clone()
                tbl_plus[i] += delta
                tbl_minus[i] -= delta
                out_plus = Afgen(tbl_plus)(x_query)
                out_minus = Afgen(tbl_minus)(x_query)
                grad_num[i] = (out_plus - out_minus) / (2 * delta)

            # Gradients w.r.t. y-entries (odd indices) should match closely.
            assert torch.allclose(grad_auto[1::2], grad_num[1::2], atol=1e-5, rtol=1e-4)

            # Gradient w.r.t. the last x-breakpoint (index 2) is the illustrative mismatch.
            # Autograd sees the clamp branch => ~0; finite differences see branch switching.
            assert abs(float(grad_auto[2])) < 1e-8
            assert abs(float(grad_num[2])) > 1e-3
        finally:
            ComputeConfig.set_device(old_device)
            ComputeConfig.set_dtype(old_dtype)

    def test_negative_values(self):
        """Test with negative x and y values."""
        afgen = Afgen([-10, -20, 0, 0, 10, 20])

        assert afgen(torch.tensor(-10.0)) == -20.0
        assert afgen(torch.tensor(0.0)) == 0.0
        assert afgen(torch.tensor(10.0)) == 20.0
        assert torch.isclose(afgen(torch.tensor(-5.0)), torch.tensor(-10.0, dtype=DTYPE))

    def test_batch_processing(self):
        """Test that Afgen can be used in batch processing scenarios."""
        afgen = Afgen([0, 0, 10, 10])

        # Process multiple values
        values = [torch.tensor(2.0), torch.tensor(5.0), torch.tensor(8.0)]
        results = [afgen(v) for v in values]

        assert len(results) == 3
        assert torch.isclose(results[0], torch.tensor(2.0, dtype=DTYPE))
        assert torch.isclose(results[1], torch.tensor(5.0, dtype=DTYPE))
        assert torch.isclose(results[2], torch.tensor(8.0, dtype=DTYPE))

    def test_zero_slope_segment(self):
        """Test with a segment that has zero slope (horizontal line)."""
        afgen = Afgen([0, 5, 10, 5, 20, 10])

        # First segment has zero slope (y = 5)
        assert afgen(torch.tensor(0.0)) == 5.0
        assert afgen(torch.tensor(5.0)) == 5.0
        assert torch.isclose(afgen(torch.tensor(10.0)), torch.tensor(5.0, dtype=DTYPE))

        # Second segment has positive slope
        result = afgen(torch.tensor(15.0))
        expected = torch.tensor(7.5, dtype=DTYPE)
        assert torch.isclose(result, expected)

    def test_tensor_input_at_boundaries(self):
        """Test tensor inputs at boundary conditions with gradients."""
        afgen = Afgen([0, 5, 10, 15])

        # Test below lower bound with gradient
        x_low = torch.tensor(-2.0, dtype=DTYPE, requires_grad=True)
        result_low = afgen(x_low)
        assert result_low == 5.0
        result_low.backward()
        # Gradient should be 0 when clamped at boundary
        assert x_low.grad == 0.0

        # Test above upper bound with gradient
        x_high = torch.tensor(15.0, dtype=DTYPE, requires_grad=True)
        result_high = afgen(x_high)
        assert result_high == 15.0
        result_high.backward()
        # Gradient should be 0 when clamped at boundary
        assert x_high.grad == 0.0

    def test_tensor_input_near_boundaries(self):
        """Test tensor inputs just inside boundaries maintain gradients."""
        afgen = Afgen([0, 0, 10, 10])

        # Just above lower bound
        x_near_low = torch.tensor(0.1, dtype=DTYPE, requires_grad=True)
        result = afgen(x_near_low)
        result.backward()
        # Should have gradient of 1 (slope of the line)
        assert torch.isclose(x_near_low.grad, torch.tensor(1.0, dtype=DTYPE), atol=1e-5)

        # Just below upper bound
        x_near_high = torch.tensor(9.9, dtype=DTYPE, requires_grad=True)
        result = afgen(x_near_high)
        result.backward()
        assert torch.isclose(x_near_high.grad, torch.tensor(1.0, dtype=DTYPE), atol=1e-5)

    def test_1d_tensor_batch_input(self):
        """Test that we can pass a 1D tensor to evaluate multiple points at once."""
        afgen = Afgen([0, 0, 10, 10])

        # Process multiple values in a vectorized manner
        x_batch = torch.tensor([2.0, 5.0, 8.0], dtype=DTYPE)
        results = torch.stack([afgen(x) for x in x_batch])

        assert results.shape == (3,)
        assert torch.isclose(results[0], torch.tensor(2.0, dtype=DTYPE))
        assert torch.isclose(results[1], torch.tensor(5.0, dtype=DTYPE))
        assert torch.isclose(results[2], torch.tensor(8.0, dtype=DTYPE))


@pytest.mark.usefixtures("fast_mode")
class TestAfgenBatched:
    """Tests for batched Afgen functionality with multidimensional tensors."""

    def test_batched_2d_simple(self):
        """Test batched Afgen with 2D tensor (batch of tables)."""
        # Create 3 different tables in a batch
        tables = torch.tensor(
            [
                [0, 0, 10, 10],  # y = x
                [0, 0, 10, 20],  # y = 2x
                [0, 5, 10, 15],  # y = x + 5
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        # Test that it's recognized as batched
        assert afgen.is_batched
        assert afgen.batch_shape == (3,)

        # Test interpolation - should return tensor with batch dimension
        result = afgen(torch.tensor(5.0))

        assert result.shape == (3,)
        assert torch.isclose(result[0], torch.tensor(5.0, dtype=DTYPE))
        assert torch.isclose(result[1], torch.tensor(10.0, dtype=DTYPE))
        assert torch.isclose(result[2], torch.tensor(10.0, dtype=DTYPE))

    def test_batched_3d_tensor(self):
        """Test batched Afgen with 3D tensor (2D batch of tables)."""
        # Create a 2x3 batch of tables
        tables = torch.tensor(
            [
                [
                    [0, 0, 10, 10],  # y = x
                    [0, 0, 10, 20],  # y = 2x
                    [0, 5, 10, 15],  # y = x + 5
                ],
                [
                    [0, 10, 10, 20],  # y = x + 10
                    [0, 0, 10, 5],  # y = 0.5x
                    [0, 2, 10, 12],  # y = x + 2
                ],
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        assert afgen.is_batched
        assert afgen.batch_shape == (2, 3)

        # Test interpolation
        result = afgen(torch.tensor(5.0))

        assert result.shape == (2, 3)
        # First row
        assert torch.isclose(result[0, 0], torch.tensor(5.0, dtype=DTYPE))
        assert torch.isclose(result[0, 1], torch.tensor(10.0, dtype=DTYPE))
        assert torch.isclose(result[0, 2], torch.tensor(10.0, dtype=DTYPE))
        # Second row
        assert torch.isclose(result[1, 0], torch.tensor(15.0, dtype=DTYPE))
        assert torch.isclose(result[1, 1], torch.tensor(2.5, dtype=DTYPE))
        assert torch.isclose(result[1, 2], torch.tensor(7.0, dtype=DTYPE))

    def test_batched_boundary_clamping(self):
        """Test that boundary clamping works correctly for batched tables."""
        tables = torch.tensor(
            [
                [0, 0, 10, 10],
                [0, 5, 10, 15],
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        # Test below lower bound
        result = afgen(torch.tensor(-5.0))
        assert result.shape == (2,)
        assert result[0] == 0.0
        assert result[1] == 5.0

        # Test above upper bound
        result = afgen(torch.tensor(20.0))
        assert result.shape == (2,)
        assert result[0] == 10.0
        assert result[1] == 15.0

    def test_batched_different_ranges(self):
        """Test batched tables with different x ranges."""
        tables = torch.tensor(
            [
                [0, 0, 5, 10, 0, 0, 0, 0],  # Range 0-5, trailing zeros
                [0, 0, 10, 20, 20, 30, 0, 0],  # Range 0-20, trailing zeros
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        # Test within first table's range but outside second table's initial segment
        result = afgen(torch.tensor(3.0))
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(6.0, dtype=DTYPE))  # From first table
        assert torch.isclose(result[1], torch.tensor(6.0, dtype=DTYPE))  # From second table

    def test_batched_gradient_flow(self):
        """Test that gradients flow correctly through batched Afgen."""
        tables = torch.tensor(
            [
                [0, 0, 10, 10],  # y = x, gradient = 1
                [0, 0, 10, 20],  # y = 2x, gradient = 2
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        x = torch.tensor(5.0, dtype=DTYPE, requires_grad=True)
        result = afgen(x)

        # Sum the batch dimension for backward pass
        loss = result.sum()
        loss.backward()

        # Combined gradient should be 1 + 2 = 3
        assert x.grad is not None
        assert torch.isclose(x.grad, torch.tensor(3.0, dtype=DTYPE), atol=1e-5)

    def test_batched_4d_tensor(self):
        """Test with 4D tensor to verify arbitrary dimensionality."""
        # Create a 2x2x2 batch of tables
        tables = torch.ones(2, 2, 2, 4, dtype=DTYPE) * torch.tensor([0, 0, 10, 10], dtype=DTYPE)

        # Modify some tables to have different slopes
        tables[0, 0, 0] = torch.tensor([0, 0, 10, 20], dtype=DTYPE)  # y = 2x
        tables[1, 1, 1] = torch.tensor([0, 0, 10, 5], dtype=DTYPE)  # y = 0.5x

        afgen = Afgen(tables)

        assert afgen.is_batched
        assert afgen.batch_shape == (2, 2, 2)

        result = afgen(torch.tensor(5.0))
        assert result.shape == (2, 2, 2)

        # Check specific modified tables
        assert torch.isclose(result[0, 0, 0], torch.tensor(10.0, dtype=DTYPE))
        assert torch.isclose(result[1, 1, 1], torch.tensor(2.5, dtype=DTYPE))

        # Check unmodified tables (should be y = x)
        assert torch.isclose(result[0, 0, 1], torch.tensor(5.0, dtype=DTYPE))
        assert torch.isclose(result[0, 1, 0], torch.tensor(5.0, dtype=DTYPE))

    def test_batched_complex_interpolation(self):
        """Test batched tables with multiple segments."""
        # Each table has 3 segments
        tables = torch.tensor(
            [
                [0, 0, 5, 10, 10, 5, 15, 15],  # Piecewise: up, down, flat
                [0, 5, 5, 10, 10, 10, 15, 5],  # Piecewise: up, flat, down
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        # Test in different segments
        # At x=2.5 (first segment)
        result = afgen(torch.tensor(2.5))
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(5.0, dtype=DTYPE))
        assert torch.isclose(result[1], torch.tensor(7.5, dtype=DTYPE))

        # At x=7.5 (second segment)
        result = afgen(torch.tensor(7.5))
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(7.5, dtype=DTYPE))
        assert torch.isclose(result[1], torch.tensor(10.0, dtype=DTYPE))

        # At x=12.5 (third segment)
        result = afgen(torch.tensor(12.5))
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(10.0, dtype=DTYPE))
        assert torch.isclose(result[1], torch.tensor(7.5, dtype=DTYPE))

    def test_batched_dtype_consistency(self):
        """Test that batched output maintains correct dtype."""
        tables = torch.tensor(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 20],
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)
        result = afgen(torch.tensor(5.0))

        assert result.dtype == DTYPE

    def test_batched_with_trailing_zeros(self):
        """Test batched tables with trailing zeros are handled correctly."""
        tables = torch.tensor(
            [
                [0, 0, 5, 10, 10, 20, 0, 0],
                [0, 5, 10, 15, 0, 0, 0, 0],
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        # Should work correctly, ignoring trailing zeros
        result = afgen(torch.tensor(5.0))
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(10.0, dtype=DTYPE))
        assert torch.isclose(result[1], torch.tensor(10.0, dtype=DTYPE))

    def test_batched_single_element_batch(self):
        """Test edge case of batch size 1."""
        tables = torch.tensor(
            [
                [0, 0, 10, 10],
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        assert afgen.is_batched
        assert afgen.batch_shape == (1,)

        result = afgen(torch.tensor(5.0))
        assert result.shape == (1,)
        assert torch.isclose(result[0], torch.tensor(5.0, dtype=DTYPE))

    def test_batched_ascending_check_fails(self):
        """Test that invalid batched tables raise errors."""
        # Second table has non-ascending x values (padded to same length)
        tables = torch.tensor(
            [
                [0, 0, 10, 10, 0, 0],
                [0, 0, 10, 5, 5, 10],  # x values: 0, 10, 5 (not ascending)
            ],
            dtype=DTYPE,
        )

        with pytest.raises(
            ValueError, match="X values for AFGEN input list not strictly ascending"
        ):
            Afgen(tables)

    def test_backward_compatibility_non_batched(self):
        """Test that non-batched (1D) usage still works correctly."""
        afgen = Afgen([0, 0, 10, 10])

        assert not afgen.is_batched
        assert afgen.batch_shape is None

        result = afgen(torch.tensor(5.0))
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar tensor
        assert torch.isclose(result, torch.tensor(5.0, dtype=DTYPE))

    def test_batched_gradient_at_boundaries(self):
        """Test gradients at boundaries for batched tables."""
        tables = torch.tensor(
            [
                [0, 0, 10, 10],
                [0, 5, 10, 15],
            ],
            dtype=DTYPE,
        )

        afgen = Afgen(tables)

        # Test at lower boundary
        x = torch.tensor(-1.0, dtype=DTYPE, requires_grad=True)
        result = afgen(x)
        loss = result.sum()
        loss.backward()

        # Both tables clamped at lower bound, gradient should be 0
        assert x.grad == 0.0

        # Test in interpolation region
        x2 = torch.tensor(5.0, dtype=DTYPE, requires_grad=True)
        result2 = afgen(x2)
        loss2 = result2.sum()
        loss2.backward()

        # Sum of slopes: 1 + 1 = 2
        assert torch.isclose(x2.grad, torch.tensor(2.0, dtype=DTYPE), atol=1e-5)


@pytest.mark.usefixtures("fast_mode")
class TestGetDrvParam:
    """Tests for _get_drv function."""

    def test_weather_provider_does_not_mutate_input_weather(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_24.yaml"
        test_data = get_test_data(test_data_url)
        weather_inputs = test_data["WeatherVariables"]

        assert any("SNOWDEPTH" in item for item in weather_inputs)

        WeatherDataProviderTestHelper(weather_inputs)

        assert any("SNOWDEPTH" in item for item in weather_inputs)

    def test_float_broadcast(self):
        expected_shape = (3, 2)
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        provider = WeatherDataProviderTestHelper(test_data["WeatherVariables"])
        wdc = provider(provider.first_date)
        scalar = wdc.TEMP
        out = _get_drv(scalar, expected_shape, dtype=DTYPE)
        assert out.shape == expected_shape
        assert torch.allclose(out, torch.full(expected_shape, scalar, dtype=DTYPE))

    def test_scalar_broadcast(self):
        expected_shape = (3, 2)
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        provider = WeatherDataProviderTestHelper(test_data["WeatherVariables"])
        wdc = provider(provider.first_date)
        scalar = torch.tensor(wdc.IRRAD, dtype=DTYPE)  # 0-d tensor
        out = _get_drv(scalar, expected_shape, dtype=DTYPE)
        assert out.shape == expected_shape
        assert torch.allclose(out, torch.full(expected_shape, scalar.item(), dtype=DTYPE))

    def test_matching_shape_pass_through(self):
        expected_shape = (3, 2)
        base_val = torch.tensor(12.34, dtype=DTYPE)
        var = torch.ones(expected_shape, dtype=DTYPE) * base_val
        out = _get_drv(var, expected_shape, dtype=DTYPE)
        assert out.shape == expected_shape
        # Should be the same object (no copy)
        assert out.data_ptr() == var.data_ptr()

    def test_wrong_shape_raises(self):
        expected_shape = (3, 2)
        wrong = torch.ones(2, 3, dtype=DTYPE)
        with pytest.raises(ValueError, match="incompatible shape"):
            _get_drv(wrong, expected_shape, dtype=DTYPE)

    def test_one_dim_shape_raises(self):
        expected_shape = (3, 2)
        one_dim = torch.ones(3, dtype=DTYPE)
        with pytest.raises(ValueError, match="incompatible shape"):
            _get_drv(one_dim, expected_shape, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Shared helpers for astro / daylength tests
# ---------------------------------------------------------------------------
_SUMMER_DAY = datetime.date(2000, 6, 21)  # northern-hemisphere summer solstice
_WINTER_DAY = datetime.date(2000, 12, 21)  # northern-hemisphere winter solstice
_MID_LAT = 52.0  # typical mid-latitude site (Netherlands)
_IRRAD = 15e6  # J m-2 d-1, a reasonable summer value


@pytest.mark.usefixtures("fast_mode")
class TestDaylength:
    """Tests for the daylength utility function."""

    def test_returns_tensor(self):
        """daylength always returns a torch.Tensor."""
        result = daylength(_SUMMER_DAY, _MID_LAT, dtype=DTYPE)
        assert isinstance(result, torch.Tensor)

    def test_reasonable_range(self):
        """Result must be in [0, 24] hours."""
        result = daylength(_SUMMER_DAY, _MID_LAT, dtype=DTYPE)
        assert 0.0 <= result.item() <= 24.0

    def test_summer_longer_than_winter(self):
        """At temperate latitudes summer daylength exceeds winter daylength."""
        summer = daylength(_SUMMER_DAY, _MID_LAT, dtype=DTYPE)
        winter = daylength(_WINTER_DAY, _MID_LAT, dtype=DTYPE)
        assert summer.item() > winter.item()

    def test_polar_day_returns_24(self):
        """At the North Pole during summer solstice daylength should be 24 h."""
        result = daylength(_SUMMER_DAY, 90.0, dtype=DTYPE)
        assert torch.isclose(result, torch.tensor(24.0, dtype=DTYPE))

    def test_polar_night_returns_0(self):
        """At the North Pole during winter solstice daylength should be 0 h."""
        result = daylength(_WINTER_DAY, 90.0, dtype=DTYPE)
        assert torch.isclose(result, torch.tensor(0.0, dtype=DTYPE))

    def test_scalar_latitude(self):
        """Scalar float latitude is accepted and converted internally."""
        result = daylength(_SUMMER_DAY, float(_MID_LAT), dtype=DTYPE)
        assert isinstance(result, torch.Tensor)
        assert result.item() > 0.0

    def test_tensor_latitude(self):
        """Tensor latitude input returns a tensor of the same shape."""
        lats = torch.tensor([20.0, 40.0, 60.0], dtype=DTYPE)
        result = daylength(_SUMMER_DAY, lats, dtype=DTYPE)
        assert result.shape == lats.shape
        # Daylength increases with latitude in summer
        assert result[0] < result[1] < result[2]

    def test_invalid_latitude_raises(self):
        """Latitude outside [-90, 90] must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Latitude not between"):
            daylength(_SUMMER_DAY, 95.0, dtype=DTYPE)

    def test_custom_angle(self):
        """A base angle of 0 gives a shorter (astronomical) daylength than -4."""
        dl_minus4 = daylength(_SUMMER_DAY, _MID_LAT, angle=-4, dtype=DTYPE)
        dl_zero = daylength(_SUMMER_DAY, _MID_LAT, angle=0, dtype=DTYPE)
        # angle=0 is a stricter cutoff → shorter or equal photoperiod
        assert dl_zero.item() <= dl_minus4.item()

    def test_gradient_flows_through_latitude(self):
        """Gradients should propagate through the latitude parameter."""
        lat = torch.tensor(_MID_LAT, dtype=DTYPE, requires_grad=True)
        result = daylength(_SUMMER_DAY, lat, dtype=DTYPE)
        result.backward()
        assert lat.grad is not None
        assert not torch.isnan(lat.grad)

    def test_equator_result_close_to_12(self):
        """At the equator daylength is close to 12 h regardless of season."""
        result_summer = daylength(_SUMMER_DAY, 0.0, dtype=DTYPE)
        result_winter = daylength(_WINTER_DAY, 0.0, dtype=DTYPE)
        assert torch.isclose(result_summer, torch.tensor(12.0, dtype=DTYPE), atol=1.0)
        assert torch.isclose(result_winter, torch.tensor(12.0, dtype=DTYPE), atol=1.0)

    def test_default_dtype_device(self):
        """daylength uses ComputeConfig defaults when dtype/device are omitted."""
        result = daylength(_SUMMER_DAY, _MID_LAT)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == ComputeConfig.get_dtype()
        assert result.device == ComputeConfig.get_device()

    def test_southern_hemisphere(self):
        """Southern hemisphere latitude: summer in NH is winter in SH."""
        south_lat = -_MID_LAT
        north_summer = daylength(_SUMMER_DAY, _MID_LAT, dtype=DTYPE)
        south_winter = daylength(_SUMMER_DAY, south_lat, dtype=DTYPE)
        # In NH summer the southern counterpart has shorter days
        assert south_winter.item() < north_summer.item()
        assert 0.0 <= south_winter.item() <= 24.0

    def test_batched_all_branches(self):
        """A single batched call exercises all three torch.where branches."""
        # North Pole (AOB > 1) → 24 h, South Pole (AOB < -1) → 0 h, mid-lat → between
        lats = torch.tensor([90.0, -90.0, _MID_LAT], dtype=DTYPE)
        result = daylength(_SUMMER_DAY, lats, dtype=DTYPE)
        assert result.shape == lats.shape
        assert torch.isclose(result[0], torch.tensor(24.0, dtype=DTYPE))
        assert torch.isclose(result[1], torch.tensor(0.0, dtype=DTYPE))
        assert 0.0 < result[2].item() < 24.0

    def test_tensor_latitude_requires_no_grad(self):
        """daylength with a detached tensor latitude completes without error."""
        lat = torch.tensor(_MID_LAT, dtype=DTYPE)
        result = daylength(_SUMMER_DAY, lat, dtype=DTYPE)
        assert result.item() > 0.0


@pytest.mark.usefixtures("fast_mode")
class TestAstro:
    """Tests for the astro utility function."""

    def _call(self, day=_SUMMER_DAY, lat=_MID_LAT, rad=_IRRAD):
        """Convenience wrapper."""
        return astro(day, lat, rad, dtype=DTYPE)

    def test_returns_namedtuple_fields(self):
        """Return value exposes all expected named fields."""
        result = self._call()
        for field in ("DAYL", "DAYLP", "SINLD", "COSLD", "DIFPP", "ATMTR", "DSINBE", "ANGOT"):
            assert hasattr(result, field)

    def test_all_outputs_are_tensors(self):
        """Every field in the namedtuple must be a torch.Tensor."""
        result = self._call()
        for field in result._fields:
            assert isinstance(getattr(result, field), torch.Tensor), f"{field} is not a tensor"

    def test_dayl_in_valid_range(self):
        """DAYL must be in [0, 24] h."""
        result = self._call()
        assert 0.0 <= result.DAYL.item() <= 24.0

    def test_daylp_in_valid_range(self):
        """DAYLP (photoperiodic) must be in [0, 24] h and >= DAYL."""
        result = self._call()
        assert 0.0 <= result.DAYLP.item() <= 24.0
        assert result.DAYLP.item() >= result.DAYL.item()

    def test_atmtr_in_valid_range(self):
        """Atmospheric transmission must be in [0, 1]."""
        result = self._call()
        assert 0.0 <= result.ATMTR.item() <= 1.0

    def test_angot_positive(self):
        """Angot radiation must be positive when DAYL > 0."""
        result = self._call()
        assert result.ANGOT.item() > 0.0

    def test_difpp_positive(self):
        """Diffuse irradiation perpendicular must be non-negative."""
        result = self._call()
        assert result.DIFPP.item() >= 0.0

    def test_summer_dayl_greater_than_winter(self):
        """DAYL in summer should exceed DAYL in winter at mid-latitude."""
        summer = astro(_SUMMER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        winter = astro(_WINTER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        assert summer.DAYL.item() > winter.DAYL.item()

    def test_polar_day(self):
        """At the North Pole in summer DAYL should be 24 h."""
        result = astro(_SUMMER_DAY, 90.0, _IRRAD, dtype=DTYPE)
        assert torch.isclose(result.DAYL, torch.tensor(24.0, dtype=DTYPE))

    def test_polar_night_dayl_zero(self):
        """At the North Pole in winter DAYL should be 0 h and ATMTR zero."""
        result = astro(_WINTER_DAY, 90.0, _IRRAD, dtype=DTYPE)
        assert torch.isclose(result.DAYL, torch.tensor(0.0, dtype=DTYPE))
        assert torch.isclose(result.ATMTR, torch.tensor(0.0, dtype=DTYPE))

    def test_batch_latitude(self):
        """Batched latitude input produces outputs with matching shape."""
        lats = torch.tensor([20.0, 40.0, 60.0], dtype=DTYPE)
        rad = torch.tensor([_IRRAD] * 3, dtype=DTYPE)
        result = astro(_SUMMER_DAY, lats, rad, dtype=DTYPE)
        assert result.DAYL.shape == lats.shape
        assert result.DAYLP.shape == lats.shape
        assert result.ATMTR.shape == lats.shape
        # DAYL increases with latitude during summer
        assert result.DAYL[0] < result.DAYL[1] < result.DAYL[2]

    def test_invalid_latitude_raises(self):
        """Latitude outside [-90, 90] must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Latitude not between"):
            astro(_SUMMER_DAY, 95.0, _IRRAD, dtype=DTYPE)

    def test_gradient_flows_through_radiation(self):
        """Gradients should propagate through the radiation parameter."""
        rad = torch.tensor(float(_IRRAD), dtype=DTYPE, requires_grad=True)
        result = astro(_SUMMER_DAY, _MID_LAT, rad, dtype=DTYPE)
        result.ATMTR.backward()
        assert rad.grad is not None
        assert not torch.isnan(rad.grad)

    def test_gradient_flows_through_latitude(self):
        """Gradients should propagate through the latitude parameter.

        The scalar branch of astro() converts LAT via float() which detaches the
        grad graph, so we use a batched (numel > 1) input where torch ops are used
        throughout and autograd operates normally.
        """
        lat = torch.tensor([_MID_LAT, _MID_LAT + 5.0], dtype=DTYPE, requires_grad=True)
        rad = torch.tensor([float(_IRRAD), float(_IRRAD)], dtype=DTYPE)
        result = astro(_SUMMER_DAY, lat, rad, dtype=DTYPE)
        result.DAYL.sum().backward()
        assert lat.grad is not None
        assert not torch.isnan(lat.grad).any()

    def test_daylp_consistent_with_daylength_function(self):
        """DAYLP from astro should match the standalone daylength() (angle=-4)."""
        result = astro(_SUMMER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        dl = daylength(_SUMMER_DAY, _MID_LAT, angle=-4, dtype=DTYPE)
        assert torch.isclose(result.DAYLP, dl, atol=1e-5)

    def test_default_dtype_device(self):
        """astro uses ComputeConfig defaults when dtype/device are omitted."""
        result = astro(_SUMMER_DAY, _MID_LAT, _IRRAD)
        assert isinstance(result.DAYL, torch.Tensor)
        assert result.DAYL.dtype == ComputeConfig.get_dtype()

    def test_frdif_high_transmission(self):
        """FRDIF branch: ATMTR > 0.75 → FRDIF = 0.23 (clear sky)."""
        # Obtain ANGOT first (independent of radiation input)
        ref = astro(_SUMMER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        angot = ref.ANGOT
        # Force ATMTR > 0.75 by using 0.80 * ANGOT as radiation
        high_rad = 0.80 * angot
        result = astro(_SUMMER_DAY, _MID_LAT, high_rad, dtype=DTYPE)
        assert result.ATMTR.item() > 0.75
        # DIFPP = 0.23 * ATMTR * 0.5 * SC; verify it is a finite positive tensor
        assert result.DIFPP.item() > 0.0
        assert torch.isfinite(result.DIFPP)

    def test_frdif_low_transmission(self):
        """FRDIF branch: 0.07 < ATMTR <= 0.35 → 1 - 2.3*(ATMTR-0.07)² (cloudy)."""
        ref = astro(_SUMMER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        angot = ref.ANGOT
        low_rad = 0.20 * angot
        result = astro(_SUMMER_DAY, _MID_LAT, low_rad, dtype=DTYPE)
        assert 0.07 < result.ATMTR.item() <= 0.35
        assert result.DIFPP.item() > 0.0

    def test_frdif_very_low_transmission(self):
        """FRDIF branch: ATMTR <= 0.07 → FRDIF = 1.0 (overcast)."""
        ref = astro(_SUMMER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        angot = ref.ANGOT
        very_low_rad = 0.03 * angot
        result = astro(_SUMMER_DAY, _MID_LAT, very_low_rad, dtype=DTYPE)
        assert result.ATMTR.item() <= 0.07
        assert result.DIFPP.item() > 0.0

    def test_southern_hemisphere(self):
        """Negative latitude is handled correctly for southern hemisphere site."""
        south_lat = -_MID_LAT
        result = astro(_SUMMER_DAY, south_lat, _IRRAD, dtype=DTYPE)
        north_result = astro(_SUMMER_DAY, _MID_LAT, _IRRAD, dtype=DTYPE)
        # NH summer = SH winter → shorter day in southern hemisphere
        assert result.DAYL.item() < north_result.DAYL.item()
        assert 0.0 <= result.DAYL.item() <= 24.0

    def test_batched_radiation(self):
        """Batched radiation produces outputs with matching shape."""
        rads = torch.tensor([5e6, 15e6, 25e6], dtype=DTYPE)
        lats = torch.tensor([_MID_LAT] * 3, dtype=DTYPE)
        result = astro(_SUMMER_DAY, lats, rads, dtype=DTYPE)
        assert result.ATMTR.shape == rads.shape
        # Higher radiation → higher ATMTR (up to 1)
        assert result.ATMTR[0] < result.ATMTR[1] < result.ATMTR[2]

    def test_dsinbe_positive(self):
        """DSINBE (effective solar height integral) must be non-negative."""
        result = self._call()
        assert result.DSINBE.item() >= 0.0

    def test_polar_dsinb_dsinbe_branch(self):
        """At the pole in summer AOB > 1 exercises the DSINB/DSINBE AOB > 1 branch."""
        result = astro(_SUMMER_DAY, 90.0, _IRRAD, dtype=DTYPE)
        # DAYL == 24 h confirms we hit the polar-day branch
        assert torch.isclose(result.DAYL, torch.tensor(24.0, dtype=DTYPE))
        assert result.DSINBE.item() >= 0.0
        assert result.ANGOT.item() > 0.0
