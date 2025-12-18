"""Tests for the utils module, specifically Afgen and AfgenTrait classes."""

import pytest
import torch
from diffwofost.physical_models.utils import Afgen
from diffwofost.physical_models.utils import AfgenTrait
from diffwofost.physical_models.utils import WeatherDataProviderTestHelper
from diffwofost.physical_models.utils import _get_drv
from diffwofost.physical_models.utils import get_test_data
from . import phy_data_folder

DTYPE = torch.float32  # Default dtype for tests


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


class TestAfgenTrait:
    """Tests for the AfgenTrait class."""

    def test_default_value(self):
        """Test that the default value is set correctly."""
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


class TestGetDrvParam:
    """Tests for _get_drv function."""

    def test_float_broadcast(self):
        expected_shape = (3, 2)
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        provider = WeatherDataProviderTestHelper(test_data["WeatherVariables"])
        wdc = provider(provider.first_date)
        scalar = wdc.TEMP
        out = _get_drv(scalar, expected_shape, dtype=DTYPE)
        assert out.shape == expected_shape
        assert torch.allclose(out, torch.full(expected_shape, scalar, dtype=DTYPE))

    def test_scalar_broadcast(self):
        expected_shape = (3, 2)
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
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
