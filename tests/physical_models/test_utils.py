"""Tests for the utils module, specifically Afgen and AfgenTrait classes."""

import pytest
import torch
from diffwofost.physical_models.utils import DTYPE
from diffwofost.physical_models.utils import Afgen
from diffwofost.physical_models.utils import AfgenTrait


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
