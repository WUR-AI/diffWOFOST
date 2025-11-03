from collections.abc import Iterable
import torch
from pcse.traitlets import TraitType

DTYPE = torch.float64  # Default data type for tensors in this module


class Afgen:
    """Differentiable AFGEN function, expanded from pcse.

    :param tbl_xy: Tensor of XY value pairs describing the function
        the X values should be mononically increasing.

    Returns the interpolated value provided with the
    absicca value at which the interpolation should take place.

    example::

        >>> tbl_xy = torch.tensor([0,0,1,1,5,10])
        >>> f =  Afgen(tbl_xy)
        >>> f(torch.tensor(0.5))
        tensor(0.5)
        >>> f(torch.tensor(1.5))
        tensor(2.125)
    """

    def _check_x_ascending(self, tbl_xy):
        """Checks that the x values are strictly ascending.

        Also truncates any trailing (0.,0.) pairs as a results of data coming
        from a CGMS database.
        """
        x_list = tbl_xy[0::2]
        n = len(x_list)

        # Check if x range is ascending continuously
        rng = list(range(1, n))
        x_asc = [True if (x_list[i] > x_list[i - 1]) else False for i in rng]

        # Check for breaks in the series where the ascending sequence stops.
        # Only 0 or 1 breaks are allowed. Use the XOR operator '^' here
        sum_break = sum([1 if (x0 ^ x1) else 0 for x0, x1 in zip(x_asc, x_asc[1:], strict=False)])
        if sum_break == 0:
            indices = list(range(len(x_list)))
        elif sum_break == 1:
            indices = [0]
            for i, p in zip(rng, x_asc, strict=False):
                if p is True:
                    indices.append(i)
        else:
            msg = f"X values for AFGEN input list not strictly ascending: {x_list.tolist()}"
            raise ValueError(msg)

        return indices

    def __init__(self, tbl_xy):
        # Convert to tensor if needed
        tbl_xy = torch.as_tensor(tbl_xy, dtype=DTYPE)

        # Get valid indices
        indices = self._check_x_ascending(tbl_xy)

        # Extract x and y values using indices
        x_indices = torch.tensor([2 * i for i in indices])
        y_indices = torch.tensor([2 * i + 1 for i in indices])
        self.x_list = tbl_xy[x_indices]
        self.y_list = tbl_xy[y_indices]

        # Calculate slopes
        x1 = self.x_list[:-1]
        x2 = self.x_list[1:]
        y1 = self.y_list[:-1]
        y2 = self.y_list[1:]
        self.slopes = (y2 - y1) / (x2 - x1)

    def __call__(self, x):
        """Returns the interpolated value at abscissa x."""
        # Differentiable path using PyTorch
        x = torch.as_tensor(x, dtype=DTYPE)

        # Clamp to boundaries
        if x <= self.x_list[0]:
            return self.y_list[0]
        if x >= self.x_list[-1]:
            return self.y_list[-1]

        # Find interval index using torch.searchsorted for differentiability
        i = torch.searchsorted(self.x_list, x, right=False) - 1
        i = torch.clamp(i, 0, len(self.x_list) - 2)

        # Linear interpolation
        v = self.y_list[i] + self.slopes[i] * (x - self.x_list[i])
        return v


class AfgenTrait(TraitType):
    """An AFGEN table trait."""

    default_value = Afgen([0, 0, 1, 1])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj, value):
        """Validate that the value is an Afgen instance or an iterable to create one."""
        if isinstance(value, Afgen):
            return value
        elif isinstance(value, Iterable):
            return Afgen(value)
        self.error(obj, value)
