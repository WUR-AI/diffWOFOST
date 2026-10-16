import torch
from traitlets_pcse import All
from traitlets_pcse import TraitType
from traitlets_pcse import Undefined
from diffwofost.physical_models.config import ComputeConfig


class Tensor(TraitType):
    info_text = "an object that could be casted into a tensor"

    def __init__(
        self,
        default_value=Undefined,
        allow_none=False,
        read_only=None,
        help=None,
        config=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            default_value=default_value,
            allow_none=allow_none,
            read_only=read_only,
            help=help,
            config=config,
            **kwargs,
        )
        self.dtype = dtype

    def validate(self, obj, value):
        """Validate input object, recasting it into a tensor if possible."""
        device = ComputeConfig.get_device()
        dtype = ComputeConfig.get_dtype() if self.dtype is None else self.dtype
        if isinstance(value, torch.Tensor):
            casted = value.to(dtype=dtype, device=device)
            return casted
        try:
            # Try casting value into a tensor, raise validation error if it fails
            return torch.tensor(value, dtype=dtype, device=device)
        except:  # noqa: E722
            self.error(obj, value)

    def set(self, obj, value):
        """Set the value without comparing it to the old one.

        The base implementation evaluates ``bool(old_value == new_value)`` to skip
        'change' notifications. For tensors that launches a comparison kernel and,
        for one-element tensors, waits for the device; for larger tensors it raises
        and is caught. Tensors are therefore always treated as changed.
        """
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        for name in [All, self.name]:
            for change_type in obj._trait_notifiers.get(name, {}):
                if change_type == All or change_type == "change":
                    obj._notify_trait(self.name, old_value, new_value, change_type)

    def from_string(self, s):
        """Casting tensor from string is not supported for now."""
        raise NotImplementedError
