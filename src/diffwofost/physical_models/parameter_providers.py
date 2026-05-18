from pcse.base.parameter_providers import ParameterProvider as PcseParameterProvider


class ParameterProvider(PcseParameterProvider):
    """Temporary implementation of PCSE's ParameterProvider.

    Fixes the `__iter__` method in order to allow for access via dict-like `.items()`, `.keys()`,
    and `.values()`.
    """

    def __iter__(self):
        return iter(self._unique_parameters)
