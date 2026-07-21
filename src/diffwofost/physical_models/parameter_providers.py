from pcse.base.parameter_providers import ParameterProvider as PcseParameterProvider


class ParameterProvider(PcseParameterProvider):
    """Temporary implementation of PCSE's ParameterProvider.

    Fixes the `__iter__` method in order to allow for access via dict-like `.items()`, `.keys()`,
    and `.values()`. Could be dropped when https://github.com/ajwdewit/pcse/pull/121 is merged and
    a new version of PCSE released.
    """

    def __iter__(self):
        return iter(self._unique_parameters)
