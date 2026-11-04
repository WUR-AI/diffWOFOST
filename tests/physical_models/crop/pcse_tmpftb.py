"""Follow the TMPFTB driver of the installed PCSE.

Released PCSE 8.1 evaluates the WOFOST 8.1 temperature response at the daily
mean (``drv.TEMP``). WOFOST 7.2 already used the daytime temperature, and the
PCSE develop branch does too. diffWOFOST follows that correction. Season
comparisons patch the reference only while the installed package still has
the daily-mean call, so the same tests match a corrected PCSE without a change.
"""

import inspect
from contextlib import contextmanager


def pcse_wofost81_tmpftb_uses_daytime():
    """True when this PCSE passes ``drv.DTEMP`` into WOFOST 8.1 ``TMPFTB``."""
    import pcse.crop.assimilation as assimilation

    source = inspect.getsource(assimilation.WOFOST81_Assimilation.__call__)
    daytime = "TMPFTB(drv.DTEMP)" in source
    daily_mean = "TMPFTB(drv.TEMP)" in source
    if daytime == daily_mean:
        raise RuntimeError("PCSE WOFOST 8.1 TMPFTB driver was not recognized")
    return daytime


class _MeanAsDaytime:
    """Weather whose daily mean reads as the daytime temperature.

    ``WOFOST81_Assimilation`` uses ``drv.TEMP`` only for ``TMPFTB``. Light
    use efficiency already reads ``drv.DTEMP``.
    """

    def __init__(self, drv):
        object.__setattr__(self, "_drv", drv)

    def __getattr__(self, name):
        drv = object.__getattribute__(self, "_drv")
        if name == "TEMP":
            return drv.DTEMP
        return getattr(drv, name)


@contextmanager
def pcse_reference_matching_daytime_tmpftb():
    """Run PCSE with the same TMPFTB driver as diffWOFOST."""
    if pcse_wofost81_tmpftb_uses_daytime():
        yield
        return

    import pcse.crop.assimilation as assimilation

    cls = assimilation.WOFOST81_Assimilation
    original = cls.__call__

    def call(self, day, drv):
        return original(self, day, _MeanAsDaytime(drv))

    cls.__call__ = call
    try:
        yield
    finally:
        cls.__call__ = original
