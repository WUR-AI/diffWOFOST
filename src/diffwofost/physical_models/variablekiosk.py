from pcse.base.variablekiosk import VariableKiosk as _PcseVariableKiosk


class VariableKiosk(_PcseVariableKiosk):
    """Extends pcse's VariableKiosk with support for external dependencies.

    The external_state_list parameter accepts a list of per-day dicts, each
    containing a ``"DAY"`` key and the variable values to inject for that day.
    Calling the kiosk with a day (``kiosk(day)``) advances to the next entry
    and makes those variables available via normal attribute/item access.

    All original VariableKiosk behaviour (registering, publishing, flushing)
    is inherited unchanged from pcse.
    """

    def __init__(self, external_state_list=None):
        super().__init__()
        self.current_externals = {}
        self.external_state_list = list(external_state_list) if external_state_list else None
        self._external_state_index = 0

    def __call__(self, day):
        """Set the external state/rate variables for the current day.

        Advances to the next entry in the external state list and makes those
        values available via normal kiosk access. Does nothing if the list is
        exhausted or was not provided. Always returns False; use
        ``is_exhausted`` to check whether all external states have been consumed.
        """
        if self.external_state_list and self._external_state_index < len(self.external_state_list):
            current_externals = dict(self.external_state_list[self._external_state_index])
            forcing_day = current_externals.pop("DAY")
            msg = "Failure updating VariableKiosk with external states: days are not matching!"
            assert forcing_day == day, msg
            self.current_externals.clear()
            self.current_externals.update(current_externals)
            self._external_state_index += 1
        return False

    @property
    def is_exhausted(self):
        """True when all external states have been consumed."""
        if self.external_state_list is None:
            return False
        return self._external_state_index >= len(self.external_state_list)

    def is_external_state(self, item):
        """Returns True if the item is an external state."""
        return item in self.current_externals

    def __contains__(self, item):
        """Checks external states first, then the published kiosk variables."""
        return item in self.current_externals or dict.__contains__(self, item)

    def __getitem__(self, item):
        """Look in external states before falling back to published variables."""
        current_externals = self.__dict__.get("current_externals", {})
        if item in current_externals:
            return current_externals[item]
        return dict.__getitem__(self, item)

    def __getattr__(self, item):
        """Allow attribute notation (e.g. ``kiosk.LAI``), checking externals first."""
        current_externals = self.__dict__.get("current_externals", {})
        if item in current_externals:
            return current_externals[item]
        return dict.__getitem__(self, item)

    def flush_rates(self):
        """Flush the values of all published rate variable from the kiosk."""
        for key in self.published_rates.keys():
            self.pop(key, None)

    def flush_states(self):
        """Flush the values of all state variable from the kiosk."""
        for key in self.published_states.keys():
            self.pop(key, None)
