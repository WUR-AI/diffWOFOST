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
        self._last_called_day = None
        # Build a day-keyed dict for O(1) lookup, preserving the original list untouched
        self._external_states = {}
        if self.external_state_list:
            for entry in self.external_state_list:
                entry_copy = dict(entry)
                day = entry_copy.pop("DAY")
                self._external_states[day] = entry_copy

    def __call__(self, day):
        """Set the external state/rate variables for the current day.

        If the day has an entry in the external state list, its values are
        injected into ``current_externals``. If the day has no entry,
        ``current_externals`` is cleared so the module falls back to normally
        registered kiosk variables. Does nothing when no list was provided.
        Always returns False; use ``external_states_exhausted`` to check whether
        the last entry has been passed.
        """
        self._last_called_day = day
        if self._external_states:
            self.current_externals.clear()
            if day in self._external_states:
                self.current_externals.update(self._external_states[day])
        return False

    @property
    def external_states_exhausted(self):
        """True when the simulation has advanced past the last external state entry."""
        if not self._external_states or self._last_called_day is None:
            return False
        return self._last_called_day >= max(self._external_states.keys())

    def is_external_state(self, item):
        """Returns True if the item is an external state."""
        return item in self.current_externals

    def __contains__(self, item):
        """Checks external states first, then the published kiosk variables."""
        current_externals = self.__dict__.get("current_externals", {})
        return item in current_externals or dict.__contains__(self, item)

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
