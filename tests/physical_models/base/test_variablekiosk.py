import datetime
import pytest
from pcse.base.variablekiosk import VariableKiosk as PcseVariableKiosk
from diffwofost.physical_models.variablekiosk import VariableKiosk

DAY1 = datetime.date(2000, 1, 1)
DAY2 = datetime.date(2000, 1, 2)
DAY3 = datetime.date(2000, 1, 3)


def _make_external_states():
    return [
        {"DAY": DAY1, "LAI": 0.5, "DVS": 0.1},
        {"DAY": DAY2, "LAI": 1.0, "DVS": 0.2},
        {"DAY": DAY3, "LAI": 1.5, "DVS": 0.3},
    ]


@pytest.mark.usefixtures("fast_mode")
class TestVariableKioskIsSubclassOfPcse:
    def test_is_instance_of_pcse_variablekiosk(self):
        """Must satisfy the pcse Instance(VariableKiosk) trait used in BaseEngine."""
        kiosk = VariableKiosk()
        assert isinstance(kiosk, PcseVariableKiosk)


@pytest.mark.usefixtures("fast_mode")
class TestVariableKioskInit:
    def test_init_without_external_states(self):
        kiosk = VariableKiosk()
        assert kiosk.current_externals == {}
        assert kiosk.external_state_list is None

    def test_init_with_external_states_stores_copy(self):
        ext = _make_external_states()
        kiosk = VariableKiosk(ext)
        assert kiosk.external_state_list is not None
        assert len(kiosk.external_state_list) == 3

    def test_init_makes_independent_copy_of_list(self):
        ext = _make_external_states()
        kiosk = VariableKiosk(ext)
        ext.clear()
        assert len(kiosk.external_state_list) == 3


@pytest.mark.usefixtures("fast_mode")
class TestVariableKioskCall:
    def test_call_populates_current_externals(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert kiosk.current_externals == {"LAI": 0.5, "DVS": 0.1}

    def test_call_advances_on_each_day(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        kiosk(DAY2)
        assert kiosk.current_externals == {"LAI": 1.0, "DVS": 0.2}

    def test_call_always_returns_false(self):
        kiosk = VariableKiosk(_make_external_states())
        assert kiosk(DAY1) is False
        assert kiosk(DAY2) is False
        assert kiosk(DAY3) is False

    def test_call_returns_false_without_external_list(self):
        kiosk = VariableKiosk()
        assert kiosk(DAY1) is False

    def test_call_raises_on_day_mismatch(self):
        """A day not present in the list simply clears externals — no error."""
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        missing_day = datetime.date(1999, 6, 15)
        kiosk(missing_day)  # should not raise
        assert kiosk.current_externals == {}

    def test_external_states_exhausted_false_before_last_entry(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert kiosk.external_states_exhausted is False

    def test_external_states_exhausted_true_after_last_entry(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        kiosk(DAY2)
        kiosk(DAY3)
        assert kiosk.external_states_exhausted is True

    def test_external_states_exhausted_false_without_external_list(self):
        kiosk = VariableKiosk()
        assert kiosk.external_states_exhausted is False

    def test_original_list_is_not_modified(self):
        ext = _make_external_states()
        kiosk = VariableKiosk(ext)
        kiosk(DAY1)
        kiosk(DAY2)
        kiosk(DAY3)
        # The stored list must be intact even after full consumption
        assert len(kiosk.external_state_list) == 3
        assert all("DAY" in entry for entry in kiosk.external_state_list)

    def test_sparse_external_states_injects_on_matching_days(self):
        """Only days present in the list inject externals; gaps clear current_externals."""
        ext = [
            {"DAY": DAY1, "LAI": 0.5},
            {"DAY": DAY3, "LAI": 1.5},  # DAY2 intentionally absent
        ]
        kiosk = VariableKiosk(ext)
        kiosk(DAY1)
        assert kiosk.current_externals == {"LAI": 0.5}
        kiosk(DAY2)  # no entry → externals cleared
        assert kiosk.current_externals == {}
        kiosk(DAY3)
        assert kiosk.current_externals == {"LAI": 1.5}

    def test_external_states_exhausted_false_on_gap_day_before_last_entry(self):
        """A gap day before the last entry does not signal finished."""
        ext = [
            {"DAY": DAY1, "LAI": 0.5},
            {"DAY": DAY3, "LAI": 1.5},
        ]
        kiosk = VariableKiosk(ext)
        kiosk(DAY2)  # gap day, max is DAY3
        assert kiosk.external_states_exhausted is False


@pytest.mark.usefixtures("fast_mode")
class TestVariableKioskExternalAccess:
    def test_getitem_returns_external_variable(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert kiosk["LAI"] == 0.5

    def test_getattr_returns_external_variable(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert kiosk.LAI == 0.5

    def test_contains_finds_external_variable(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert "LAI" in kiosk

    def test_is_external_state_returns_true_for_external(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert kiosk.is_external_state("LAI") is True

    def test_is_external_state_returns_false_for_non_external(self):
        kiosk = VariableKiosk(_make_external_states())
        kiosk(DAY1)
        assert kiosk.is_external_state("NONEXISTENT") is False

    def test_is_external_state_returns_false_before_first_call(self):
        kiosk = VariableKiosk(_make_external_states())
        assert kiosk.is_external_state("LAI") is False

    def test_external_shadows_published_variable(self):
        """An external state with the same name as a published variable takes precedence."""
        kiosk = VariableKiosk(_make_external_states())
        oid = 42
        kiosk.register_variable(oid, "LAI", type="S", publish=True)
        kiosk.set_variable(oid, "LAI", 99.0)
        assert kiosk["LAI"] == 99.0  # before external update: published value
        kiosk(DAY1)
        assert kiosk["LAI"] == 0.5  # after: external overrides
        assert kiosk.LAI == 0.5

    def test_externals_cleared_between_days(self):
        """current_externals only holds variables from the most recent day."""
        ext = [
            {"DAY": DAY1, "LAI": 0.5},
            {"DAY": DAY2, "DVS": 0.2},  # LAI absent on day 2
        ]
        kiosk = VariableKiosk(ext)
        kiosk(DAY1)
        assert "LAI" in kiosk.current_externals
        kiosk(DAY2)
        assert "LAI" not in kiosk.current_externals
        assert "DVS" in kiosk.current_externals


@pytest.mark.usefixtures("fast_mode")
class TestVariableKioskInheritedBehaviour:
    def test_register_and_set_published_variable(self):
        kiosk = VariableKiosk()
        oid = 1
        kiosk.register_variable(oid, "DVS", type="S", publish=True)
        kiosk.set_variable(oid, "DVS", 1.0)
        assert kiosk["DVS"] == 1.0

    def test_flush_rates_clears_published_rates(self):
        kiosk = VariableKiosk()
        oid = 1
        kiosk.register_variable(oid, "DVR", type="R", publish=True)
        kiosk.set_variable(oid, "DVR", 0.05)
        kiosk.flush_rates()
        assert "DVR" not in kiosk

    def test_flush_states_clears_published_states(self):
        kiosk = VariableKiosk()
        oid = 1
        kiosk.register_variable(oid, "DVS", type="S", publish=True)
        kiosk.set_variable(oid, "DVS", 0.5)
        kiosk.flush_states()
        assert "DVS" not in kiosk

    def test_variable_exists(self):
        kiosk = VariableKiosk()
        oid = 1
        kiosk.register_variable(oid, "DVS", type="S")
        assert kiosk.variable_exists("DVS") is True
        assert kiosk.variable_exists("LAI") is False
