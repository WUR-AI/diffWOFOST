"""Device-side checks that are evaluated together, once per simulated day.

Testing a tensor in a Python ``if`` makes the host wait for the device. The
model components therefore hand such conditions to the active
:class:`DeferredQueue` instead; the engine evaluates all of them with a single
host-device synchronisation at the end of each simulated day and then runs the
callbacks of the conditions that hold (raise an error, send a signal, ...).

Outside an engine run (e.g. a component used on its own) there is no active
queue and conditions are evaluated immediately.
"""

from contextlib import contextmanager
import torch

_active = None


class DeferredValue:
    """Boolean value of a condition tensor, read from the device when needed."""

    __slots__ = ("_queue", "_tensor", "_value")

    def __init__(self, queue, tensor, value=None):
        self._queue = queue
        self._tensor = tensor
        self._value = value

    def get(self) -> bool:
        """Return the value; evaluates the pending conditions if necessary."""
        if self._value is None:
            self._queue.flush()
        return self._value


class DeferredQueue:
    """Conditions waiting to be evaluated in one synchronisation."""

    MAX_PENDING = 10_000  # evaluate anyway when nobody flushes the queue

    def __init__(self):
        self._pending = []

    def defer(self, condition: torch.Tensor, callback=None) -> DeferredValue:
        """Queue `condition` (any boolean tensor, reduced with all()).

        `callback` is called without arguments when the condition is True.
        """
        value = DeferredValue(self, torch.all(condition.detach()))
        self._pending.append((value, callback))
        if len(self._pending) >= self.MAX_PENDING:
            self.flush()
        return value

    def flush(self):
        """Evaluate all queued conditions and run the callbacks of those that hold."""
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        flags = torch.stack([v._tensor for v, _ in pending]).tolist()
        for (value, _), flag in zip(pending, flags, strict=True):
            value._value = bool(flag)
            value._tensor = None
        for (_, callback), flag in zip(pending, flags, strict=True):
            if flag and callback is not None:
                callback()

    @contextmanager
    def active(self):
        """Make this the queue used by `defer` and flush it on exit."""
        global _active
        previous, _active = _active, self
        try:
            yield self
        finally:
            _active = previous
        self.flush()


def defer(condition: torch.Tensor, callback=None) -> DeferredValue:
    """Queue `condition` on the active queue, or evaluate it now if there is none."""
    if _active is not None:
        return _active.defer(condition, callback)
    value = bool(torch.all(condition))
    if value and callback is not None:
        callback()
    return DeferredValue(None, None, value)
