"""
resilience.py - circuit breaker backing the fail-open guarantee.

If the cache store (Redis) starts erroring or timing out, every request would
otherwise pay a connect timeout before falling back. After `threshold`
consecutive failures the breaker opens and requests skip the cache entirely
(fail open -> straight to the LLM provider) until `cooldown` seconds pass,
then one probe request is allowed through (half-open).
"""
import time


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 10.0, clock=time.time):
        self.threshold, self.cooldown, self._clock = threshold, cooldown, clock
        self.failures = 0
        self.opened_at = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if self._clock() - self.opened_at >= self.cooldown:
            return False  # half-open: let a probe through
        return True

    def allow(self) -> bool:
        return not self.is_open

    def success(self):
        self.failures, self.opened_at = 0, None

    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self._clock()
