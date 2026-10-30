"""
budget.py - dynamic token budgeting & throttling.

Cache hits cost nothing, so only LIVE provider calls are metered. Per tenant:
  * rpm                      hard requests-per-minute limit on live calls
  * daily_tokens / daily_usd / monthly_usd   spend caps
  * soft threshold (default 80% of any cap): the tenant is *throttled* -
    live-call RPM is cut to `throttle_factor` of its normal limit, so a
    consumer racing toward its cap is slowed before it is cut off.
  * hard cap (100%): live calls are denied with 429 + Retry-After until the
    window resets. Cache hits are still served (configurable), which is the
    financial incentive for good cache behaviour.
Counters live in Redis (shared across gateway replicas); the same code runs
against MemoryStore in tests.
"""
import time
from dataclasses import dataclass
from typing import Optional

from .tenants import Tenant


@dataclass
class Decision:
    allowed: bool
    state: str                      # "ok" | "throttled" | "denied"
    reason: str = ""
    retry_after: int = 0


def _day(now: float) -> str:
    return time.strftime("%Y%m%d", time.gmtime(now))


def _month(now: float) -> str:
    return time.strftime("%Y%m", time.gmtime(now))


def _secs_to_next_day(now: float) -> int:
    t = time.gmtime(now)
    return 86400 - (t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec)


def _secs_to_next_month(now: float) -> int:
    t = time.gmtime(now)
    nm = (t.tm_year + (t.tm_mon // 12), t.tm_mon % 12 + 1)
    import calendar
    nxt = calendar.timegm((nm[0], nm[1], 1, 0, 0, 0))
    return max(1, int(nxt - now))


class BudgetManager:
    def __init__(self, store, soft_threshold: float = 0.8, throttle_factor: float = 0.25,
                 clock=time.time):
        self.r, self.soft, self.factor, self._clock = store, soft_threshold, throttle_factor, clock

    # ---- keys ----
    def _rpm_key(self, t: Tenant, now) -> str:
        return f"bud:rpm:{t.tenant}:{int(now // 60)}"

    def _day_key(self, t, now) -> str:
        return f"bud:day:{t.tenant}:{_day(now)}"

    def _month_key(self, t, now) -> str:
        return f"bud:mon:{t.tenant}:{_month(now)}"

    def usage(self, t: Tenant) -> dict:
        now = self._clock()
        d = self.r.hgetall(self._day_key(t, now))
        m = self.r.hgetall(self._month_key(t, now))
        return {
            "day_tokens": int(float(d.get("tokens", 0))), "day_usd": float(d.get("usd", 0)),
            "month_usd": float(m.get("usd", 0)),
            "rpm_used": int(self.r.get(self._rpm_key(t, now)) or 0),
        }

    def check(self, t: Tenant) -> Decision:
        """Call BEFORE a live provider call. Counts the request against RPM."""
        now = self._clock()
        u = self.usage(t)

        caps = [
            ("daily_tokens", u["day_tokens"], t.daily_tokens, _secs_to_next_day(now)),
            ("daily_usd", u["day_usd"], t.daily_usd, _secs_to_next_day(now)),
            ("monthly_usd", u["month_usd"], t.monthly_usd, _secs_to_next_month(now)),
        ]
        near = None
        for name, used, cap, reset in caps:
            if cap is None:
                continue
            if used >= cap:
                return Decision(False, "denied", f"{name} cap reached ({used:.4g}/{cap})", reset)
            if used >= cap * self.soft and near is None:
                near = f"{name} at {used / cap:.0%} of cap"

        limit = t.rpm
        state = "ok"
        if near and limit is not None:
            limit = max(1, int(limit * self.factor))
            state = "throttled"
        elif near:                    # no rpm configured: throttle to a sane floor
            limit, state = max(1, int(60 * self.factor)), "throttled"

        if limit is not None:
            key = self._rpm_key(t, now)
            n = self.r.incr(key)
            if n == 1:
                self.r.expire(key, 70)
            if n > limit:
                retry = 60 - int(now % 60)
                return Decision(False, "denied" if state == "ok" else "throttled",
                                f"rate limit {limit}/min" + (f" ({near})" if near else ""), retry)
        return Decision(True, state, near or "")

    def record(self, t: Tenant, tokens: int, usd: float):
        """Call AFTER a live call completes, with real usage."""
        now = self._clock()
        dk, mk = self._day_key(t, now), self._month_key(t, now)
        self.r.hincrbyfloat(dk, "tokens", tokens)
        self.r.hincrbyfloat(dk, "usd", usd)
        self.r.hincrbyfloat(mk, "usd", usd)
        self.r.expire(dk, 2 * 86400)
        self.r.expire(mk, 40 * 86400)
