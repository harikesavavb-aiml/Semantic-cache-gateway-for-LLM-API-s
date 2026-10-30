"""
telemetry.py - ROI, latency, quality and reliability metrics (spec sections 1-4).

Everything is stored in Redis hashes/lists so all gateway replicas aggregate
into one picture, and read back through `report()`. Pure Python; runs against
MemoryStore in tests.

Counter hashes:  tm:global, tm:tenant:<t>, tm:unit:<u>, tm:feature:<f>, tm:key:<key_id>
Sample lists  :  tm:lat:{cached,live}, tm:lookup, tm:ttft:{cached,live}   (capped)
"""
import functools
import json
import logging
import math
import time
from typing import Dict, List, Optional, Sequence

log = logging.getLogger("gateway.telemetry")
SAMPLE_CAP = 5000
HIT_OUTCOMES = ("exact_hit", "semantic_hit", "semantic_verified")


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Nearest-rank percentile; None for empty input."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return s[k]


def jaccard(a: str, b: str) -> float:
    """Token-set overlap used as a cheap answer-similarity proxy."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def threshold_drift(near_misses: List[dict], min_samples: int = 20) -> dict:
    """
    Threshold Drift Coefficient from the near-miss log (newest first).
    Splits the window into an older and a recent half and compares them:
      drift        = mean(sim recent) - mean(sim older)
      verified_gap = verified-pass rate recent - older
    Positive drift with a rising verified rate means users are phrasing the
    same intents closer to cached wording or the threshold is too strict;
    negative means intent expressions are diverging from cached prompts.
    Only advisory: it never changes a threshold automatically.
    """
    if len(near_misses) < min_samples:
        return {"samples": len(near_misses), "drift": None,
                "note": f"need >= {min_samples} near-misses"}
    chron = list(reversed(near_misses))                  # oldest -> newest
    half = len(chron) // 2
    old, new = chron[:half], chron[half:]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def vr(xs):
        return mean([1.0 if e.get("verified") else 0.0 for e in xs])

    drift = mean([e["similarity"] for e in new]) - mean([e["similarity"] for e in old])
    gap = vr(new) - vr(old)
    advice = "stable"
    if vr(near_misses) > 0.85:
        advice = "most near-misses pass verification: consider lowering auto_hit"
    elif vr(near_misses) < 0.15:
        advice = "few near-misses pass verification: borderline_low may be too permissive"
    return {"samples": len(near_misses), "drift": round(drift, 4),
            "verified_rate_gap": round(gap, 4), "advice": advice}


class Telemetry:
    def __init__(self, store, clock=time.time, embed_cost_per_1k: float = 0.0,
                 vector_db_hourly_usd: float = 0.0, lookup_slo_ms: float = 20.0):
        self.r = store
        self._clock = clock
        self.embed_cost_per_1k = embed_cost_per_1k
        self.vector_db_hourly_usd = vector_db_hourly_usd
        self.lookup_slo_ms = lookup_slo_ms
        try:
            if self.r.hget("tm:meta", "started") is None:
                self.r.hset("tm:meta", "started", self._clock())
        except Exception as e:          # store down at boot must not stop the gateway
            log.warning("telemetry init: store unavailable (%s)", e)

    # ------------------------------ write side ------------------------------

    def _incr(self, keys: List[str], **fields):
        for k in keys:
            for f, v in fields.items():
                if v:
                    if isinstance(v, float):
                        self.r.hincrbyfloat(k, f, v)
                    else:
                        self.r.hincrby(k, f, v)

    def _sample(self, key: str, value: Optional[float]):
        if value is None:
            return
        self.r.lpush(key, repr(round(float(value), 3)))
        self.r.ltrim(key, 0, SAMPLE_CAP - 1)

    def record_request(self, *, outcome: str, tenant: str, unit: str, key_id: str,
                       feature: str, model: str, total_ms: float,
                       lookup_ms: Optional[float] = None, ttft_ms: Optional[float] = None,
                       streamed: bool = False, saved_in: int = 0, saved_out: int = 0,
                       saved_usd: float = 0.0, spent_in: int = 0, spent_out: int = 0,
                       spent_usd: float = 0.0, embedded: bool = False, age_ratio: Optional[float] = None):
        keys = ["tm:global", f"tm:tenant:{tenant}", f"tm:unit:{unit}",
                f"tm:feature:{feature}", f"tm:key:{key_id}"]
        is_hit = outcome in HIT_OUTCOMES
        self._incr(keys, requests=1, **{f"o_{outcome}": 1},
                   hits=1 if is_hit else 0,
                   saved_in_tokens=saved_in, saved_out_tokens=saved_out, saved_usd=float(saved_usd),
                   spent_in_tokens=spent_in, spent_out_tokens=spent_out, spent_usd=float(spent_usd),
                   embeddings=1 if embedded else 0,
                   streamed=1 if streamed else 0)
        bucket = "cached" if is_hit else "live"
        self._sample(f"tm:lat:{bucket}", total_ms)
        self._sample("tm:lookup", lookup_ms)
        if streamed:
            self._sample(f"tm:ttft:{bucket}", ttft_ms)
        if is_hit and age_ratio is not None:
            self.r.hincrbyfloat("tm:decay", "age_ratio_sum", float(age_ratio))
            self.r.hincrby("tm:decay", "hits", 1)
            self.r.hincrbyfloat(f"tm:decay:{feature}", "age_ratio_sum", float(age_ratio))
            self.r.hincrby(f"tm:decay:{feature}", "hits", 1)
        if lookup_ms is not None and lookup_ms > self.lookup_slo_ms:
            self.r.hincrby("tm:global", "lookup_slo_breaches", 1)

    def fail_open(self, reason: str = ""):
        self.r.hincrby("tm:global", "fail_open", 1)
        self.r.lpush("tm:failopen:log", json.dumps({"t": self._clock(), "reason": reason[:200]}))
        self.r.ltrim("tm:failopen:log", 0, 99)

    def isolation_block(self, tenant: str, detail: str):
        self.r.hincrby("tm:global", "isolation_blocks", 1)
        self.r.hincrby(f"tm:tenant:{tenant}", "isolation_blocks", 1)
        self.r.lpush("tm:isolation:log", json.dumps({"t": self._clock(), "tenant": tenant, "detail": detail[:200]}))
        self.r.ltrim("tm:isolation:log", 0, 99)

    def budget_event(self, tenant: str, state: str):
        self.r.hincrby(f"tm:tenant:{tenant}", f"budget_{state}", 1)
        self.r.hincrby("tm:global", f"budget_{state}", 1)

    def near_miss(self, similarity: float, verified: bool, feature: str):
        self.r.lpush("tm:nearmiss", json.dumps(
            {"t": self._clock(), "similarity": float(similarity), "verified": bool(verified), "feature": feature}))
        self.r.ltrim("tm:nearmiss", 0, 999)

    # quality feedback -------------------------------------------------------
    def remember_hit(self, request_id: str, info: dict, ttl: int = 7 * 86400):
        self.r.set(f"tm:hit:{request_id}", json.dumps(info), ex=ttl)

    def feedback(self, request_id: str, bad: bool, source: str = "human") -> bool:
        raw = self.r.get(f"tm:hit:{request_id}")
        if raw is None:
            return False
        if self.r.hget(f"tm:fb:{request_id}", "done"):
            return True                                   # idempotent
        self.r.hset(f"tm:fb:{request_id}", "done", 1)
        self.r.expire(f"tm:fb:{request_id}", 7 * 86400)
        self.r.hincrby("tm:quality", f"{source}_judged", 1)
        if bad:
            self.r.hincrby("tm:quality", f"{source}_bad", 1)
            self.r.hincrby("tm:quality", "bad_total", 1)
            info = json.loads(raw)
            self.r.hincrby(f"tm:quality:{info.get('feature', 'default')}", "bad", 1)
        return True

    def revalidation(self, feature: str, cached: str, fresh: str):
        """Sampled freshness probe: compare a cached answer to a fresh live
        answer to the same prompt. divergence = 1 - overlap."""
        div = 1.0 - jaccard(cached, fresh)
        for k in ("tm:diverge", f"tm:diverge:{feature}"):
            self.r.hincrbyfloat(k, "sum", div)
            self.r.hincrby(k, "n", 1)
        return div

    # ------------------------------ read side -------------------------------

    @staticmethod
    def _f(h: Dict[str, str], k: str, cast=float):
        try:
            return cast(float(h.get(k, 0)))
        except (TypeError, ValueError):
            return cast(0)

    def _group(self, h: Dict[str, str]) -> dict:
        f = lambda k: self._f(h, k)                       # noqa: E731
        i = lambda k: self._f(h, k, int)                  # noqa: E731
        req = i("requests")
        return {
            "requests": req, "hits": i("hits"),
            "hit_rate": round(i("hits") / req, 4) if req else 0.0,
            "exact_hits": i("o_exact_hit"),
            "semantic_hits": i("o_semantic_hit") + i("o_semantic_verified"),
            "misses": i("o_miss"), "bypass": i("o_bypass"), "fail_open": i("o_fail_open"),
            "saved_input_tokens": i("saved_in_tokens"), "saved_output_tokens": i("saved_out_tokens"),
            "saved_usd": round(f("saved_usd"), 6), "spent_usd": round(f("spent_usd"), 6),
            "embeddings": i("embeddings"), "isolation_blocks": i("isolation_blocks"),
            "budget_throttled": i("budget_throttled"), "budget_denied": i("budget_denied"),
        }

    def _floats(self, key: str) -> List[float]:
        return [float(x) for x in self.r.lrange(key, 0, -1)]

    def _scan_groups(self, prefix: str) -> Dict[str, dict]:
        out = {}
        for k in self.r.scan_iter(match=f"{prefix}*"):
            out[k[len(prefix):]] = self._group(self.r.hgetall(k))
        return out

    def latency_block(self) -> dict:
        def stats(vals):
            return {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95),
                    "p99": percentile(vals, 99)}
        cached, live = self._floats("tm:lat:cached"), self._floats("tm:lat:live")
        lookup = self._floats("tm:lookup")
        g = self.r.hgetall("tm:global")
        p95c, p95l = percentile(cached, 95), percentile(live, 95)
        p99c, p99l = percentile(cached, 99), percentile(live, 99)
        return {
            "cached_ms": stats(cached), "live_ms": stats(live),
            "p95_reduction_ms": round(p95l - p95c, 2) if p95c is not None and p95l is not None else None,
            "p99_reduction_ms": round(p99l - p99c, 2) if p99c is not None and p99l is not None else None,
            "lookup_overhead_ms": {**stats(lookup), "slo_ms": self.lookup_slo_ms,
                                   "slo_breaches": self._f(g, "lookup_slo_breaches", int),
                                   "slo_met_p95": (percentile(lookup, 95) or 0) <= self.lookup_slo_ms},
            "ttft_cached_ms": stats(self._floats("tm:ttft:cached")),
            "ttft_live_ms": stats(self._floats("tm:ttft:live")),
        }

    def report(self) -> dict:
        g = self.r.hgetall("tm:global")
        glob = self._group(g)
        started = float(self.r.hget("tm:meta", "started") or self._clock())
        hours = max((self._clock() - started) / 3600, 1e-9)
        embed_cost = glob["embeddings"] / 1000 * self.embed_cost_per_1k
        db_cost = hours * self.vector_db_hourly_usd
        net = glob["saved_usd"] - (embed_cost + db_cost)

        q = self.r.hgetall("tm:quality")
        judged = self._f(q, "human_judged", int) + self._f(q, "judge_judged", int)
        bad = self._f(q, "bad_total", int)
        decay = self.r.hgetall("tm:decay")
        dv = self.r.hgetall("tm:diverge")
        nm = [json.loads(x) for x in self.r.lrange("tm:nearmiss", 0, -1)]
        return {
            "overview": glob,
            "cost": {
                "saved_usd": glob["saved_usd"], "spent_usd": glob["spent_usd"],
                "saved_input_tokens": glob["saved_input_tokens"],
                "saved_output_tokens": glob["saved_output_tokens"],
                "embedding_cost_usd": round(embed_cost, 6), "vector_db_cost_usd": round(db_cost, 6),
                "net_roi_usd": round(net, 6), "uptime_hours": round(hours, 3),
            },
            "latency": self.latency_block(),
            "quality": {
                "judged_hits": judged, "bad_hits": bad,
                "false_positive_rate": round(bad / judged, 4) if judged else None,
                "human_bad": self._f(q, "human_bad", int), "judge_bad": self._f(q, "judge_bad", int),
                "threshold_drift": threshold_drift(nm),
                "semantic_divergence": {
                    "revalidation_samples": self._f(dv, "n", int),
                    "index": round(self._f(dv, "sum") / self._f(dv, "n"), 4) if self._f(dv, "n") else None,
                    "mean_age_ratio_at_hit": round(self._f(decay, "age_ratio_sum") / self._f(decay, "hits"), 4)
                    if self._f(decay, "hits") else None,
                },
            },
            "reliability": {
                "fail_open_count": self._f(g, "fail_open", int),
                "isolation_blocks": self._f(g, "isolation_blocks", int),
                "budget_throttled": self._f(g, "budget_throttled", int),
                "budget_denied": self._f(g, "budget_denied", int),
                "recent_fail_open": [json.loads(x) for x in self.r.lrange("tm:failopen:log", 0, 9)],
                "recent_isolation_blocks": [json.loads(x) for x in self.r.lrange("tm:isolation:log", 0, 9)],
            },
            "by_tenant": self._scan_groups("tm:tenant:"),
            "by_unit": self._scan_groups("tm:unit:"),
            "by_feature": self._scan_groups("tm:feature:"),
        }

    def chargeback(self) -> List[dict]:
        """One row per tenant/unit: live spend and cache savings, for
        internal business-unit chargebacks."""
        rows = []
        for k in self.r.scan_iter(match="tm:unit:*"):
            g = self._group(self.r.hgetall(k))
            rows.append({"unit": k[len("tm:unit:"):], "live_spend_usd": g["spent_usd"],
                         "saved_usd": g["saved_usd"], "requests": g["requests"], "hit_rate": g["hit_rate"]})
        return sorted(rows, key=lambda x: -x["live_spend_usd"])


def _fail_safe(default=None):
    """Telemetry is observability, never a dependency: a store error while
    recording must not fail (or slow) the request being measured."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:
                log.warning("telemetry %s skipped: %s", fn.__name__, e)
                return default
        return wrapper
    return deco


for _n, _d in [("record_request", None), ("fail_open", None), ("isolation_block", None),
               ("budget_event", None), ("near_miss", None), ("remember_hit", None),
               ("revalidation", None), ("feedback", False), ("_incr", None)]:
    setattr(Telemetry, _n, _fail_safe(_d)(getattr(Telemetry, _n)))
