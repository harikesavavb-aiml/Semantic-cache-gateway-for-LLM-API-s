"""Shared fakes for gateway tests: no network, no Redis, no ML models."""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_logic import DEFAULT_POLICIES
from gateway.adapters import CanonicalRequest
from gateway.budget import BudgetManager
from gateway.engine import Engine, EngineConfig
from gateway.index import CacheIndex
from gateway.memstore import MemoryStore
from gateway.pricing import DEFAULT_PRICES
from gateway.providers import BaseProvider, ProviderError, ProviderResult, ProviderRouter, ProviderStream
from gateway.resilience import CircuitBreaker
from gateway.telemetry import Telemetry
from gateway.tenants import Tenant


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def fake_embed(text: str):
    """Bag-of-words hashing embedding: texts sharing words are similar."""
    v = np.zeros(64, dtype=np.float32)
    for w in text.lower().replace("?", "").replace(",", "").split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeProvider(BaseProvider):
    name, dialect = "fake", "openai"

    def __init__(self, dialect="openai"):
        self.dialect, self.calls, self.fail = dialect, 0, None
        self.answer = lambda c: f"ANSWER[{c.last_user_message}]"

    def complete(self, c):
        self.calls += 1
        if self.fail:
            raise self.fail
        t = self.answer(c)
        return ProviderResult(t, 10, max(1, len(t) // 4))

    def stream(self, c):
        self.calls += 1
        text = self.answer(c)
        words = text.split(" ")
        ps = ProviderStream(iter([w + " " for w in words[:-1]] + [words[-1]]))
        ps.usage = {"in": 10, "out": len(words)}
        return ps

    def passthrough(self, c):
        self.calls += 1
        return {"id": "x", "usage": {"prompt_tokens": 5, "completion_tokens": 7}, "tool_calls": True}


class FakeRerank:
    def __init__(self, score=0.95):
        self.score, self.calls = score, 0

    def __call__(self, pairs):
        self.calls += 1
        return [self.score for _ in pairs]


def make_engine(policies=None, verifier=None, rerank=None, store=None, clock=None, cfg_kw=None):
    clock = clock or Clock()
    store = store or MemoryStore(clock=clock)
    prov = FakeProvider()
    router = ProviderRouter()
    router.add(["m"], prov, default=True)
    pol = dict(DEFAULT_POLICIES)
    pol.update(policies or {})
    tm = Telemetry(store, clock=clock, embed_cost_per_1k=0.01, vector_db_hourly_usd=0.0)
    eng = Engine(cfg=EngineConfig(policies=pol, prices=DEFAULT_PRICES, default_model="m", **(cfg_kw or {})),
                 store=store, index=CacheIndex(store, clock=clock), embedder=fake_embed, reranker=rerank,
                 router=router, telemetry=tm, budget=BudgetManager(store, clock=clock),
                 verifier=verifier, breaker=CircuitBreaker(clock=clock), clock=clock)
    eng.prov, eng.store, eng.tm, eng.clock = prov, store, tm, clock
    return eng


def req(msg, *, tenant="acme", user=None, tag="default", feature="f", model="m", system="sys",
        tenant_obj=None, stream=False, tools=None, dialect="native", messages=None, unit="u1"):
    return CanonicalRequest(
        dialect=dialect, model=model, system=system,
        messages=messages or [{"role": "user", "content": msg}], temperature=0.0, stream=stream,
        feature=feature, tag=tag, user_id=user, tenant_id=tenant, tools=tools,
        extra={"tenant_obj": tenant_obj, "unit": unit, "key_id": "k1"})
