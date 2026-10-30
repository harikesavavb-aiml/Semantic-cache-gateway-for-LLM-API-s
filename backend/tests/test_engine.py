"""End-to-end pipeline tests with fakes: caching, isolation, fail-open, streaming, budgets."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import *  # noqa
from gateway.engine import GatewayError
from gateway.providers import DialectMismatch


# ------------------------------- caching -------------------------------------

def test_miss_then_exact_hit_then_semantic_hit():
    e = make_engine()
    r1 = e.handle(req("how do I reset my password"))
    assert r1.outcome == "miss" and e.prov.calls == 1
    r2 = e.handle(req("how do I reset my password"))
    assert r2.outcome == "exact_hit" and r2.text == r1.text and e.prov.calls == 1
    r3 = e.handle(req("reset my password how do I"))          # different string, same meaning
    assert r3.outcome == "semantic_hit" and e.prov.calls == 1 and r3.matched_question == "how do I reset my password"


def test_unrelated_prompt_is_a_miss():
    e = make_engine()
    e.handle(req("how do I reset my password"))
    assert e.handle(req("what is the capital of India")).outcome == "miss" and e.prov.calls == 2


def test_hit_records_token_and_dollar_savings():
    e = make_engine()
    e.handle(req("what are your business hours", model="gpt-4o"))
    r = e.handle(req("what are your business hours", model="gpt-4o"))
    assert r.outcome == "exact_hit" and r.saved_usd > 0 and r.out_tokens > 0
    rep = e.tm.report()["cost"]
    assert rep["saved_output_tokens"] == r.out_tokens and rep["saved_usd"] > 0


def test_ttl_expiry_forces_refresh():
    e = make_engine(policies={"default": {"auto_hit": .93, "borderline_low": .7, "ttl": 60, "cache_allowed": True}})
    e.handle(req("hello there friend"))
    e.clock.advance(61)
    assert e.handle(req("hello there friend")).outcome == "miss"


def test_different_model_system_prompt_do_not_share():
    e = make_engine()
    e.handle(req("same question here"))
    assert e.handle(req("same question here", system="other")).outcome == "miss"


def test_multiturn_context_is_part_of_the_key():
    e = make_engine()
    a = [{"role": "user", "content": "I love cats"}, {"role": "assistant", "content": "ok"},
         {"role": "user", "content": "why is that"}]
    b = [{"role": "user", "content": "I love rockets"}, {"role": "assistant", "content": "ok"},
         {"role": "user", "content": "why is that"}]
    e.handle(req("", messages=a))
    assert e.handle(req("", messages=b)).outcome == "miss"            # same last turn, different conversation
    assert e.handle(req("", messages=a)).outcome == "exact_hit"


def test_borderline_match_uses_verifier():
    yes = make_engine(rerank=FakeRerank(0.80), verifier=lambda q, a, m: True)
    yes.handle(req("can I get a refund"))
    assert yes.handle(req("is a refund possible")).outcome == "semantic_verified"
    no = make_engine(rerank=FakeRerank(0.80), verifier=lambda q, a, m: False)
    no.handle(req("can I get a refund"))
    assert no.handle(req("is a refund possible")).outcome == "miss" and no.prov.calls == 2
    assert no.tm.report()["quality"]["threshold_drift"]["samples"] == 1       # near-miss logged


def test_verifier_exception_never_reuses():
    def boom(*a):
        raise RuntimeError("judge down")
    e = make_engine(rerank=FakeRerank(0.80), verifier=boom)
    e.handle(req("can I get a refund"))
    assert e.handle(req("is a refund possible")).outcome == "miss"


# ------------------------------- isolation -----------------------------------

def test_tenants_never_share_entries():
    e = make_engine()
    e.handle(req("what is our internal pricing", tenant="acme"))
    r = e.handle(req("what is our internal pricing", tenant="globex"))
    assert r.outcome == "miss" and e.prov.calls == 2


def test_scope_mismatch_is_blocked_and_counted():
    """Defence in depth: even if an entry landed in a namespace it doesn't belong to
    (bad migration, key collision, tampering), the scope check refuses to serve it."""
    e = make_engine()
    e.handle(req("secret roadmap question", tenant="acme"))
    import json
    key = next(k for k in e.store.scan_iter(match="entry:*"))
    entry = json.loads(e.store.get(key))
    entry["tenant"] = "someone-else"
    e.store.set(key, json.dumps(entry))
    e.index._mem.clear()
    r = e.handle(req("secret roadmap question", tenant="acme"))
    assert r.outcome == "miss"
    assert e.tm.report()["reliability"]["isolation_blocks"] >= 1


def test_user_specific_never_cached_by_default():
    e = make_engine()
    e.handle(req("what is my balance", tag="user_specific", user="alice"))
    r = e.handle(req("what is my balance", tag="user_specific", user="alice"))
    assert r.outcome == "bypass" and e.prov.calls == 2


def test_user_specific_without_user_id_is_blocked_even_if_policy_allows():
    pol = {"user_specific": {"auto_hit": .93, "borderline_low": .7, "ttl": 60, "cache_allowed": True}}
    e = make_engine(policies=pol)
    r = e.handle(req("what is my balance", tag="user_specific", user=None))
    assert r.outcome == "bypass" and e.tm.report()["reliability"]["isolation_blocks"] == 1
    # with a user id the entry is per-user
    e.handle(req("what is my balance", tag="user_specific", user="alice"))
    assert e.handle(req("what is my balance", tag="user_specific", user="alice")).outcome == "exact_hit"
    assert e.handle(req("what is my balance", tag="user_specific", user="bob")).outcome == "miss"


# ------------------------------- fail-open -----------------------------------

class BrokenStore(MemoryStore):
    broken = False

    def get(self, k):
        if self.broken:
            raise ConnectionError("redis down")
        return super().get(k)

    def set(self, *a, **kw):
        if self.broken:
            raise ConnectionError("redis down")
        return super().set(*a, **kw)


def test_store_outage_fails_open_and_recovers():
    clock = Clock()
    store = BrokenStore(clock=clock)
    e = make_engine(store=store, clock=clock)
    e.handle(req("cache me please"))
    store.broken = True
    r = e.handle(req("cache me please"))
    assert r.outcome == "fail_open" and "fail-open" in r.note and r.text           # client still served
    assert e.tm.report()["reliability"]["fail_open_count"] >= 1
    store.broken = False
    e.breaker.success()
    assert e.handle(req("cache me please")).outcome == "exact_hit"                  # recovered


def test_circuit_breaker_skips_cache_after_repeated_failures():
    clock = Clock()
    store = BrokenStore(clock=clock)
    store.broken = True
    e = make_engine(store=store, clock=clock)
    for _ in range(3):
        e.handle(req("x y z"))
    assert e.breaker.is_open
    store.broken = False                       # store is healthy again, but the breaker is still open...
    assert e.handle(req("x y z")).outcome == "fail_open"      # ...so the cache is skipped entirely
    clock.advance(11)                                                                # cooldown -> half-open probe
    assert e.breaker.allow()


# ------------------------------- streaming -----------------------------------

def test_stream_miss_stores_then_replays_as_stream():
    e = make_engine()
    info, it = e.handle_stream(req("tell me a long story about caching", stream=True))
    live = "".join(it)
    assert info.outcome == "miss" and live == "ANSWER[tell me a long story about caching]"
    info2, it2 = e.handle_stream(req("tell me a long story about caching", stream=True))
    chunks = list(it2)
    assert info2.outcome == "exact_hit" and len(chunks) > 1 and "".join(chunks) == live and e.prov.calls == 1
    ttft = e.tm.report()["latency"]
    assert ttft["ttft_cached_ms"]["n"] == 1 and ttft["ttft_live_ms"]["n"] == 1


def test_stream_cache_is_shared_with_non_stream():
    e = make_engine()
    e.handle(req("shared across modes"))
    info, it = e.handle_stream(req("shared across modes", stream=True))
    assert info.cache_hit and "".join(it) == "ANSWER[shared across modes]"


def test_partial_stream_is_never_cached():
    e = make_engine()
    info, it = e.handle_stream(req("client will hang up", stream=True))
    next(it)
    it.close()                                                  # client disconnect
    assert e.handle(req("client will hang up")).outcome == "miss"


# ------------------------------- budgets -------------------------------------

def test_budget_blocks_live_calls_but_still_serves_cache_hits():
    e = make_engine()
    t = Tenant(tenant="acme", key_id="k", daily_tokens=1000)
    e.handle(req("warm the cache", tenant_obj=t))
    e.budget.record(t, 5000, 1.0)                                # blow the cap
    hit = e.handle(req("warm the cache", tenant_obj=t))
    assert hit.outcome == "exact_hit"                           # free, so allowed
    try:
        e.handle(req("a brand new question", tenant_obj=t))
        assert False, "should be rate limited"
    except GatewayError as ex:
        assert ex.status == 429 and ex.retry_after > 0
    assert e.tm.report()["reliability"]["budget_denied"] == 1


def test_only_live_calls_consume_budget():
    e = make_engine()
    t = Tenant(tenant="acme", key_id="k", daily_tokens=100000)
    e.handle(req("count me once", tenant_obj=t))
    used = e.budget.usage(t)["day_tokens"]
    for _ in range(5):
        e.handle(req("count me once", tenant_obj=t))
    assert e.budget.usage(t)["day_tokens"] == used


# ------------------------------- tools ---------------------------------------

def test_tool_requests_are_forwarded_and_never_cached():
    e = make_engine()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    r1 = e.handle(req("weather in Chennai", tools=tools))
    r2 = e.handle(req("weather in Chennai", tools=tools))
    assert r1.outcome == "bypass" and r2.outcome == "bypass" and e.prov.calls == 2 and r1.raw


# ------------------------------- admin ---------------------------------------

def test_invalidate_by_feature_and_tenant_and_time():
    e = make_engine()
    e.handle(req("alpha question one", feature="billing"))
    e.handle(req("beta question two", feature="support"))
    e.handle(req("gamma question three", feature="support", tenant="globex"))
    assert e.index.invalidate(feature="billing") == 1
    assert e.handle(req("alpha question one", feature="billing")).outcome == "miss"
    assert e.handle(req("beta question two", feature="support")).outcome == "exact_hit"
    assert e.index.invalidate(tenant="globex") == 1
    e.clock.advance(100)
    assert e.index.invalidate(created_before=e.clock() - 50) >= 1      # time-range rule
    assert e.index.count() == 0


def test_provider_error_maps_to_gateway_error():
    from gateway.providers import ProviderError
    e = make_engine()
    e.prov.fail = ProviderError(429, "upstream rate limit", 7)
    try:
        e.handle(req("anything at all"))
        assert False
    except GatewayError as ex:
        assert ex.status == 429 and ex.retry_after == 7


def test_index_capacity_evicts_oldest():
    clock = Clock()
    store = MemoryStore(clock=clock)
    idx = CacheIndex(store, max_per_ns=3, clock=clock)
    for i in range(5):
        clock.advance(1)
        idx.put("ns", f"id{i}", {"question": str(i), "answer": "a", "embedding": [1.0, 0.0]}, 1000)
    assert store.zcard("idx:ns") == 3
    assert idx.get_entry("ns", "id0") is None and idx.get_entry("ns", "id4") is not None


class DeadStore:
    """Every operation raises, like a Redis that is completely unreachable."""
    def __getattr__(self, name):
        def boom(*a, **kw):
            raise ConnectionError("redis unreachable")
        return boom


def test_total_store_outage_still_serves_every_request_live():
    clock = Clock()
    e = make_engine(store=DeadStore(), clock=clock)
    t = Tenant(tenant="acme", key_id="k", rpm=5, daily_tokens=1000)
    for _ in range(6):                                           # more than the rpm limit: budget store is down too
        r = e.handle(req("is anyone there", tenant_obj=t))
        assert r.text == "ANSWER[is anyone there]" and r.outcome == "fail_open"
    info, it = e.handle_stream(req("streaming while down", stream=True, tenant_obj=t))
    assert "".join(it) == "ANSWER[streaming while down]"
