"""Dynamic token budgeting & throttling, API-key auth (spec: budgeting, multi-tenant)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import *  # noqa
from gateway.tenants import PUBLIC, authenticate, extract_key, hash_key, load_tenants


def mk(**kw):
    clock = Clock()
    store = MemoryStore(clock=clock)
    return BudgetManager(store, clock=clock), Tenant(tenant="acme", key_id="k", **kw), clock


def test_rpm_limit_denies_with_retry_after():
    b, t, _ = mk(rpm=3)
    assert all(b.check(t).allowed for _ in range(3))
    d = b.check(t)
    assert not d.allowed and d.retry_after > 0


def test_hard_cap_denies_until_reset():
    b, t, clock = mk(daily_usd=1.0)
    b.record(t, 1000, 1.0)
    d = b.check(t)
    assert not d.allowed and d.state == "denied" and "daily_usd" in d.reason
    clock.advance(86400)
    assert b.check(t).allowed


def test_soft_threshold_throttles_rpm_before_cutoff():
    b, t, _ = mk(rpm=100, daily_tokens=1000)
    b.record(t, 850, 0.0)                                  # 85% of cap
    results = [b.check(t) for _ in range(30)]
    assert results[0].state == "throttled"
    assert sum(r.allowed for r in results) == 25            # 100 * 0.25 throttle factor
    assert not results[-1].allowed


def test_token_cap_counts_live_usage_only():
    b, t, _ = mk(daily_tokens=100)
    b.record(t, 60, 0.0)
    assert b.check(t).allowed
    b.record(t, 60, 0.0)
    assert not b.check(t).allowed


def test_unlimited_tenant_never_blocked():
    b, t, _ = mk()
    for _ in range(200):
        assert b.check(t).allowed


def test_monthly_cap():
    b, t, clock = mk(monthly_usd=5.0)
    b.record(t, 1, 5.0)
    assert not b.check(t).allowed
    clock.advance(32 * 86400)
    assert b.check(t).allowed


def test_authentication():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({hash_key("sk-good"): {"tenant": "acme", "unit": "support", "rpm": 5}}, f)
    tenants = load_tenants(f.name)
    assert authenticate(tenants, "sk-good", True).tenant == "acme"
    assert authenticate(tenants, "sk-bad", True) is None
    assert authenticate(tenants, None, True) is None
    assert authenticate({}, None, False) is PUBLIC          # dev mode
    assert authenticate({}, None, True) is None             # auth required but nothing configured


def test_key_extraction_from_either_sdk_header():
    assert extract_key({"Authorization": "Bearer sk-1"}) == "sk-1"
    assert extract_key({"x-api-key": "sk-2"}) == "sk-2"
    assert extract_key({}) is None
