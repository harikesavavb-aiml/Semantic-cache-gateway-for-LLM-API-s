"""ROI, latency and quality metrics (spec sections 1-3)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import *  # noqa
from gateway.pricing import call_cost, estimate_tokens, net_roi, price_for
from gateway.telemetry import jaccard, percentile, threshold_drift


def test_savings_formula_matches_spec():
    # Savings = in*Price_In + out*Price_Out  (prices are per 1M tokens)
    assert abs(call_cost("gpt-4o", 1_000_000, 1_000_000, DEFAULT_PRICES) - (2.50 + 10.00)) < 1e-9
    assert abs(call_cost("gpt-4o", 2000, 500, DEFAULT_PRICES) - (2000 * 2.5 + 500 * 10) / 1e6) < 1e-12


def test_longest_prefix_pricing_wins():
    assert price_for("gpt-4o-mini-2024", DEFAULT_PRICES) == DEFAULT_PRICES["gpt-4o-mini"]
    assert price_for("gpt-4o-2024", DEFAULT_PRICES) == DEFAULT_PRICES["gpt-4o"]
    assert price_for("claude-sonnet-4-5", DEFAULT_PRICES) == DEFAULT_PRICES["claude-sonnet"]
    assert price_for("unknown-model", DEFAULT_PRICES) == DEFAULT_PRICES["default"]


def test_net_roi_subtracts_opex():
    assert abs(net_roi(10.0, 0.25, 1.75) - 8.0) < 1e-12


def test_token_estimate_floor():
    assert estimate_tokens("") == 0 and estimate_tokens("a") == 1 and estimate_tokens("a" * 40) == 10


def test_percentile_nearest_rank():
    v = list(range(1, 101))
    assert percentile(v, 50) == 50 and percentile(v, 95) == 95 and percentile(v, 99) == 99
    assert percentile([], 95) is None


def test_jaccard():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b", "c d") == 0.0
    assert abs(jaccard("a b c", "a b d") - 0.5) < 1e-9


def test_threshold_drift_needs_samples_and_detects_direction():
    assert threshold_drift([{"similarity": .8, "verified": True}] * 5)["drift"] is None
    # newest first: recent near-misses score LOWER than older ones -> negative drift
    nm = [{"similarity": 0.72, "verified": False}] * 15 + [{"similarity": 0.88, "verified": True}] * 15
    d = threshold_drift(nm)
    assert d["drift"] < 0 and d["verified_rate_gap"] < 0


def test_threshold_advice_when_everything_verifies():
    nm = [{"similarity": 0.9, "verified": True}] * 30
    assert "lowering auto_hit" in threshold_drift(nm)["advice"]


def test_report_cost_latency_and_tenant_attribution():
    eng = make_engine()
    tm = eng.tm
    kw = dict(tenant="acme", unit="support", key_id="k1", feature="faq", model="gpt-4o")
    tm.record_request(outcome="miss", total_ms=1500, lookup_ms=8, spent_in=100, spent_out=200, spent_usd=0.0023, embedded=True, **kw)
    tm.record_request(outcome="exact_hit", total_ms=6, lookup_ms=1, saved_in=100, saved_out=200, saved_usd=0.0023, **kw)
    tm.record_request(outcome="semantic_hit", total_ms=14, lookup_ms=9, saved_in=100, saved_out=200, saved_usd=0.0023, embedded=True, **kw)
    tm.record_request(outcome="miss", total_ms=900, lookup_ms=30, tenant="globex", unit="sales", key_id="k2",
                      feature="faq", model="gpt-4o", spent_usd=0.01)
    r = tm.report()
    assert r["overview"]["requests"] == 4 and r["overview"]["hits"] == 2
    assert r["cost"]["saved_input_tokens"] == 200 and r["cost"]["saved_output_tokens"] == 400
    assert abs(r["cost"]["saved_usd"] - 0.0046) < 1e-9
    assert abs(r["cost"]["embedding_cost_usd"] - 2 / 1000 * 0.01) < 1e-12      # 2 embeddings @ $0.01/1k
    assert abs(r["cost"]["net_roi_usd"] - (0.0046 - 2 / 1000 * 0.01)) < 1e-9
    assert r["by_tenant"]["acme"]["saved_usd"] > 0 and r["by_tenant"]["globex"]["saved_usd"] == 0
    assert r["by_unit"]["support"]["hits"] == 2
    lat = r["latency"]
    assert lat["cached_ms"]["p95"] == 14 and lat["live_ms"]["p95"] == 1500 and lat["p95_reduction_ms"] == 1486
    assert lat["lookup_overhead_ms"]["slo_breaches"] == 1                       # the 30ms lookup
    assert [row["unit"] for row in tm.chargeback()] == ["sales", "support"]       # sorted by live spend


def test_false_positive_rate_from_feedback_is_idempotent():
    eng = make_engine()
    tm = eng.tm
    for i in range(4):
        tm.remember_hit(f"r{i}", {"feature": "faq"})
    assert tm.feedback("r0", bad=True) and tm.feedback("r1", bad=False)
    assert tm.feedback("r2", bad=False, source="judge") and tm.feedback("r3", bad=False, source="judge")
    tm.feedback("r0", bad=False)                                   # duplicate vote ignored
    q = tm.report()["quality"]
    assert q["judged_hits"] == 4 and q["bad_hits"] == 1 and q["false_positive_rate"] == 0.25
    assert not tm.feedback("unknown", bad=True)


def test_semantic_divergence_index():
    eng = make_engine()
    eng.tm.revalidation("news", "the sky is blue today", "the sky is blue today")
    eng.tm.revalidation("news", "markets rose sharply", "markets fell sharply")
    idx = eng.tm.report()["quality"]["semantic_divergence"]
    assert idx["revalidation_samples"] == 2 and 0 < idx["index"] < 1
