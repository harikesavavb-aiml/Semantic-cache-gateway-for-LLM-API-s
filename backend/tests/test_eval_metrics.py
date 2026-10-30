"""
Tests for the evaluation suite's metric functions and orchestration.

Every expected value here is small enough to verify by hand. These tests
check the MATH of the evaluation; they say nothing about how well the
models perform (that requires running evaluation/run_eval.py on real data).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))

import eval_metrics as em  # noqa: E402
import run_eval  # noqa: E402


# ---------- confusion / summary ----------

def test_metrics_at_threshold_hand_computed():
    # threshold 0.5 -> predicted hit: 0.9(1) 0.8(0) 0.6(1) ; missed: 0.4(1) 0.2(0)
    m = em.metrics_at_threshold([0.9, 0.8, 0.6, 0.4, 0.2], [1, 0, 1, 1, 0], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["hits"] == 3
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_hit_rate"] == pytest.approx(1 / 3)


def test_zero_hits_means_zero_false_hits_but_is_visible_via_hits():
    m = em.metrics_at_threshold([0.1, 0.2], [1, 0], 0.99)
    assert m["hits"] == 0 and m["false_hit_rate"] == 0.0 and m["recall"] == 0.0


def test_threshold_boundary_is_inclusive():
    m = em.metrics_at_threshold([0.5], [1], 0.5)
    assert m["tp"] == 1


# ---------- sweep / thresholds ----------

def test_best_f1_threshold_on_perfectly_separable_data():
    assert em.best_f1_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 0.8


def test_threshold_for_precision_picks_max_recall_meeting_target():
    scores, labels = [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1]
    # precision by threshold: 0.9->1.0, 0.8->1.0, 0.7->0.667, 0.6->0.75
    assert em.threshold_for_precision(scores, labels, 0.99) == 0.8
    assert em.threshold_for_precision(scores, labels, 0.70) == 0.6
    assert em.threshold_for_precision(scores, labels, 1.01) is None


def test_ties_are_handled_as_one_threshold():
    thr, tp, fp, fn, tn = em.sweep([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert len(thr) == 1 and (tp[0], fp[0]) == (2, 2)


# ---------- AUC ----------

def test_auc_perfect_and_inverted_and_tied():
    assert em.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert em.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert em.auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_auc_none_when_single_class():
    assert em.auc([0.1, 0.9], [1, 1]) is None


# ---------- Wilson interval ----------

def test_wilson_interval_zero_of_ten():
    lo, hi = em.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)


def test_wilson_interval_empty_is_none():
    assert em.wilson_interval(0, 0) is None


def test_wilson_interval_contains_point_estimate():
    lo, hi = em.wilson_interval(30, 100)
    assert lo < 0.30 < hi


# ---------- split ----------

def test_split_is_disjoint_complete_and_deterministic():
    a1, b1 = em.split_indices(100, seed=7)
    a2, b2 = em.split_indices(100, seed=7)
    assert set(a1).isdisjoint(set(b1))
    assert sorted(np.concatenate([a1, b1])) == list(range(100))
    assert list(a1) == list(a2) and list(b1) == list(b2)


# ---------- calibration ----------

def test_calibrated_thresholds_are_ordered_and_meet_precision_on_calibration_data():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.4).astype(int)
    s = np.where(y == 1, rng.beta(8, 1.5, 500), rng.beta(1.2, 6, 500))
    z = em.calibrate_thresholds(s, y, target_precision=0.99, recall_floor=0.95)
    assert z["borderline_low"] <= z["auto_hit"]
    assert em.metrics_at_threshold(s, y, z["auto_hit"])["precision"] >= 0.99


def test_unreachable_precision_target_means_never_auto_hit():
    # A negative outscores every positive, so no threshold reaches 100% precision
    # except ones serving zero... the top-ranked item is wrong.
    s, y = [0.9, 0.8, 0.7], [0, 1, 1]
    z = em.calibrate_thresholds(s, y, target_precision=1.0)
    assert em.metrics_at_threshold(s, y, z["auto_hit"])["hits"] == 0


# ---------- three-zone pipeline ----------

def test_pipeline_without_verdicts_treats_border_as_miss():
    r = em.simulate_pipeline([0.95, 0.8, 0.8, 0.3], [1, 1, 0, 0], auto_hit=0.9, borderline_low=0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 0, 1, 2)
    assert r["border_n"] == 2 and r["verified_n"] == 0 and r["unverified_n"] == 2


def test_pipeline_with_correct_judge_recovers_the_duplicate():
    r = em.simulate_pipeline([0.95, 0.8, 0.8, 0.3], [1, 1, 0, 0], 0.9, 0.5, verdicts={1: True, 2: False})
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["judge_call_fraction"] == pytest.approx(2 / 4)


def test_pipeline_with_wrong_judge_creates_a_false_hit():
    r = em.simulate_pipeline([0.95, 0.8, 0.8, 0.3], [1, 1, 0, 0], 0.9, 0.5, verdicts={1: True, 2: True})
    assert r["fp"] == 1 and r["precision"] == pytest.approx(2 / 3)


def test_pipeline_below_borderline_low_is_never_judged():
    r = em.simulate_pipeline([0.2], [1], 0.9, 0.5, verdicts={0: True})
    assert r["border_n"] == 0 and r["tp"] == 0 and r["fn"] == 1


# ---------- orchestration ----------

def overlapping_scores(n=2000, seed=0):
    """Positives and negatives overlap heavily, so a verification zone exists."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    ce = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2.5, 3, n))
    cos = np.clip(ce + rng.normal(0, 0.05, n), 0, 1)
    return cos, ce, y


def border_count(ce, y, seed):
    cal, ev = em.split_indices(len(y), seed)
    z = em.calibrate_thresholds(ce[cal], y[cal], 0.99, 0.95)
    return len(em.border_positions(ce[ev], z["auto_hit"], z["borderline_low"]))


def test_evaluate_scores_end_to_end_exercises_the_judge_path():
    cos, ce, y = overlapping_scores(seed=3)
    assert border_count(ce, y, 3) > 10, "precondition: test data must produce a verification zone"
    judge = lambda i: bool(y[i])  # perfect judge
    res = run_eval.evaluate_scores("synthetic", cos, ce, y, seed=3, judge_fn=judge, judge_limit=2000)
    assert res["n_calibration"] + res["n_eval"] == len(y)

    base = res["methods"]["cross-encoder, three-zone (borderline treated as miss)"]
    with_judge = res["methods"]["cross-encoder, three-zone + LLM judge"]
    assert with_judge["verified_n"] > 0                      # the judge really ran
    assert with_judge["recall"] > base["recall"]             # and a perfect judge recovered duplicates
    assert with_judge["fp"] == base["fp"]                    # without adding any false hits


def test_judge_limit_refuses_rather_than_judging_a_biased_subset():
    cos, ce, y = overlapping_scores(seed=3)
    assert border_count(ce, y, 3) > 1, "precondition: need more borderline pairs than the limit"
    with pytest.raises(RuntimeError, match="judge-limit"):
        run_eval.evaluate_scores("synthetic", cos, ce, y, seed=3, judge_fn=lambda i: True, judge_limit=1)


def test_separable_data_has_no_verification_zone_so_judge_is_never_called():
    # When positives and negatives do not overlap, the calibrated thresholds
    # coincide: nothing is ambiguous, so nothing should cost a judge call.
    cos, ce, y = run_eval.synthetic_scores(n=1000, seed=3)
    assert border_count(ce, y, 3) == 0

    def must_not_be_called(i):
        raise AssertionError("judge was called although there is no borderline zone")

    res = run_eval.evaluate_scores("synthetic", cos, ce, y, seed=3, judge_fn=must_not_be_called)
    assert res["methods"]["cross-encoder, three-zone + LLM judge"]["judge_call_fraction"] == 0.0


def test_thresholds_are_chosen_on_calibration_half_only():
    # If calibration used the eval half, flipping eval labels would change the thresholds.
    cos, ce, y = overlapping_scores(n=800, seed=5)
    _, eval_idx = em.split_indices(len(y), 5)
    y2 = y.copy()
    y2[eval_idx] = 1 - y2[eval_idx]
    r1 = run_eval.evaluate_scores("a", cos, ce, y, seed=5)
    r2 = run_eval.evaluate_scores("a", cos, ce, y2, seed=5)
    assert r1["zones"] == r2["zones"]
    assert r1["thresholds"] == r2["thresholds"]


def test_markdown_report_contains_caveats_and_every_method():
    cos, ce, y = overlapping_scores(n=600, seed=1)
    res = run_eval.evaluate_scores("synthetic", cos, ce, y, seed=1)
    md = run_eval.format_markdown([res], {"seed": 1})
    assert "Caveats" in md and "QQP overlap" in md
    for method in res["methods"]:
        assert method in md
