"""
Tests for the synthetic hard-negative set (evaluation/hardneg.py).

The key test is label correctness: every generated label is re-derived from
an independent lookup table (meaning_of), so a pairing bug cannot silently
mislabel the evaluation data.
"""

import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))

import hardneg as h  # noqa: E402
import eval_metrics as em  # noqa: E402
import run_eval  # noqa: E402

N = 1200
ROWS = h.generate_pairs(N, seed=7)


def test_deterministic_for_seed_and_different_across_seeds():
    assert h.generate_pairs(300, seed=1) == h.generate_pairs(300, seed=1)
    assert h.generate_pairs(300, seed=1) != h.generate_pairs(300, seed=2)


def test_size_and_positive_fraction():
    assert len(ROWS) == N
    assert sum(1 for r in ROWS if r[2] == 1) == round(N * 0.4)


def test_negative_quotas_match_design():
    neg = Counter(r[3] for r in ROWS if r[2] == 0)
    n_neg = N - round(N * 0.4)
    for cat, share in h.NEGATIVE_QUOTAS.items():
        assert abs(neg[cat] - share * n_neg) <= 1, cat


def test_paraphrase_category_is_exactly_the_positives():
    assert all((r[3] == "paraphrase") == (r[2] == 1) for r in ROWS)


def test_every_text_is_known_and_unambiguous():
    # build_meaning_table raises on ambiguous text; also every generated text must resolve.
    assert len(h.build_meaning_table()) > 500
    for a, b, _, _ in ROWS:
        assert h.meaning_of(a) is not None and h.meaning_of(b) is not None


def test_labels_agree_with_independent_oracle():
    for a, b, label, cat in ROWS:
        assert (h.meaning_of(a) == h.meaning_of(b)) == bool(label), (a, b, label, cat)


def test_no_duplicate_pairs_and_no_identical_questions():
    keys = [h._key(a, b) for a, b, _, _ in ROWS]
    assert len(set(keys)) == len(keys)
    assert all(h._norm(a) != h._norm(b) for a, b, _, _ in ROWS)


def test_each_category_means_what_its_name_says():
    for a, b, label, cat in ROWS:
        ma, mb = h.meaning_of(a), h.meaning_of(b)
        if cat == "entity_swap":      # same family + attribute, different entity
            assert ma[:3] == mb[:3] and ma[3] != mb[3]
        elif cat == "attribute_swap":  # same family + entity, different attribute
            assert ma[1] == mb[1] and ma[3] == mb[3] and ma[2] != mb[2]
        elif cat == "number_swap":     # same intent, different number
            assert ma[:2] == mb[:2] and ma[2] != mb[2]
        elif cat == "polarity_swap":   # same object, opposite verb
            assert ma[0] == mb[0] == "pol" and ma[2] == mb[2] and ma[1] != mb[1]
        elif cat == "unrelated":       # different topic groups
            assert ma[0] != mb[0]
        else:
            assert cat == "paraphrase" and ma == mb


def test_polarity_pairs_use_genuine_opposites():
    opposite = {}
    for v1, v2, _ in h.POLARITY:
        opposite[v1], opposite[v2] = v2, v1
    for a, b, _, cat in ROWS:
        if cat == "polarity_swap":
            assert opposite[h.meaning_of(a)[1]] == h.meaning_of(b)[1]


def test_formatting_does_not_reveal_the_label():
    # Casual (lowercase, no punctuation) text must be equally common for
    # paraphrases and non-paraphrases, or a model could cheat on style.
    # Averaged over many seeds: a single seed's gap is mostly sampling noise
    # (sd ~0.04 at n=600), while a real leak produces a gap near 0.6.
    gaps, pos_rates, neg_rates = [], [], []
    for seed in range(10):
        rows = h.generate_pairs(600, seed=seed)

        def casual_rate(label):
            # Un-transformed questions always start with a capital letter.
            return np.mean([r[1] == r[1].lower() for r in rows if r[2] == label])

        pos_rates.append(casual_rate(1))
        neg_rates.append(casual_rate(0))
        gaps.append(pos_rates[-1] - neg_rates[-1])
    assert abs(np.mean(gaps)) < 0.04
    assert 0.25 < np.mean(pos_rates) < 0.35 and 0.25 < np.mean(neg_rates) < 0.35


def test_capacity_error_is_explicit():
    with pytest.raises(ValueError, match="at most"):
        h.generate_pairs(10 ** 5)
    assert h.max_pairs() >= 1500


def test_max_supported_size_actually_generates():
    rows = h.generate_pairs(h.max_pairs(), seed=0)
    assert len(rows) == h.max_pairs()


# ---------- integration with the evaluation runner ----------

def test_pipeline_predictions_match_simulate_pipeline():
    s = [0.95, 0.8, 0.8, 0.3]
    y = [1, 1, 0, 0]
    pred, border, verified = em.pipeline_predictions(s, 0.9, 0.5, {1: True, 2: False})
    assert list(pred) == [True, True, False, False] and list(border) == [1, 2] and verified == 2
    m = em.simulate_pipeline(s, y, 0.9, 0.5, {1: True, 2: False})
    assert (m["tp"], m["fp"]) == (2, 0)


def test_category_breakdown_counts_and_kinds():
    pred = [True, True, False, True, False, False]
    y = [1, 1, 0, 0, 0, 1]
    cats = ["paraphrase", "paraphrase", "entity_swap", "entity_swap", "unrelated", "paraphrase"]
    b = em.category_breakdown(pred, y, cats)
    assert b["paraphrase"]["n"] == 3 and b["paraphrase"]["reused"] == 2 and b["paraphrase"]["kind"] == "duplicate"
    assert b["entity_swap"]["n"] == 2 and b["entity_swap"]["reused"] == 1 and b["entity_swap"]["kind"] == "non-duplicate"
    assert b["unrelated"]["reused"] == 0 and b["unrelated"]["ci"][0] == 0.0


def test_evaluate_scores_reports_breakdown_and_markdown_renders_it():
    cos, ce, y = run_eval.synthetic_scores(n=1000, seed=2)
    rng = np.random.default_rng(2)
    cats = np.where(y == 1, "paraphrase", rng.choice(["entity_swap", "unrelated"], len(y)))
    res = run_eval.evaluate_scores("hardneg", cos, ce, y, seed=2, categories=cats)
    _, eval_idx = em.split_indices(len(y), 2)
    for r in res["methods"].values():
        assert r["by_category"]["paraphrase"]["n"] == int(y[eval_idx].sum())
        assert sum(b["n"] for b in r["by_category"].values()) == len(eval_idx)
    md = run_eval.format_markdown([res], {"seed": 2})
    assert "Reuse rate by pair type" in md and "hardneg` is synthetic" in md


def test_evaluate_scores_without_categories_has_no_breakdown():
    cos, ce, y = run_eval.synthetic_scores(n=600, seed=2)
    res = run_eval.evaluate_scores("x", cos, ce, y, seed=2)
    assert all("by_category" not in r for r in res["methods"].values())
    assert "Reuse rate by pair type" not in run_eval.format_markdown([res], {})
