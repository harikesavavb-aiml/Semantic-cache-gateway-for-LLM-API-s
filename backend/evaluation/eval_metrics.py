"""
eval_metrics.py

Pure metric functions for evaluating a semantic cache's matching decision.
Depends only on numpy so everything here is unit-testable without models,
datasets, or network access.

Framing: every labelled pair (question A, question B) is a cache decision.
  label 1 = the questions are duplicates -> serving a cached answer is correct
  label 0 = they are NOT duplicates      -> serving a cached answer is a WRONG HIT

  "hit"            = the system decided to reuse a cached answer
  "false-hit rate" = wrong hits / all hits served   (= 1 - precision)
                     The safety metric. A cache that serves wrong answers is
                     worse than no cache.
  "recall"         = true duplicates that were caught / all true duplicates
                     The savings metric. Missed duplicates only cost money.

Convention: predict "hit" when score >= threshold.
"""

import numpy as np


def _arrays(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same shape")
    return s, y


def summarize(tp, fp, fn, tn):
    """Derive standard metrics from confusion counts."""
    hits = tp + fp
    # With zero hits served there are zero wrong hits; callers should look at
    # `hits` to notice that a "perfect" precision here means "did nothing".
    precision = tp / hits if hits else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "hits": int(hits),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_hit_rate": float(fp / hits) if hits else 0.0,
    }


def metrics_from_pred(pred, labels):
    pred = np.asarray(pred, dtype=bool)
    y = np.asarray(labels, dtype=int)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return summarize(tp, fp, fn, tn)


def metrics_at_threshold(scores, labels, threshold):
    s, y = _arrays(scores, labels)
    out = metrics_from_pred(s >= threshold, y)
    out["threshold"] = float(threshold)
    return out


def sweep(scores, labels):
    """
    Evaluate every distinct score as a threshold, in O(n log n).
    Returns arrays ordered from the HIGHEST threshold to the lowest:
    (thresholds, tp, fp, fn, tn).
    """
    s, y = _arrays(scores, labels)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    distinct = np.where(np.diff(s_sorted) != 0)[0]
    idx = np.concatenate([distinct, [len(s_sorted) - 1]])
    total_pos = int(y.sum())
    total_neg = int(len(y) - total_pos)
    thr = s_sorted[idx]
    tp, fp = tp_cum[idx], fp_cum[idx]
    return thr, tp, fp, total_pos - tp, total_neg - fp


def best_f1_threshold(scores, labels):
    thr, tp, fp, fn, _ = sweep(scores, labels)
    precision = tp / (tp + fp)
    total_pos = tp[-1] + fn[-1]
    recall = tp / total_pos if total_pos else np.zeros_like(tp, dtype=float)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    return float(thr[int(np.argmax(f1))])


def threshold_for_precision(scores, labels, min_precision):
    """
    The threshold that catches the most true duplicates while keeping
    precision >= min_precision. Returns None if no threshold reaches it.
    """
    thr, tp, fp, fn, _ = sweep(scores, labels)
    precision = tp / (tp + fp)
    total_pos = tp[-1] + fn[-1]
    recall = tp / total_pos if total_pos else np.zeros_like(tp, dtype=float)
    ok = precision >= min_precision
    if not ok.any():
        return None
    masked = np.where(ok, recall, -1.0)
    # argmax returns the first maximum = the highest threshold among ties,
    # i.e. the most conservative way to reach that recall.
    return float(thr[int(np.argmax(masked))])


def auc(scores, labels):
    """Area under the ROC curve (trapezoid rule). None if only one class."""
    thr, tp, fp, fn, tn = sweep(scores, labels)
    total_pos = int(tp[-1] + fn[-1])
    total_neg = int(fp[-1] + tn[-1])
    if total_pos == 0 or total_neg == 0:
        return None
    tpr = np.concatenate([[0.0], tp / total_pos])
    fpr = np.concatenate([[0.0], fp / total_neg])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n. None if n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def split_indices(n, seed, frac=0.5):
    """Deterministic calibration/evaluation split, so thresholds are never
    tuned on the data they are reported on."""
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(n * frac)
    return perm[:cut], perm[cut:]


def calibrate_thresholds(cal_scores, cal_labels, target_precision=0.99, recall_floor=0.95):
    """
    Choose three-zone thresholds from a calibration split only.

      auto_hit       : the lowest threshold that still keeps precision at
                       target_precision -> reuse with no further check.
      borderline_low : the score below which we discard at most
                       (1 - recall_floor) of true duplicates -> below this,
                       don't bother verifying.
    Scores between the two go to verification.
    """
    s, y = _arrays(cal_scores, cal_labels)
    auto_hit = threshold_for_precision(s, y, target_precision)
    if auto_hit is None:
        auto_hit = float(np.max(s)) + 1e-6  # unreachable: never auto-hit
    pos = s[y == 1]
    borderline_low = float(np.quantile(pos, 1 - recall_floor)) if len(pos) else 0.0
    borderline_low = min(borderline_low, auto_hit)
    return {"auto_hit": float(auto_hit), "borderline_low": float(borderline_low)}


def border_positions(scores, auto_hit, borderline_low):
    s = np.asarray(scores, dtype=float)
    return np.flatnonzero((s >= borderline_low) & (s < auto_hit))


def pipeline_predictions(scores, auto_hit, borderline_low, verdicts=None):
    """
    Per-pair hit decisions of the three-zone pipeline.
    Returns (pred, border_positions, verified_count).
    """
    s = np.asarray(scores, dtype=float)
    pred = s >= auto_hit
    border = border_positions(s, auto_hit, borderline_low)
    verified = 0
    if verdicts:
        for pos in border:
            pos = int(pos)
            if pos in verdicts:
                pred[pos] = bool(verdicts[pos])
                verified += 1
    return pred, border, verified


def simulate_pipeline(scores, labels, auto_hit, borderline_low, verdicts=None):
    """
    Simulate the gateway's three-zone decision on labelled pairs.

      score >= auto_hit          -> hit (no verification)
      borderline_low <= score    -> verification zone:
                                      verdict True  -> hit
                                      verdict False -> miss
                                      no verdict    -> miss (conservative)
      score <  borderline_low    -> miss

    `verdicts` maps position-in-this-array -> bool (LLM judge's answer).
    """
    s, y = _arrays(scores, labels)
    pred, border, verified = pipeline_predictions(s, auto_hit, borderline_low, verdicts)
    out = metrics_from_pred(pred, y)
    out.update({
        "auto_hit_n": int(np.sum(s >= auto_hit)),
        "border_n": int(len(border)),
        "verified_n": int(verified),
        "unverified_n": int(len(border) - verified),
        "judge_call_fraction": float(verified / len(y)) if len(y) else 0.0,
    })
    return out


def category_breakdown(pred, labels, categories):
    """
    How often a method reuses a cached answer, split by pair type.

    For a category of true duplicates, `rate` is the share reused (recall:
    higher is better). For a category of non-duplicates, `rate` is the share
    WRONGLY reused (lower is better). `kind` says which one it is.
    """
    pred = np.asarray(pred, dtype=bool)
    y = np.asarray(labels, dtype=int)
    cats = np.asarray(categories)
    out = {}
    for c in sorted(set(cats.tolist())):
        m = cats == c
        n = int(m.sum())
        k = int(pred[m].sum())
        if y[m].min() == y[m].max():
            kind = "duplicate" if y[m][0] == 1 else "non-duplicate"
        else:
            kind = "mixed"
        ci = wilson_interval(k, n)
        out[c] = {"n": n, "reused": k, "rate": float(k / n) if n else None,
                  "ci": None if ci is None else [ci[0], ci[1]], "kind": kind}
    return out
