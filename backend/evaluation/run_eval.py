"""
run_eval.py - reproducible evaluation of the semantic cache's matching stage
on public, labelled duplicate-detection datasets.

Usage (from the backend/ folder):
    pip install -r evaluation/requirements-eval.txt
    python3 evaluation/run_eval.py --datasets qqp,paws --n 2000
    python3 evaluation/run_eval.py --datasets qqp,paws --n 2000 --judge   # needs GROQ_API_KEY
    python3 evaluation/run_eval.py --selftest                             # no models/network needed

Method: for each dataset, sample n labelled pairs (fixed seed), split them in
half. Thresholds are chosen on the calibration half ONLY and every number in
the report is computed on the held-out half.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_metrics as em  # noqa: E402

BI_ENCODER_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/quora-distilroberta-base"


# ----------------------------------------------------------------------------
# Orchestration (pure: operates on already-computed scores)
# ----------------------------------------------------------------------------

def _or_never(threshold, scores):
    """A threshold of None means 'precision target unreachable' -> serve no hits."""
    return float(np.max(scores)) + 1.0 if threshold is None else float(threshold)


def evaluate_scores(name, cos, ce, labels, seed, judge_fn=None, judge_limit=400,
                    target_precision=0.99, recall_floor=0.95, categories=None):
    cos = np.asarray(cos, dtype=float)
    ce = np.asarray(ce, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = len(y)

    cal_idx, eval_idx = em.split_indices(n, seed)
    cal_cos, cal_ce, cal_y = cos[cal_idx], ce[cal_idx], y[cal_idx]
    ev_cos, ev_ce, ev_y = cos[eval_idx], ce[eval_idx], y[eval_idx]

    thr = {
        "cosine_best_f1": em.best_f1_threshold(cal_cos, cal_y),
        "cosine_precision_target": em.threshold_for_precision(cal_cos, cal_y, target_precision),
        "ce_best_f1": em.best_f1_threshold(cal_ce, cal_y),
        "ce_precision_target": em.threshold_for_precision(cal_ce, cal_y, target_precision),
    }
    zones = em.calibrate_thresholds(cal_ce, cal_y, target_precision, recall_floor)

    methods = {}
    preds = {}
    ev_cats = None if categories is None else np.asarray(categories)[eval_idx]

    def add(label, scores, threshold):
        t = _or_never(threshold, scores)
        r = em.metrics_at_threshold(scores, ev_y, t)
        r["judge_call_fraction"] = 0.0
        r["threshold_display"] = "unreachable (serves no hits)" if threshold is None else f"{t:.3f}"
        methods[label] = r
        preds[label] = np.asarray(scores, dtype=float) >= t

    pct = int(target_precision * 100)
    add("cosine, single threshold (best F1)", ev_cos, thr["cosine_best_f1"])
    add(f"cosine, single threshold ({pct}% precision)", ev_cos, thr["cosine_precision_target"])
    add("cross-encoder, single threshold (best F1)", ev_ce, thr["ce_best_f1"])
    add(f"cross-encoder, single threshold ({pct}% precision)", ev_ce, thr["ce_precision_target"])

    zone_label = f"auto>={zones['auto_hit']:.3f} / verify>={zones['borderline_low']:.3f}"
    sim = em.simulate_pipeline(ev_ce, ev_y, zones["auto_hit"], zones["borderline_low"])
    sim["threshold_display"] = zone_label
    methods["cross-encoder, three-zone (borderline treated as miss)"] = sim
    preds["cross-encoder, three-zone (borderline treated as miss)"] = em.pipeline_predictions(
        ev_ce, zones["auto_hit"], zones["borderline_low"])[0]

    if judge_fn is not None:
        border = em.border_positions(ev_ce, zones["auto_hit"], zones["borderline_low"])
        if len(border) > judge_limit:
            raise RuntimeError(
                f"{len(border)} borderline pairs need judging but --judge-limit is {judge_limit}. "
                f"Lower --n or raise --judge-limit (judging a subset would bias the result)."
            )
        verdicts = {int(p): bool(judge_fn(int(eval_idx[p]))) for p in border}
        simj = em.simulate_pipeline(ev_ce, ev_y, zones["auto_hit"], zones["borderline_low"], verdicts)
        simj["threshold_display"] = zone_label
        methods["cross-encoder, three-zone + LLM judge"] = simj
        preds["cross-encoder, three-zone + LLM judge"] = em.pipeline_predictions(
            ev_ce, zones["auto_hit"], zones["borderline_low"], verdicts)[0]

    for label, r in methods.items():
        r["false_hit_ci"] = em.wilson_interval(r["fp"], r["hits"])
        if ev_cats is not None:
            r["by_category"] = em.category_breakdown(preds[label], ev_y, ev_cats)

    return {
        "dataset": name,
        "n_total": int(n),
        "n_calibration": int(len(cal_idx)),
        "n_eval": int(len(eval_idx)),
        "duplicates_in_eval": int(ev_y.sum()),
        "auc": {"cosine": em.auc(ev_cos, ev_y), "cross_encoder": em.auc(ev_ce, ev_y)},
        "thresholds": {k: (None if v is None else float(v)) for k, v in thr.items()},
        "zones": zones,
        "methods": methods,
    }


def format_markdown(all_results, meta):
    L = []
    L.append("# Evaluation results\n")
    L.append("Generated by `evaluation/run_eval.py`. All numbers below are computed on a held-out "
             "half of each sample; thresholds were chosen on the other half.\n")
    L.append("## Run configuration\n")
    for k, v in meta.items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## How to read the tables\n")
    L.append("- **Hits served**: cached answers the method chose to reuse.")
    L.append("- **False-hit rate**: share of those hits that were WRONG (the pair was not a duplicate). "
             "This is the safety metric. 95% Wilson interval in brackets.")
    L.append("- **Duplicates caught**: share of true duplicates that were reused (recall). "
             "This is the savings metric; a miss only costs an extra LLM call.")
    L.append("- **Judge calls**: LLM verification calls as a share of all evaluated pairs (the cost of the safety step).\n")

    for res in all_results:
        L.append(f"## {res['dataset']}\n")
        L.append(f"Held-out pairs: {res['n_eval']} ({res['duplicates_in_eval']} duplicates). "
                 f"ROC-AUC: cosine {_fmt(res['auc']['cosine'])}, cross-encoder {_fmt(res['auc']['cross_encoder'])}.\n")
        L.append("| Method | Threshold | Hits served | False-hit rate (95% CI) | Duplicates caught | F1 | Judge calls |")
        L.append("|---|---|---|---|---|---|---|")
        for label, r in res["methods"].items():
            ci = r["false_hit_ci"]
            ci_s = "n/a" if ci is None else f"[{ci[0]:.1%}, {ci[1]:.1%}]"
            L.append(
                f"| {label} | {r['threshold_display']} | {r['hits']} | "
                f"{r['false_hit_rate']:.1%} {ci_s} | {r['recall']:.1%} | {r['f1']:.3f} | "
                f"{r['judge_call_fraction']:.1%} |"
            )
        L.append("")
        L.extend(_category_table(res))

    L.append("## Caveats (read before citing these numbers)\n")
    L.append("- **QQP overlap.** The cross-encoder (`quora-distilroberta-base`) and likely the bi-encoder were "
             "trained on data that includes Quora duplicate questions. QQP results are probably optimistic and "
             "should not be presented as unseen-data performance. PAWS is the fairer out-of-distribution test.")
    L.append("- **PAWS is not user questions.** It is adversarial Wikipedia sentence pairs (high word overlap, "
             "different meaning). It stress-tests the failure mode that matters, but real chat traffic differs.")
    L.append("- **The judge is a proxy.** Public datasets contain no cached answers, so the LLM judge here decides "
             "whether two questions have the same intent. The gateway's production check also sees the cached answer.")
    L.append("- **One seed, one sample.** Intervals reflect sampling error on the held-out half only, not variation "
             "across seeds or model versions. Re-run with other `--seed` values before drawing strong conclusions.")
    L.append("- **Cross-domain transfer is untested.** Thresholds calibrated on one dataset are not evaluated on another.")
    if any(r["dataset"] == "hardneg" for r in all_results):
        L.append("- **`hardneg` is synthetic.** Templated pairs written by the project authors; language is far "
                 "more repetitive than real traffic, seeds draw from one finite pool (so are not independent "
                 "replications), and it tests a single failure mode. Good results here do not show general quality; "
                 "poor results are informative.")
    return "\n".join(L) + "\n"


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


CATEGORY_ORDER = ["paraphrase", "entity_swap", "number_swap", "attribute_swap",
                  "polarity_swap", "unrelated"]


def _category_table(res):
    first = next(iter(res["methods"].values()))
    if "by_category" not in first:
        return []
    cats = [c for c in CATEGORY_ORDER if c in first["by_category"]]
    cats += [c for c in first["by_category"] if c not in cats]
    lines = ["**Reuse rate by pair type** (held-out). The `paraphrase` column is the share of true "
             "paraphrases reused (higher is better). Every other column is the share of that kind of "
             "NON-duplicate that was wrongly reused (lower is better). Format: reused/n, rate, "
             "95% interval.\n",
             "| Method | " + " | ".join(cats) + " |",
             "|---|" + "---|" * len(cats)]
    for label, r in res["methods"].items():
        cells = []
        for c in cats:
            b = r["by_category"][c]
            if b["n"] == 0 or b["ci"] is None:
                cells.append("n/a")
            else:
                cells.append(f"{b['reused']}/{b['n']}, {b['rate']:.0%} [{b['ci'][0]:.0%}-{b['ci'][1]:.0%}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# ----------------------------------------------------------------------------
# Data loading, scoring, judging (heavy deps imported lazily)
# ----------------------------------------------------------------------------

def _sample(rows, n, seed):
    rows = [r for r in rows if r[0] and r[1]]
    if n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def load_qqp(n, seed):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    return _sample([(r["question1"], r["question2"], int(r["label"])) for r in ds], n, seed)


def load_paws(n, seed):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    return _sample([(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds], n, seed)


def load_hardneg(n, seed):
    from hardneg import generate_pairs, max_pairs
    cap = max_pairs()
    if n > cap:
        print(f"hardneg can supply at most {cap} pairs; using n={cap} instead of {n}.")
        n = cap
    return generate_pairs(n, seed)


LOADERS = {"qqp": load_qqp, "paws": load_paws, "hardneg": load_hardneg}


def score_pairs(pairs, bi_encoder, cross_encoder, batch_size=64):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    ea = bi_encoder.encode(a, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    eb = bi_encoder.encode(b, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    cos = (ea * eb).sum(axis=1)  # unit vectors, so dot product == cosine similarity
    ce = cross_encoder.predict(list(zip(a, b)), batch_size=batch_size, show_progress_bar=True)
    return cos, np.asarray(ce, dtype=float)


JUDGE_PROMPT = """Question A: "{a}"
Question B: "{b}"

Are these two questions asking for the same thing, such that one answer would correctly and fully answer both?
Reply with only one word: YES or NO."""


def make_llm_judge(pairs, cache_path):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("DEFAULT_LLM_MODEL", "llama-3.1-8b-instant")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    def save():
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    def judge(i):
        a, b = pairs[i][0], pairs[i][1]
        key = hashlib.sha256(f"{model}|{a}|{b}".encode()).hexdigest()
        if key in cache:
            return cache[key]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(a=a, b=b)}],
                )
                verdict = resp.choices[0].message.content.strip().upper().startswith("YES")
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("LLM judge failed repeatedly; check API key, model name, and rate limits.")
        cache[key] = verdict
        if len(cache) % 25 == 0:
            save()
        return verdict

    return judge, save, model


def _version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# Self-test: exercises the whole orchestration path on synthetic scores
# ----------------------------------------------------------------------------

def synthetic_scores(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    cos = np.where(y == 1, rng.beta(6, 2.5, n), rng.beta(3, 4, n))
    ce = np.where(y == 1, rng.beta(8, 1.5, n), rng.beta(1.2, 6, n))
    return cos, ce, y


def run_selftest(seed):
    cos, ce, y = synthetic_scores(seed=seed)
    # A noisy simulated judge: right 90% of the time (deterministic per pair).
    judge = lambda i: bool(y[i]) if np.random.default_rng(i).random() < 0.9 else (not bool(y[i]))
    res = evaluate_scores("synthetic (self-test, NOT a real result)", cos, ce, y, seed, judge_fn=judge)
    meta = {"mode": "SELFTEST on synthetic data - these numbers mean nothing about real performance"}
    print(format_markdown([res], meta))


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="qqp,paws")
    ap.add_argument("--n", type=int, default=2000, help="pairs sampled per dataset (half calibrate, half evaluate)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-precision", type=float, default=0.99)
    ap.add_argument("--recall-floor", type=float, default=0.95)
    ap.add_argument("--judge", action="store_true", help="also evaluate LLM verification (needs GROQ_API_KEY)")
    ap.add_argument("--judge-limit", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args.seed)
        return

    from sentence_transformers import SentenceTransformer, CrossEncoder
    os.makedirs(args.out, exist_ok=True)
    bi = SentenceTransformer(BI_ENCODER_NAME)
    ce_model = CrossEncoder(CROSS_ENCODER_NAME)

    all_results, judge_model = [], None
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if name not in LOADERS:
            sys.exit(f"Unknown dataset '{name}'. Choices: {', '.join(LOADERS)}")
        print(f"\n=== {name}: loading and scoring ===")
        pairs = LOADERS[name](args.n, args.seed)
        labels = [p[2] for p in pairs]
        categories = [p[3] for p in pairs] if len(pairs[0]) > 3 else None
        cos, ce = score_pairs(pairs, bi, ce_model)

        judge_fn, save = None, None
        if args.judge:
            judge_fn, save, judge_model = make_llm_judge(pairs, os.path.join(args.out, f"judge_cache_{name}.json"))
        res = evaluate_scores(name, cos, ce, labels, args.seed, judge_fn, args.judge_limit,
                              args.target_precision, args.recall_floor, categories)
        if save:
            save()
        all_results.append(res)

    meta = {
        "date (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "seed": args.seed,
        "pairs sampled per dataset": args.n,
        "bi-encoder": BI_ENCODER_NAME,
        "cross-encoder": CROSS_ENCODER_NAME,
        "LLM judge": judge_model if args.judge else "not run",
        "target precision / recall floor": f"{args.target_precision} / {args.recall_floor}",
        "sentence-transformers": _version("sentence-transformers"),
        "datasets": _version("datasets"),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)
    md = format_markdown(all_results, meta)
    with open(os.path.join(args.out, "RESULTS.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Saved to {args.out}/RESULTS.md and results.json")


if __name__ == "__main__":
    main()
