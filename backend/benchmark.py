"""
benchmark.py

Runs a fixed, repeatable workload against the gateway and computes real
efficiency, performance, and cost metrics - satisfying the report's
requirement to report MEASURED, not assumed, numbers.

Usage:
    python3 benchmark.py --host http://127.0.0.1:8000 --requests 100

Requires: pip install requests
"""

import argparse
import random
import statistics
import os
import time
import requests

# Reused across runs so hit rate is measurable and reproducible.
CLUSTERS = [
    ["What is the capital of India?", "Which city is the capital of India?", "Tell me India's capital city"],
    ["How do I reset my password?", "I forgot my password, what do I do?", "Password reset steps please"],
    ["What are your business hours?", "When are you open?", "What time do you open and close?"],
    ["Can I get a refund?", "Is a refund possible for this?", "How do refunds work here?"],
    ["Is there a free trial?", "Can I try this for free?", "Do you offer a trial period?"],
]

# Rough token-cost assumption for reporting purposes only. Adjust to your
# actual provider's published per-token price for a real cost figure.
ASSUMED_TOKENS_PER_REQUEST = 150
ASSUMED_PRICE_PER_1M_TOKENS = 0.05  # USD, e.g. Llama 3.1 8B on Groq


def percentile(data, pct):
    if not data:
        return None
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


def run_benchmark(host: str, n_requests: int, clear_cache_first: bool):
    if clear_cache_first:
        requests.delete(f"{host}/cache")
        print("Cache cleared before benchmark run.\n")

    hit_latencies = []
    miss_latencies = []
    result_counts = {"exact_hit": 0, "semantic_hit": 0, "semantic_verified": 0, "miss": 0}
    errors = 0

    for i in range(n_requests):
        cluster = random.choice(CLUSTERS)
        question = random.choice(cluster)

        start = time.perf_counter()
        try:
            resp = requests.post(f"{host}/chat", headers=(
                {"X-API-Key": os.environ["GATEWAY_API_KEY"]} if os.environ.get("GATEWAY_API_KEY") else {}), json={
                "message": question,
                "system_prompt": "You are a helpful assistant.",
                "temperature": 0,
                "feature": "benchmark",
                "tag": "default"
            }, timeout=30)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                errors += 1
                continue

            data = resp.json()
            hit_type = data.get("hit_type")
            cache_hit = data.get("cache_hit")

            if cache_hit:
                hit_latencies.append(elapsed_ms)
                key = hit_type if hit_type in result_counts else "semantic_hit"
                result_counts[key] += 1
            else:
                miss_latencies.append(elapsed_ms)
                result_counts["miss"] += 1

        except requests.RequestException as e:
            errors += 1
            print(f"Request {i} failed: {e}")

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{n_requests} requests sent")

    return hit_latencies, miss_latencies, result_counts, errors


def report(hit_latencies, miss_latencies, result_counts, errors, n_requests):
    total_hits = sum(v for k, v in result_counts.items() if k != "miss")
    total = sum(result_counts.values())
    hit_rate = total_hits / total if total else 0

    print("\n" + "=" * 60)
    print("EFFICIENCY METRICS")
    print("=" * 60)
    print(f"Total requests sent:     {n_requests}")
    print(f"Errors:                  {errors}")
    print(f"Exact hits:              {result_counts['exact_hit']}")
    print(f"Semantic hits:           {result_counts['semantic_hit']}")
    print(f"Verified semantic hits:  {result_counts['semantic_verified']}")
    print(f"Misses (live LLM calls): {result_counts['miss']}")
    print(f"Overall hit rate:        {hit_rate:.1%}")

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS (measured, not assumed)")
    print("=" * 60)
    if hit_latencies:
        print(f"Cache-hit latency  — P50: {percentile(hit_latencies, 50):.1f}ms  "
              f"P95: {percentile(hit_latencies, 95):.1f}ms  "
              f"P99: {percentile(hit_latencies, 99):.1f}ms")
    else:
        print("No cache hits recorded — run again after the cache warms up.")

    if miss_latencies:
        print(f"Cache-miss latency — P50: {percentile(miss_latencies, 50):.1f}ms  "
              f"P95: {percentile(miss_latencies, 95):.1f}ms  "
              f"P99: {percentile(miss_latencies, 99):.1f}ms")

    if hit_latencies and miss_latencies:
        avg_hit = statistics.mean(hit_latencies)
        avg_miss = statistics.mean(miss_latencies)
        reduction = (avg_miss - avg_hit) / avg_miss * 100
        print(f"\nAverage latency reduction from caching: {reduction:.1f}%")
        print(f"  (avg miss: {avg_miss:.1f}ms  ->  avg hit: {avg_hit:.1f}ms)")

    print("\n" + "=" * 60)
    print("COST METRICS (illustrative — adjust price/token constants for your provider)")
    print("=" * 60)
    cost_per_request = (ASSUMED_TOKENS_PER_REQUEST / 1_000_000) * ASSUMED_PRICE_PER_1M_TOKENS
    calls_avoided = total_hits
    cost_avoided = calls_avoided * cost_per_request
    print(f"Assumed cost per LLM call:  ${cost_per_request:.6f}")
    print(f"LLM calls avoided by cache: {calls_avoided}")
    print(f"Estimated cost avoided:     ${cost_avoided:.4f}")
    print("(Scale this up: at 1M requests/month with the same hit rate,")
    print(f" estimated monthly savings would be ~${cost_avoided * (1_000_000 / max(n_requests,1)):.2f})")

    print("\n" + "=" * 60)
    print("NOTE ON FALSE-HIT RATE")
    print("=" * 60)
    print("This benchmark cannot automatically detect false hits (a cached")
    print("answer that doesn't actually fit the new question) without human")
    print("or LLM-judge review. Recommended: manually review a random sample")
    print("of 'semantic_verified' events from /events and score correctness.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--no-clear", action="store_true", help="Don't clear the cache before running")
    args = parser.parse_args()

    print(f"Running benchmark: {args.requests} requests against {args.host}\n")
    hit_lat, miss_lat, counts, errs = run_benchmark(args.host, args.requests, not args.no_clear)
    report(hit_lat, miss_lat, counts, errs, args.requests)
