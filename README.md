# Semantic Cache Gateway- for LLM APIs

A proxy gateway that sits between an application and an LLM provider,
recognizes semantically similar requests, and reuses cached responses
instead of re-calling the LLM for every request — reducing cost and
latency while guarding against unsafe cache reuse.
-
## Architecture

```
 OpenAI SDK ──► /v1/chat/completions ─┐
 Anthropic SDK ► /v1/messages ────────┼─► Auth (API key → tenant) ─► Engine ─► Provider router ─► OpenAI / Anthropic / Groq
 Legacy / dashboard ► /chat ──────────┘                              │
                                                                     ├─► Redis: entries, vector-index versions, budgets, metrics
 React dashboard ◄── /admin/report, /admin/chargeback ◄──────────────┘
```

Request flow (identical for every endpoint, streaming or not):

1. **Authenticate** — API key → tenant (+ business unit, budget limits).
2. **Normalize** — namespace = model + system prompt + temperature + feature + tag
   (+ tenant, + earlier conversation turns, + tool set, + user_id for `user_specific`).
3. **Exact match** — one Redis GET.
4. **Semantic search** — embed the last user message, one numpy matrix–vector
   product against the namespace's in-process index (rebuilt only when the
   Redis version counter changes), top-K → cross-encoder re-rank.
5. **Three-zone decision** — `auto_hit` / `borderline` (LLM verification) / `miss`.
6. **Isolation check** — a hit is refused unless the entry's stored tenant
   (and user, for `user_specific`) matches the request. Refusals are counted.
7. **On a miss** — check the tenant's budget, call the provider (streaming or
   not), store the answer + embedding under the policy TTL.
8. **Record telemetry** — outcome, tokens, dollars, lookup overhead, latency, TTFT.

Any failure in steps 3–4, 7 (Redis timeout, index error) is caught, counted as
a **fail-open event**, and the request is served live. A circuit breaker stops
retrying a dead Redis on every request.

## Why cosine alone isn't enough, and what the cross-encoder adds

`calibrate_threshold.py` measures similarity scores from BOTH the
bi-encoder (cosine) and the cross-encoder against the same golden set
of paraphrase / non-paraphrase pairs, and prints the separation gap for
each side by side. The original cosine-only measurement showed real
overlap: paraphrases scored as low as ~0.49, non-paraphrases as high as
~0.67 — meaning no fixed cutoff could cleanly separate them.

A cross-encoder scores both questions jointly instead of comparing two independently computed vectors, which in principle lets it use interactions between them. **Whether that helps is only partly supported by measurement.** In one run of `evaluation/run_eval.py` (seed 42, 2,000 sampled pairs per dataset, 1,000 held-out):

- **QQP:** the cross-encoder separates duplicates from non-duplicates much better than cosine (ROC-AUC 0.97 vs 0.87). QQP is probably contaminated, though, because the model was trained on Quora duplicate data, so this is an optimistic upper bound.
- **PAWS** (harder, out-of-distribution): neither method is useful. The cross-encoder did *not* beat cosine (ROC-AUC 0.57 vs 0.64, where 0.5 is chance), and neither beat the trivial strategy of calling every pair a duplicate on F1.
- The LLM-verification stage has not been evaluated yet.

Treat the cross-encoder's advantage as **unproven outside Quora-style questions**. The "capital of India vs capital of Japan" failure mode that motivated the design is not directly tested by either public dataset.

The tradeoff is speed: a cross-encoder is too slow to run against every
cached entry, so the architecture keeps cosine similarity as a fast
first-pass filter (`EngineConfig.top_k` candidates, default 5), and only
runs the cross-encoder on that narrowed set — cheap filter first,
expensive-but-accurate judge second, same layered principle already
used for the LLM borderline verification.

Run `calibrate_threshold.py` after any model change and update the
thresholds in `config.json` to match — the cross-encoder's output scale
is not the same as raw cosine similarity, so thresholds tuned for one
do not transfer directly to the other.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | Accept a chat-completion-style request (message, system_prompt, temperature, model, feature, tag, user_id) |
| FR2 | Normalize requests into a namespace from model + system_prompt + temperature + feature + tag (+ user_id) |
| FR3 | Exact-match cache checked before semantic search |
| FR4 | Semantic similarity search against cached entries in the same namespace |
| FR5 | Per-tag policy: auto-hit threshold, borderline threshold, TTL, cache_allowed |
| FR6 | Lightweight LLM verification on borderline-zone matches before reuse |
| FR7 | On a miss, call the LLM provider and cache the response + embedding |
| FR8 | Log every decision with similarity score, feature, tag, timestamp |
| FR9 | Expose `/stats`, `/events`, `/metrics` (Prometheus), `/policies` |
| FR10 | Expose `/cache` DELETE for invalidation, filterable by feature / tag / model / tenant / time range |
| FR11 | Never serve a cached response across two different `user_id`s under `user_specific` |
| FR12 | Provider-agnostic LLM calls (swap model/provider without code changes) |
| FR13 | Frontend: chat with live hit/miss visibility, dashboard, event log, policy viewer |

## Production upgrade: what was added (and where)

| Spec item | Implementation | Code |
|---|---|---|
| **Token in/out savings, provider pricing** | Per-request saved/spent tokens priced from a longest-prefix price table, overridable via `pricing.json` | `gateway/pricing.py`, `telemetry.py` |
| **Cache infrastructure ROI** | `Net_ROI = saved$ − (embedding cost + vector-DB hosting)`; embedding cost/1k and hosting $/hour are configured, hosting is accrued over uptime | `telemetry.report()` → `cost` |
| **Multi-tenant cost attribution** | API key → tenant + business unit; counters kept per tenant, unit, feature, key; `/admin/chargeback` | `tenants.py`, `telemetry.py` |
| **Cache lookup overhead (15–20 ms)** | Embedding + vector match timed separately from LLM verification; P50/P95/P99, SLO-breach counter, `slo_met_p95` flag | `engine._lookup`, `telemetry.latency_block` |
| **P95/P99 reduction** | Cached vs live latency percentiles and their difference | `telemetry.latency_block` |
| **Gateway throughput (RPS)** | Measured with `loadtest.py` (Locust) against `/v1/*`; stateless workers scale via `WEB_CONCURRENCY` / replicas | `loadtest.py` |
| **False-positive rate** | Human feedback (`POST /v1/feedback`) + optional sampled LLM-judge audits (`JUDGE_SAMPLE_RATE`) | `telemetry.feedback`, `engine._judge_audit` |
| **Threshold drift coefficient** | Recent-vs-older near-miss similarity and verified-rate comparison, with advisory text (never auto-changes thresholds) | `telemetry.threshold_drift` |
| **Semantic divergence index** | Sampled live re-runs (`REVALIDATE_SAMPLE_RATE`) compared to the cached answer (token-overlap), plus mean entry-age/TTL at hit | `engine._revalidate`, `telemetry.revalidation` |
| **Fail-open trigger count** | Counter + recent-reason log; circuit breaker | `resilience.py`, `engine._try_cache` |
| **Cross-tenant isolation blocks** | Tenant is in the namespace *and* verified on the entry at serve time; blocks counted + logged | `engine._scope_ok` |
| **TTFT (cached vs live streams)** | Measured at the first chunk handed to the client, per path | `engine._replay_iter/_live_iter` |
| **Cross-provider SDK compatibility** | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, one shared cache, errors in each SDK's native shape; any model routes to any provider | `adapters.py`, `providers.py` |
| **Streaming-aware replay** | Live SSE is proxied and assembled; on completion the answer is stored. Hits are replayed as native SSE in the caller's dialect (OpenAI chunks / Anthropic events). Partial streams are never cached | `engine.handle_stream`, `adapters.py` |
| **Dynamic token budgeting & throttling** | Per-tenant RPM, daily tokens, daily/monthly USD. At 80% of any cap RPM drops to 25%; at 100% live calls get `429 + Retry-After`. Cache hits are always served (they cost nothing) | `budget.py` |

Also hardened: non-root image with models baked in, password-protected unexposed
Redis with a memory cap, admin API behind a key, CORS allow-list, `KEYS` scan
replaced with an indexed search, rule-based invalidation, multi-turn context in
the cache key, tool-call requests forwarded but never cached.

### Quick start (production-style)

```bash
cp .env.example .env                      # fill in provider key(s), ADMIN_API_KEY, REDIS_PASSWORD
cd backend && python -m gateway.tenants hash sk-my-secret-key      # -> sha256 hex
# put that hash in backend/tenants.json (see tenants.example.json), then:
cd .. && docker compose up --build
```

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-my-secret-key")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

# Anthropic SDK
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="sk-my-secret-key")
client.messages.create(model="claude-sonnet-4-5", max_tokens=200, messages=[{"role": "user", "content": "hi"}])
```
Per-request cache controls: headers `X-Cache-Feature`, `X-Cache-Tag`, `X-User-Id`
(or `metadata` in the body). Responses carry `X-Cache: HIT|MISS`, `X-Cache-Type`,
`X-Cache-Similarity`, `X-Cache-Saved-USD`, `X-Request-Id`, `X-Budget-State`.
To flag a bad hit: `POST /v1/feedback {"request_id": "...", "bad": true}`.
For local dev without auth set `AUTH_REQUIRED=false` and leave `tenants.json` as `{}`.

## Non-Functional Requirements

| ID | Requirement | Implementation |
|----|-------------|----------------|
| NFR1 | Performance — low-tens-of-ms lookups | Exact match = one GET; semantic = one matrix product on an in-process index. **Measured**, not assumed: see `latency.lookup_overhead_ms` in `/admin/report`. The cross-encoder dominates this cost on CPU — set `RERANK_ENABLED=false` if you exceed `LOOKUP_SLO_MS` (then re-calibrate thresholds) |
| NFR2 | Reliability — fail open | Every cache/budget/telemetry store call is guarded; circuit breaker; verified by a test in which every Redis operation raises |
| NFR3 | Security — no cross-user/tenant leakage | Tenant + user in the namespace, re-verified at serve time, blocks counted; API-key auth; hashed keys on disk |
| NFR4 | Configurability | `config.json` (policies), `pricing.json`, `tenants.json`, env vars; no redeploy of code |
| NFR5 | Observability | Redis event log, `/admin/report`, Prometheus `/metrics` |
| NFR6 | Scalability | Gateway is stateless; all shared state is in Redis. See Known Limitations for the index ceiling |
| NFR7 | Model drift | Startup check of the default model against the provider list, with fallback |
| NFR8 | Testability | Engine, budgets, adapters, telemetry run against in-memory fakes — no network/Redis/models |
| NFR9 | Deployability | Compose file, non-root image, health (`/health`) and readiness (`/ready`) probes |

## Known Limitations (honest, not hidden)

- **Not yet validated against live providers or a live Redis in this
  environment.** The pipeline logic is covered by unit tests with fakes
  (including SSE framing for both dialects); the real `httpx` provider calls,
  real SSE parsing against OpenAI/Anthropic, `redis-py` behaviour, the React
  dashboard build and `tests/test_api.py` (FastAPI `TestClient`) must be run
  by you: `pytest`, then a smoke test with real keys.
- **The in-process vector index is brute-force numpy per namespace.** Sub-ms
  for thousands of entries, and each replica holds its own copy, but it is
  not a distributed ANN index. Beyond roughly tens of thousands of entries per
  namespace, move to Redis Stack vector search or Qdrant behind `CacheIndex`.
- **`MAX_ENTRIES_PER_NS` caps memory** per namespace (soonest-to-expire evicted first).
- **Re-ranking latency.** The cross-encoder adds tens of ms on CPU, which can
  exceed the 15–20 ms lookup budget. The measured number is on the dashboard.
- **Dollar figures depend on the price table.** `pricing.json` /
  `gateway/pricing.py` ship illustrative defaults: verify them against your
  providers' current price sheets. Token counts use the provider's reported
  usage when present, otherwise a ~4 chars/token estimate.
- **Verification / judge / revalidation calls are not charged to tenant
  budgets** (they are gateway operating cost). Revalidation spend is tracked in
  `tm:global revalidation_spend_usd`.
- **Divergence index uses token overlap**, a crude proxy; use the LLM-judge
  audit for a stricter signal. **Threshold drift is advisory only.**
- **Tool-calling requests are forwarded, never cached**, and only when the
  endpoint dialect matches the target provider (e.g. `/v1/messages` → `claude-*`).
  Cross-dialect tool translation is not implemented (returns 400).
- **Cached text only.** Non-text content blocks (images, tool-use blocks) in
  responses are not cached; image/audio inputs are flattened to their text parts.
- **Semantic matching accuracy limits** are unchanged and documented above and
  in `evaluation/README.md`; the cross-encoder's benefit is unproven outside
  Quora-style questions.
- **Temperature > 0 responses are cacheable** (temperature is part of the key),
  which freezes one sample of a non-deterministic answer. Set a policy with
  `cache_allowed: false` for features where that is unacceptable.

## Project Structure

```
.
├── docker-compose.yml          # gateway + password-protected Redis (no published port)
├── .env.example
├── backend/
│   ├── main.py                 # FastAPI: routes, auth, SSE framing, Prometheus, admin API
│   ├── cache_logic.py          # Pure decision logic (namespace, hashing, 3-zone classifier)
│   ├── gateway/
│   │   ├── engine.py           # Request pipeline (framework-free, fully unit-tested)
│   │   ├── adapters.py         # OpenAI + Anthropic schemas, SSE build/parse/replay
│   │   ├── providers.py        # OpenAI / Anthropic / Groq clients, retries, routing
│   │   ├── index.py            # Entry store + versioned in-process vector index, invalidation
│   │   ├── budget.py           # RPM / token / USD caps, soft throttle, hard deny
│   │   ├── tenants.py          # API-key auth, per-tenant limits
│   │   ├── telemetry.py        # ROI, latency, quality, reliability, per-tenant metrics
│   │   ├── pricing.py          # Price table, savings & net-ROI math
│   │   ├── resilience.py       # Circuit breaker
│   │   └── memstore.py         # In-memory Redis stand-in for tests/dev
│   ├── config.json  pricing.json  tenants.example.json
│   ├── loadtest.py  benchmark.py  calibrate_threshold.py
│   ├── evaluation/             # Public-dataset evaluation (QQP, PAWS) - see its README
│   ├── requirements.txt  requirements-dev.txt  Dockerfile
│   └── tests/                  # cache_logic, adapters, budgets, telemetry, engine, api, eval
└── frontend/                   # Chat / Dashboard (measured metrics) / Events / Policies
```

## Running it

### Option A: Docker Compose (recommended)

```bash
cp .env.example .env
# edit .env and add your real GROQ_API_KEY

docker compose up --build
```

Gateway: http://127.0.0.1:8000 · Docs: http://127.0.0.1:8000/docs

Then, separately, run the frontend:
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

### Option B: Run locally without Docker

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# Start Redis separately (must be running on localhost:6379)
redis-server

export GROQ_API_KEY="your_key_here"
export AUTH_REQUIRED=false ADMIN_API_KEY=dev-admin   # local dev only
uvicorn main:app --reload
```

**Frontend** (new terminal):
```bash
cd frontend
npm install
npm run dev
```

### Verifying your Groq model is valid

Model availability on Groq's free tier changes over time. Check what's
currently available to your key:

```bash
python3 -c "
from groq import Groq
import os
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
for m in client.models.list().data:
    print(m.id)
"
```

Set `DEFAULT_LLM_MODEL` in `.env` (or as an environment variable) to
one of the printed IDs. The gateway also auto-detects and falls back
at startup if the configured model isn't available (NFR7).

### Running tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v          # test_api.py needs fastapi+httpx (in requirements.txt)
```

### Calibrating thresholds

```bash
cd backend
python3 calibrate_threshold.py
```

This measures BOTH the cosine (bi-encoder) and cross-encoder scores
against the same golden pairs, prints the separation gap for each side
by side, and suggests cross-encoder thresholds to put in `config.json`.
Since the two methods use different output scales, don't reuse
cosine-tuned thresholds for the cross-encoder stage.

### Evaluating matching accuracy on public datasets

`backend/evaluation/` measures the matching stage on labelled duplicate-question
data (QQP, PAWS) and compares cosine-only, cross-encoder, and cross-encoder +
LLM-verification, reporting **false-hit rate** (safety) against **duplicates
caught** (savings) with confidence intervals. Thresholds are chosen on one half
of each sample and reported on the other.

```bash
cd backend
pip install -r requirements.txt -r evaluation/requirements-eval.txt
python3 evaluation/run_eval.py --datasets qqp,paws --n 2000 --judge
```

It also includes `hardneg`, a synthetic set of near-identical-but-different question pairs
(entity, number, attribute, and polarity swaps) with a per-type breakdown. Run
`python3 evaluation/run_eval.py --datasets hardneg` (no API key needed).

No results are published in this repository yet. Run it and read
`evaluation/README.md` (especially the data-contamination caveat for QQP)
before citing any number.

### Load testing

```bash
cd backend
locust -f loadtest.py --host=http://127.0.0.1:8000
```
Open http://localhost:8089, set users/spawn rate, click Start.

### Measuring performance, efficiency, and worthiness

`benchmark.py` runs a fixed, repeatable workload and reports **measured**
numbers across four categories, rather than assumed ones:

```bash
cd backend
python3 benchmark.py --host http://127.0.0.1:8000 --requests 100
```

It reports:
- **Efficiency**: hit rate, exact vs semantic split, verification rate
- **Performance**: P50/P95/P99 latency for cache hits vs live LLM calls, and the measured latency reduction %
- **Cost**: LLM calls avoided and estimated $ saved (token-price constants at the top of the script — adjust to your actual provider pricing)
- **A note on false-hit rate**, which requires human/LLM-judge review of a sample of `semantic_verified` events (`/events`) since it can't be measured automatically

Run it twice — once right after `docker-compose up` (cold cache, mostly
misses) and once again immediately after (warm cache, mostly hits) — to
see the before/after contrast clearly for your report.

#### Metrics reference

| Category | Metric | Meaning |
|---|---|---|
| Efficiency | Hit rate | % of requests served from cache |
| Efficiency | Exact vs semantic split | How much value comes from paraphrase-matching vs literal repeats |
| Efficiency | Verification rate | % of requests that hit the borderline safety check |
| Performance | P50/P95/P99 latency (hit vs miss) | Real responsiveness difference |
| Performance | Latency reduction % | (miss_latency − hit_latency) / miss_latency |
| Cost | Calls avoided × price/token | Real (if approximate) dollar savings |
| Correctness | False-hit rate | Requires manual/LLM-judge sampling of near-miss and verified events — not auto-computed |
| Correctness | Cross-user leakage incidents | Should always be 0 — covered by unit tests in `tests/test_cache_logic.py` |


## API Reference

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/v1/chat/completions` | POST | gateway key | OpenAI-compatible (streaming, tools) |
| `/v1/messages` | POST | gateway key | Anthropic-compatible (streaming, tools) |
| `/chat` | POST | gateway key | Original native endpoint |
| `/v1/feedback` | POST | gateway key | Flag a cache hit as bad/good (`request_id`, `bad`) |
| `/admin/report` | GET | `X-Admin-Key` | Full ROI / latency / quality / reliability / per-tenant report |
| `/admin/chargeback` | GET | `X-Admin-Key` | Per-business-unit spend vs savings |
| `/admin/budget/{tenant}` | GET | `X-Admin-Key` | Limits and live usage |
| `/cache` | DELETE | `X-Admin-Key` | Invalidate by `feature`, `tag`, `model`, `tenant`, `older_than_seconds`, `newer_than_seconds` (AND-combined; none = all) |
| `/metrics` | GET | — | Prometheus (restrict at the network layer) |
| `/health` | GET | — | Liveness (never touches Redis) |
| `/ready` | GET | — | Readiness + `normal` / `fail-open` mode |
| `/stats` `/events` `/policies` | GET | — | Summary, decision log, active policies |

Gateway keys go in `Authorization: Bearer …` or `x-api-key`, so unmodified SDKs work.


//claude
  
//openai
# Semantic-cache-gateway-for-LLM-API-s
