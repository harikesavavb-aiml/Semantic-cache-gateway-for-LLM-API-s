"""
main.py - Semantic Cache Gateway for LLM APIs (production edition)

HTTP surface (thin glue; all logic lives in gateway/engine.py):

  POST /v1/chat/completions   OpenAI-compatible   (streaming + tools)
  POST /v1/messages           Anthropic-compatible (streaming + tools)
  POST /chat                  original native endpoint (kept for the dashboard / back-compat)
  POST /v1/feedback           flag a cache hit as good/bad (false-positive tracking)
  GET  /admin/report          ROI, latency, quality, reliability, per-tenant metrics
  GET  /admin/chargeback      per-business-unit spend vs savings
  GET  /admin/budget/{tenant} live budget usage
  DELETE /cache               rule-based invalidation (feature/tag/model/tenant/age)
  GET  /metrics /health /ready /stats /events /policies

Point an existing SDK at the gateway by swapping its base URL:
  OpenAI:    OpenAI(base_url="http://gateway:8000/v1", api_key="<gateway key>")
  Anthropic: Anthropic(base_url="http://gateway:8000", api_key="<gateway key>")
"""
import hmac
import json
import logging
import os
import time
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from cache_logic import load_policies
from gateway import adapters
from gateway.budget import BudgetManager
from gateway.engine import Engine, EngineConfig, GatewayError, Result, make_llm_verifier
from gateway.index import CacheIndex
from gateway.pricing import load_prices
from gateway.providers import build_router_from_env
from gateway.resilience import CircuitBreaker
from gateway.telemetry import Telemetry
from gateway.tenants import PUBLIC, authenticate, extract_key, load_tenants

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("gateway")
env = os.environ.get

app = FastAPI(title="Semantic Cache Gateway", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")],
    allow_methods=["*"], allow_headers=["*"],
    expose_headers=["X-Cache", "X-Cache-Type", "X-Cache-Similarity", "X-Request-Id", "X-Cache-Saved-USD",
                    "X-Budget-State", "Retry-After"],
)

# ------------------------------ configuration ------------------------------
POLICIES = load_policies()
PRICES = load_prices()
TENANTS = load_tenants()
AUTH_REQUIRED = env("AUTH_REQUIRED", "false").lower() == "true"
ADMIN_API_KEY = env("ADMIN_API_KEY", "")
DEFAULT_MODEL = env("DEFAULT_LLM_MODEL", "llama-3.1-8b-instant")
VERIFIER_MODEL = env("VERIFIER_MODEL", DEFAULT_MODEL)
FALLBACK_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it", "openai/gpt-oss-120b"]

# ------------------------------ Prometheus ---------------------------------
M_REQ = Counter("gateway_requests_total", "Requests by outcome", ["result", "dialect"])
M_SAVED = Counter("gateway_saved_usd_total", "LLM dollars avoided by cache hits", ["tenant"])
M_SPENT = Counter("gateway_spent_usd_total", "LLM dollars spent on live calls", ["tenant"])
M_SAVED_TOK = Counter("gateway_saved_tokens_total", "Tokens avoided", ["direction"])
M_LLM_LAT = Histogram("gateway_llm_latency_seconds", "Latency of live LLM calls")
M_CACHE_LAT = Histogram("gateway_cache_latency_seconds", "Latency of cache-served responses")
M_LOOKUP = Histogram("gateway_lookup_overhead_seconds", "Embedding + vector match time",
                     buckets=(.002, .005, .01, .015, .02, .03, .05, .1, .25))
M_TTFT = Histogram("gateway_ttft_seconds", "Time to first streamed token", ["source"])
M_FAILOPEN = Counter("gateway_fail_open_total", "Requests served live because the cache failed")
M_ISO = Counter("gateway_isolation_blocks_total", "Cross-tenant/user reuse blocked")
M_THROTTLE = Counter("gateway_budget_events_total", "Budget throttle/deny events", ["state"])
G_BREAKER = Gauge("gateway_breaker_open", "1 when the cache circuit breaker is open")

# ------------------------------ lazy singletons ----------------------------
_state = {"engine": None, "redis": None, "embed": None, "cross": None}


def get_embed_model():
    if _state["embed"] is None:
        from sentence_transformers import SentenceTransformer
        _state["embed"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _state["embed"]


def get_cross_encoder():
    """Cross-encoder re-ranker (see evaluation/README.md for measured benefit:
    strong on Quora-style pairs, unproven out-of-distribution)."""
    if _state["cross"] is None:
        from sentence_transformers import CrossEncoder
        _state["cross"] = CrossEncoder("cross-encoder/quora-distilroberta-base")
    return _state["cross"]


def embedder(text: str):
    return get_embed_model().encode([text], normalize_embeddings=True)[0]


def reranker(pairs):
    return [float(s) for s in get_cross_encoder().predict(pairs)]


def build_store():
    import redis
    url = env("REDIS_URL") or f"redis://{env('REDIS_HOST', 'localhost')}:6379/0"
    # short timeouts: a hung Redis must trip fail-open fast instead of stalling requests
    return redis.Redis.from_url(url, decode_responses=True,
                                socket_connect_timeout=float(env("REDIS_CONNECT_TIMEOUT", "0.5")),
                                socket_timeout=float(env("REDIS_SOCKET_TIMEOUT", "0.25")),
                                health_check_interval=30)


def get_engine() -> Engine:
    if _state["engine"] is None:
        store = build_store()
        _state["redis"] = store
        router = build_router_from_env()
        tm = Telemetry(store, embed_cost_per_1k=float(env("EMBED_COST_PER_1K", "0")),
                       vector_db_hourly_usd=float(env("VECTOR_DB_HOURLY_USD", "0")),
                       lookup_slo_ms=float(env("LOOKUP_SLO_MS", "20")))
        cfg = EngineConfig(
            policies=POLICIES, prices=PRICES, default_model=DEFAULT_MODEL,
            rerank_enabled=env("RERANK_ENABLED", "true").lower() == "true",
            judge_sample_rate=float(env("JUDGE_SAMPLE_RATE", "0")),
            revalidate_sample_rate=float(env("REVALIDATE_SAMPLE_RATE", "0")),
            replay_delay_ms=float(env("REPLAY_DELAY_MS", "0")))
        _state["engine"] = Engine(
            cfg=cfg, store=store, index=CacheIndex(store, max_per_ns=int(env("MAX_ENTRIES_PER_NS", "5000"))),
            embedder=embedder, reranker=reranker, router=router, telemetry=tm,
            budget=BudgetManager(store, soft_threshold=float(env("BUDGET_SOFT_THRESHOLD", "0.8")),
                                 throttle_factor=float(env("THROTTLE_FACTOR", "0.25"))),
            verifier=make_llm_verifier(router, VERIFIER_MODEL),
            breaker=CircuitBreaker(threshold=int(env("BREAKER_THRESHOLD", "3")),
                                   cooldown=float(env("BREAKER_COOLDOWN", "10"))))
    return _state["engine"]


@app.on_event("startup")
def startup():
    eng = get_engine()
    if env("PRELOAD_MODELS", "true").lower() == "true":
        try:                                   # pay model-load cost at boot, not on the first request
            embedder("warmup")
            if eng.cfg.rerank_enabled:
                reranker([("warmup", "warmup")])
        except Exception as e:
            log.warning("model preload failed (will load lazily): %s", e)
    if not env("OPENAI_API_KEY") and not env("ANTHROPIC_API_KEY") and not env("GROQ_API_KEY"):
        log.error("No provider API key configured (GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    if not TENANTS and AUTH_REQUIRED:
        log.error("AUTH_REQUIRED=true but no tenants configured: every request will be rejected")
    if not TENANTS and not AUTH_REQUIRED:
        log.warning("Running WITHOUT authentication (single 'public' tenant). Set AUTH_REQUIRED=true "
                    "and configure tenants.json for production.")
    if not ADMIN_API_KEY:
        log.warning("ADMIN_API_KEY not set: /admin/* and DELETE /cache are disabled.")
    _validate_default_model(eng)


def _validate_default_model(eng: Engine):
    """Model-drift guard (NFR7): if the default model has been retired by the
    provider, fall back to the first available candidate. Fail-open on error."""
    global DEFAULT_MODEL
    if not env("GROQ_API_KEY"):
        return
    try:
        import httpx
        r = httpx.get(env("GROQ_BASE_URL", "https://api.groq.com/openai/v1") + "/models",
                      headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"}, timeout=5)
        available = {m["id"] for m in r.json().get("data", [])}
        if DEFAULT_MODEL not in available:
            for cand in FALLBACK_MODELS:
                if cand in available:
                    log.warning("Model '%s' unavailable; falling back to '%s'", DEFAULT_MODEL, cand)
                    DEFAULT_MODEL = eng.cfg.default_model = cand
                    break
    except Exception as e:
        log.warning("Could not validate model list at startup: %s", e)


# ------------------------------ helpers ------------------------------------

def _err_body(dialect: str, status: int, message: str):
    if dialect == "anthropic":
        etype = {401: "authentication_error", 429: "rate_limit_error", 400: "invalid_request_error"}.get(status, "api_error")
        return {"type": "error", "error": {"type": etype, "message": message}}
    etype = {401: "invalid_api_key", 429: "rate_limit_exceeded", 400: "invalid_request_error"}.get(status, "api_error")
    return {"error": {"message": message, "type": etype, "code": etype}}


def _fail(dialect: str, status: int, message: str, retry_after: Optional[int] = None):
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return JSONResponse(_err_body(dialect, status, message), status_code=status, headers=headers)


def _auth(request: Request, dialect: str):
    tenant = authenticate(TENANTS, extract_key(dict(request.headers)), AUTH_REQUIRED)
    return tenant


def _build(dialect: str, body: dict, request: Request, tenant):
    headers = dict(request.headers)
    c = adapters.from_anthropic(body, headers) if dialect == "anthropic" else adapters.from_openai(body, headers)
    c.tenant_id = tenant.tenant
    c.extra.update(tenant_obj=None if tenant is PUBLIC else tenant, unit=tenant.unit,
                   key_id=tenant.key_id, raw_body=body)
    return c


def _observe(res: Result, c, ttft_ms=None):
    M_REQ.labels(res.outcome, c.dialect).inc()
    if res.cache_hit:
        M_CACHE_LAT.observe(res.latency_ms / 1000)
        M_SAVED.labels(c.tenant_id).inc(res.saved_usd)
        M_SAVED_TOK.labels("input").inc(res.in_tokens)
        M_SAVED_TOK.labels("output").inc(res.out_tokens)
    else:
        M_LLM_LAT.observe(res.latency_ms / 1000)
        M_SPENT.labels(c.tenant_id).inc(res.cost_usd)
    if res.lookup_ms is not None:
        M_LOOKUP.observe(res.lookup_ms / 1000)
    if res.outcome == "fail_open":
        M_FAILOPEN.inc()
    if res.budget_state != "ok":
        M_THROTTLE.labels(res.budget_state).inc()
    if ttft_ms is not None:
        M_TTFT.labels("cache" if res.cache_hit else "live").observe(ttft_ms / 1000)


def _cache_headers(res: Result) -> dict:
    h = {"X-Request-Id": res.request_id, "X-Cache": "HIT" if res.cache_hit else "MISS",
         "X-Budget-State": res.budget_state}
    if res.hit_type:
        h["X-Cache-Type"] = res.hit_type
    if res.similarity is not None:
        h["X-Cache-Similarity"] = str(res.similarity)
    if res.saved_usd:
        h["X-Cache-Saved-USD"] = f"{res.saved_usd:.8f}"
    return h


def _run_audits(res: Result):
    for fn in res.audits:
        fn()


def _prime(it):
    """Pull the first chunk before committing to a 200, so upstream/budget
    errors still become proper HTTP errors instead of a broken stream."""
    first = next(it, None)

    def chain():
        if first is not None:
            yield first
        yield from it
    return chain()


def _handle(dialect: str, body: dict, request: Request, bg: BackgroundTasks):
    try:
        tenant = _auth(request, dialect)
        if tenant is None:
            return _fail(dialect, 401, "Invalid or missing API key")
        c = _build(dialect, body, request, tenant)
    except (ValueError, KeyError, TypeError) as e:
        return _fail(dialect, 400, f"Invalid request: {e}")
    eng = get_engine()
    G_BREAKER.set(1 if eng.breaker.is_open else 0)
    try:
        if c.stream and c.tools:
            it = _prime(eng.handle_passthrough_stream(c))
            return StreamingResponse(it, media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Cache": "BYPASS"})
        if c.stream:
            info, text_iter = eng.handle_stream(c)
            text_iter = _prime(text_iter)
            frames = _frame_stream(dialect, text_iter, c, info)
            bg.add_task(_run_audits, info)
            return StreamingResponse(frames, media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", **_cache_headers(info)})
        res = eng.handle(c)
    except GatewayError as e:
        if e.status == 429:
            M_THROTTLE.labels("denied").inc()
        return _fail(dialect, e.status, str(e), e.retry_after)
    _observe(res, c)
    bg.add_task(_run_audits, res)
    if res.raw is not None:                       # tool call: return the provider's native body untouched
        return JSONResponse(res.raw, headers={**_cache_headers(res), "X-Cache": "BYPASS"})
    fn = adapters.to_anthropic_response if dialect == "anthropic" else adapters.to_openai_response
    return JSONResponse(fn(res.text, res.model, res.in_tokens, res.out_tokens), headers=_cache_headers(res))


def _frame_stream(dialect: str, text_iter, c, info: Result):
    """Wrap a text-delta iterator into the dialect's SSE framing, live."""
    import uuid
    model = c.model
    if dialect == "anthropic":
        mid = f"msg_{uuid.uuid4().hex[:24]}"
        yield adapters._sse({"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": info.in_tokens, "output_tokens": 0}}},
            "message_start")
        yield adapters._sse({"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}, "content_block_start")
        for d in text_iter:
            yield adapters._sse({"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": d}}, "content_block_delta")
        yield adapters._sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
        yield adapters._sse({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                             "usage": {"output_tokens": info.out_tokens}}, "message_delta")
        yield adapters._sse({"type": "message_stop"}, "message_stop")
    else:
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        yield adapters._sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                  "finish_reason": None}]})
        for d in text_iter:
            yield adapters._sse({**base, "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}]})
        yield adapters._sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": info.in_tokens, "completion_tokens": info.out_tokens,
                                       "total_tokens": info.in_tokens + info.out_tokens}})
        yield "data: [DONE]\n\n"
    _observe(info, c, ttft_ms=None)


# ------------------------------ API routes ---------------------------------

@app.post("/v1/chat/completions")
async def openai_chat(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
    except ValueError:
        return _fail("openai", 400, "Body must be valid JSON")
    return await _in_thread(_handle, "openai", body, request, bg)


@app.post("/v1/messages")
async def anthropic_messages(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
    except ValueError:
        return _fail("anthropic", 400, "Body must be valid JSON")
    return await _in_thread(_handle, "anthropic", body, request, bg)


async def _in_thread(fn, *a):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(fn, *a)


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.0
    model: Optional[str] = None
    feature: str = "default"
    tag: str = "default"
    user_id: Optional[str] = None


@app.post("/chat")
def chat(req: ChatRequest, request: Request, bg: BackgroundTasks):
    """Original endpoint, preserved for the dashboard and existing callers."""
    tenant = _auth(request, "openai")
    if tenant is None:
        raise HTTPException(401, "Invalid or missing API key")
    c = adapters.CanonicalRequest(
        dialect="native", model=req.model or "", system=req.system_prompt,
        messages=[{"role": "user", "content": req.message}], temperature=req.temperature,
        feature=req.feature, tag=req.tag, user_id=req.user_id, tenant_id=tenant.tenant,
        extra={"tenant_obj": None if tenant is PUBLIC else tenant, "unit": tenant.unit, "key_id": tenant.key_id})
    try:
        res = get_engine().handle(c)
    except GatewayError as e:
        raise HTTPException(e.status, str(e), headers={"Retry-After": str(e.retry_after)} if e.retry_after else None)
    _observe(res, c)
    bg.add_task(_run_audits, res)
    return {"answer": res.text, "cache_hit": res.cache_hit, "similarity": res.similarity,
            "matched_question": res.matched_question, "hit_type": res.hit_type, "note": res.note,
            "request_id": res.request_id, "outcome": res.outcome, "latency_ms": round(res.latency_ms, 1),
            "lookup_ms": round(res.lookup_ms, 1) if res.lookup_ms is not None else None,
            "saved_usd": res.saved_usd, "cost_usd": res.cost_usd,
            "tokens": {"input": res.in_tokens, "output": res.out_tokens}}


class Feedback(BaseModel):
    request_id: str
    bad: bool


@app.post("/v1/feedback")
def feedback(fb: Feedback, request: Request):
    if _auth(request, "openai") is None:
        raise HTTPException(401, "Invalid or missing API key")
    if not get_engine().tm.feedback(fb.request_id, fb.bad, source="human"):
        raise HTTPException(404, "Unknown or expired request_id (only cache hits can be flagged)")
    return {"ok": True}


# ------------------------------ admin ---------------------------------------

def _admin(x_admin_key: Optional[str]):
    if not ADMIN_API_KEY:
        raise HTTPException(503, "Admin API disabled: set ADMIN_API_KEY")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(401, "Invalid admin key")


@app.get("/admin/report")
def admin_report(x_admin_key: Optional[str] = Header(None)):
    _admin(x_admin_key)
    return get_engine().tm.report()


@app.get("/admin/chargeback")
def admin_chargeback(x_admin_key: Optional[str] = Header(None)):
    _admin(x_admin_key)
    return {"units": get_engine().tm.chargeback()}


@app.get("/admin/budget/{tenant_id}")
def admin_budget(tenant_id: str, x_admin_key: Optional[str] = Header(None)):
    _admin(x_admin_key)
    for t in TENANTS.values():
        if t.tenant == tenant_id:
            return {"tenant": t.tenant, "limits": {"rpm": t.rpm, "daily_tokens": t.daily_tokens,
                                                   "daily_usd": t.daily_usd, "monthly_usd": t.monthly_usd},
                    "usage": get_engine().budget.usage(t)}
    raise HTTPException(404, "Unknown tenant")


@app.delete("/cache")
def clear_cache(feature: Optional[str] = None, tag: Optional[str] = None, model: Optional[str] = None,
                tenant: Optional[str] = None, older_than_seconds: Optional[int] = None,
                newer_than_seconds: Optional[int] = None, x_admin_key: Optional[str] = Header(None)):
    """Rule-based invalidation. Filters combine with AND; no filter clears everything."""
    _admin(x_admin_key)
    now = time.time()
    n = get_engine().index.invalidate(
        feature=feature, tag=tag, model=model, tenant=tenant,
        created_before=now - older_than_seconds if older_than_seconds else None,
        created_after=now - newer_than_seconds if newer_than_seconds else None)
    return {"cleared": n}


# ------------------------------ ops / compatibility -------------------------

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    """Liveness: the process is up. Never touches Redis, so a cache outage
    doesn't get the container restarted (the gateway is designed to run through it)."""
    return {"status": "ok", "active_model": DEFAULT_MODEL}


@app.get("/ready")
def ready():
    eng = get_engine()
    try:
        _state["redis"].ping()
        cache = True
    except Exception:
        cache = False
    return {"status": "ready", "cache_up": cache, "breaker_open": eng.breaker.is_open,
            "mode": "normal" if cache and not eng.breaker.is_open else "fail-open"}


@app.get("/stats")
def stats():
    r = get_engine().tm.report()
    o, c, l = r["overview"], r["cost"], r["latency"]
    return {"redis_up": not get_engine().breaker.is_open, "active_model": DEFAULT_MODEL,
            "total_requests": o["requests"], "exact_hits": o["exact_hits"], "semantic_hits": o["semantic_hits"],
            "misses": o["misses"], "near_misses": r["quality"]["threshold_drift"]["samples"],
            "hit_rate": o["hit_rate"], "saved_usd": c["saved_usd"], "net_roi_usd": c["net_roi_usd"],
            "p95_cached_ms": l["cached_ms"]["p95"], "p95_live_ms": l["live_ms"]["p95"],
            "total_cached_entries": get_engine().index.count() if not get_engine().breaker.is_open else None}


@app.get("/events")
def recent_events(limit: int = 20):
    try:
        return {"events": [json.loads(e) for e in _state["redis"].lrange("events:log", 0, limit - 1)]}
    except Exception:
        return {"events": []}


@app.get("/policies")
def get_policies():
    return POLICIES
