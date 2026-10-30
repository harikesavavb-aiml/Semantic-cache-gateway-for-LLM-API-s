"""
engine.py - the request pipeline, independent of any web framework.

    normalize -> exact lookup -> semantic lookup (+ rerank) -> three-zone policy
    -> isolation checks -> (borderline verification) -> serve | live call
    -> store -> telemetry

Dependencies are injected (store, index, embedder, reranker, router, budget,
telemetry), so the full pipeline - including fail-open, tenant isolation,
streaming replay and budget enforcement - is unit-tested with fakes.

Cache failures NEVER fail a request: any exception from the cache layer is
caught, counted as a fail-open event, and the request is served live.
"""
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from cache_logic import classify_similarity, exact_hash, get_policy, make_namespace
from .adapters import CanonicalRequest, chunk_text
from .budget import BudgetManager
from .index import CacheIndex
from .pricing import call_cost, estimate_tokens
from .providers import ProviderError, ProviderRouter
from .resilience import CircuitBreaker
from .telemetry import Telemetry

log = logging.getLogger("gateway.engine")

VERIFY_PROMPT = """Question A: "{matched_question}"
Question B: "{question}"
Answer given for Question A: "{candidate_answer}"

Would this same answer also correctly and fully answer Question B?
Reply with only one word: YES or NO."""


class GatewayError(Exception):
    def __init__(self, status: int, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


@dataclass
class Result:
    request_id: str
    text: str
    model: str
    outcome: str                      # exact_hit|semantic_hit|semantic_verified|miss|bypass|fail_open
    in_tokens: int = 0
    out_tokens: int = 0
    similarity: Optional[float] = None
    matched_question: Optional[str] = None
    note: Optional[str] = None
    saved_usd: float = 0.0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    lookup_ms: Optional[float] = None
    budget_state: str = "ok"
    raw: Optional[dict] = None
    audits: List[Callable[[], None]] = field(default_factory=list)

    @property
    def cache_hit(self) -> bool:
        return self.outcome in ("exact_hit", "semantic_hit", "semantic_verified")

    @property
    def hit_type(self) -> Optional[str]:
        return {"exact_hit": "exact", "semantic_hit": "semantic",
                "semantic_verified": "semantic_verified"}.get(self.outcome)


@dataclass
class EngineConfig:
    policies: dict
    prices: dict
    default_model: str = "llama-3.1-8b-instant"
    top_k: int = 5
    rerank_enabled: bool = True
    judge_sample_rate: float = 0.0          # fraction of semantic hits audited by an LLM judge
    revalidate_sample_rate: float = 0.0     # fraction of hits re-run live to measure divergence
    replay_delay_ms: float = 0.0            # pacing between replayed chunks (0 = as fast as possible)
    stream_chunk_chars: int = 12


class Engine:
    def __init__(self, *, cfg: EngineConfig, store, index: CacheIndex, embedder, reranker,
                 router: ProviderRouter, telemetry: Telemetry, budget: BudgetManager,
                 verifier: Optional[Callable[[str, str, str], bool]] = None,
                 breaker: Optional[CircuitBreaker] = None, clock=time.time, rng=random.random):
        self.cfg, self.r, self.index = cfg, store, index
        self.embed, self.rerank = embedder, reranker
        self.router, self.tm, self.budget = router, telemetry, budget
        self.verifier = verifier
        self.breaker = breaker or CircuitBreaker()
        self._clock, self._rng = clock, rng

    # ------------------------------------------------------------------ helpers
    def _ms(self, t0: float) -> float:
        return (self._clock() - t0) * 1000.0

    def _event(self, etype: str, c: CanonicalRequest, rid: str, sim=None, ms=None):
        try:
            self.r.lpush("events:log", json.dumps({
                "type": etype, "request_id": rid, "tenant": c.tenant_id, "feature": c.feature,
                "tag": c.tag, "similarity": sim, "latency_ms": round(ms, 1) if ms else None,
                "timestamp": self._clock()}))
            self.r.ltrim("events:log", 0, 999)
        except Exception as e:   # observability must never break serving
            log.warning("event log write failed: %s", e)

    def _prep(self, c: CanonicalRequest):
        if not c.model:
            c.model = self.cfg.default_model
        policy = get_policy(self.cfg.policies, c.tag)
        ns = make_namespace(c.model, c.system, c.temperature, c.feature, c.tag, c.user_id,
                            tenant_id=c.tenant_id, context=c.context_fingerprint_text, tools=c.tools)
        ekey = exact_hash(ns, c.last_user_message)
        cache_ok = policy["cache_allowed"]
        note = None
        if cache_ok and c.tag == "user_specific" and not c.user_id:
            # would otherwise fall into a shared 'anonymous' scope -> never cache
            self.tm.isolation_block(c.tenant_id, "user_specific request without user_id")
            cache_ok, note = False, "user_specific without user_id: cache blocked"
        return policy, ns, ekey, cache_ok, note

    def _scope_ok(self, c: CanonicalRequest, entry: dict) -> bool:
        ok = entry.get("tenant") == c.tenant_id and \
             (c.tag != "user_specific" or entry.get("user") == c.user_id)
        if not ok:
            self.tm.isolation_block(c.tenant_id, f"scope mismatch on entry (tenant={entry.get('tenant')})")
        return ok

    # ------------------------------------------------------------------ lookup
    def _lookup(self, c: CanonicalRequest, policy: dict, ns: str, ekey: str):
        """Returns dict(kind, entry, sim, lookup_ms, embedded, vec, near)."""
        t0 = self._clock()
        out = {"kind": "miss", "entry": None, "sim": None, "vec": None, "embedded": False, "near": None}
        entry = self.index.get_entry(ns, ekey)
        if entry is not None:
            if self._scope_ok(c, entry):
                out.update(kind="exact", entry=entry, sim=1.0)
                out["lookup_ms"] = self._ms(t0)
                return out
        vec = self.embed(c.last_user_message)
        out["vec"], out["embedded"] = vec, True
        cands = self.index.top_k(ns, vec, self.cfg.top_k)
        best_score, best = None, None
        if cands:
            if self.cfg.rerank_enabled and self.rerank is not None:
                scores = self.rerank([(c.last_user_message, e["question"]) for _, e in cands])
                i = max(range(len(scores)), key=lambda j: scores[j])
                best_score, best = float(scores[i]), cands[i][1]
            else:
                best_score, best = cands[0]
        out["lookup_ms"] = self._ms(t0)
        if best is None:
            return out
        out["sim"] = best_score
        if not self._scope_ok(c, best):
            return out
        zone = classify_similarity(best_score, policy)
        if zone == "auto_hit":
            out.update(kind="semantic", entry=best)
        elif zone == "borderline":
            out["near"] = best           # verification happens outside the lookup timer
        return out

    # ------------------------------------------------------------------ serving a hit
    def _hit_result(self, c, rid, kind, entry, sim, lookup_ms, t0, embedded) -> Result:
        in_tok = estimate_tokens(c.input_text())
        out_tok = int(entry.get("out_tokens") or estimate_tokens(entry["answer"]))
        saved = call_cost(c.model, in_tok, out_tok, self.cfg.prices)
        age = (self._clock() - entry.get("created", self._clock())) / max(entry.get("ttl", 1), 1)
        res = Result(request_id=rid, text=entry["answer"], model=c.model,
                     outcome={"exact": "exact_hit", "semantic": "semantic_hit",
                              "semantic_verified": "semantic_verified"}[kind],
                     in_tokens=in_tok, out_tokens=out_tok, similarity=round(sim, 4) if sim is not None else None,
                     matched_question=entry["question"], saved_usd=saved, lookup_ms=lookup_ms)
        res._age_ratio, res._embedded = age, embedded      # type: ignore[attr-defined]
        try:
            self.tm.remember_hit(rid, {"feature": c.feature, "tenant": c.tenant_id,
                                       "question": c.last_user_message, "matched": entry["question"]})
        except Exception:
            pass
        if kind != "exact":
            if self.verifier and self._rng() < self.cfg.judge_sample_rate:
                res.audits.append(lambda: self._judge_audit(c, rid, entry))
        if self._rng() < self.cfg.revalidate_sample_rate:
            res.audits.append(lambda: self._revalidate(c, entry))
        return res

    def _judge_audit(self, c, rid, entry):
        try:
            ok = self.verifier(c.last_user_message, entry["answer"], entry["question"])
            self.tm.feedback(rid, bad=not ok, source="judge")
        except Exception as e:
            log.warning("judge audit failed: %s", e)

    def _revalidate(self, c, entry):
        try:
            fresh = self.router.resolve(c.model).complete(c)
            cost = call_cost(c.model, fresh.in_tokens or 0, fresh.out_tokens or 0, self.cfg.prices)
            self.tm.revalidation(c.feature, entry["answer"], fresh.text)
            self.tm._incr(["tm:global"], revalidation_spend_usd=float(cost))
        except Exception as e:
            log.warning("revalidation failed: %s", e)

    def _finish_hit(self, c, res: Result, t0: float, ttft_ms=None, streamed=False):
        res.latency_ms = self._ms(t0)
        self.tm.record_request(
            outcome=res.outcome, tenant=c.tenant_id, unit=c.extra.get("unit", "default"),
            key_id=c.extra.get("key_id", "anon"), feature=c.feature, model=c.model,
            total_ms=res.latency_ms, lookup_ms=res.lookup_ms, ttft_ms=ttft_ms, streamed=streamed,
            saved_in=res.in_tokens, saved_out=res.out_tokens, saved_usd=res.saved_usd,
            embedded=getattr(res, "_embedded", False), age_ratio=getattr(res, "_age_ratio", None))
        self._event(res.outcome, c, res.request_id, res.similarity, res.latency_ms)

    # ------------------------------------------------------------------ shared pre-live steps
    def _try_cache(self, c, policy, ns, ekey, rid, t0):
        """-> (hit Result | None, state dict). Never raises."""
        state = {"vec": None, "sim": None, "embedded": False, "failed": False, "lookup_ms": None}
        if not self.breaker.allow():
            state["failed"] = True
            self.tm.fail_open("circuit breaker open")
            return None, state
        try:
            lk = self._lookup(c, policy, ns, ekey)
            self.breaker.success()
        except Exception as e:
            self.breaker.failure()
            self.tm.fail_open(f"{type(e).__name__}: {e}")
            log.warning("cache lookup failed, failing open: %s", e)
            state["failed"] = True
            return None, state
        state.update(vec=lk["vec"], sim=lk["sim"], embedded=lk["embedded"], lookup_ms=lk.get("lookup_ms"))
        if lk["kind"] in ("exact", "semantic"):
            return self._hit_result(c, rid, lk["kind"], lk["entry"], lk["sim"], lk["lookup_ms"], t0,
                                    lk["embedded"]), state
        if lk["near"] is not None:
            verified = False
            if self.verifier:
                try:
                    verified = bool(self.verifier(c.last_user_message, lk["near"]["answer"], lk["near"]["question"]))
                except Exception:
                    verified = False
            self.tm.near_miss(lk["sim"], verified, c.feature)
            self._event("near_miss", c, rid, lk["sim"])
            if verified:
                return self._hit_result(c, rid, "semantic_verified", lk["near"], lk["sim"],
                                        lk["lookup_ms"], t0, lk["embedded"]), state
        return None, state

    def _budget_gate(self, c) -> str:
        t = c.extra.get("tenant_obj")
        if t is None:
            return "ok"
        try:
            d = self.budget.check(t)
        except Exception as e:           # budget store down -> fail open, never block the client
            log.warning("budget check failed, allowing request: %s", e)
            return "ok"
        if d.state != "ok":
            self.tm.budget_event(c.tenant_id, d.state)
        if not d.allowed:
            raise GatewayError(429, f"Budget/rate limit: {d.reason}", d.retry_after or 1)
        return d.state

    def _account_live(self, c, in_tok, out_tok) -> float:
        cost = call_cost(c.model, in_tok, out_tok, self.cfg.prices)
        t = c.extra.get("tenant_obj")
        if t is not None:
            try:
                self.budget.record(t, in_tok + out_tok, cost)
            except Exception as e:
                log.warning("budget record failed: %s", e)
        return cost

    def _store(self, c, policy, ns, ekey, text, in_tok, out_tok, vec):
        try:
            entry = {"question": c.last_user_message, "answer": text, "in_tokens": in_tok,
                     "out_tokens": out_tok, "model": c.model, "tenant": c.tenant_id,
                     "user": c.user_id, "feature": c.feature, "tag": c.tag,
                     "embedding": [float(x) for x in vec] if vec is not None else None}
            self.index.put(ns, ekey, entry, policy["ttl"])
            self.breaker.success()
        except Exception as e:
            self.breaker.failure()
            self.tm.fail_open(f"store write: {type(e).__name__}: {e}")
            log.warning("cache store failed (non-fatal): %s", e)

    # ------------------------------------------------------------------ non-streaming
    def handle(self, c: CanonicalRequest) -> Result:
        t0, rid = self._clock(), uuid.uuid4().hex
        if c.tools:
            return self._passthrough(c, rid, t0)
        policy, ns, ekey, cache_ok, note = self._prep(c)
        state = {"vec": None, "sim": None, "embedded": False, "failed": False, "lookup_ms": None}
        if cache_ok:
            hit, state = self._try_cache(c, policy, ns, ekey, rid, t0)
            if hit:
                self._finish_hit(c, hit, t0)
                return hit
        bstate = self._budget_gate(c)
        provider = self.router.resolve(c.model)
        try:
            pr = provider.complete(c)
        except ProviderError as e:
            raise GatewayError(e.status if e.status in (400, 401, 403, 404, 429) else 502, str(e), e.retry_after)
        in_tok = pr.in_tokens if pr.in_tokens is not None else estimate_tokens(c.input_text())
        out_tok = pr.out_tokens if pr.out_tokens is not None else estimate_tokens(pr.text)
        cost = self._account_live(c, in_tok, out_tok)
        if cache_ok and not state["failed"]:
            self._store(c, policy, ns, ekey, pr.text, in_tok, out_tok, state["vec"])
        outcome = "fail_open" if (cache_ok and state["failed"]) else ("miss" if cache_ok else "bypass")
        res = Result(request_id=rid, text=pr.text, model=c.model, outcome=outcome, in_tokens=in_tok,
                     out_tokens=out_tok, similarity=round(state["sim"], 4) if state["sim"] is not None else None,
                     cost_usd=cost, lookup_ms=state["lookup_ms"], budget_state=bstate,
                     note=note or ("cache unavailable - served live (fail-open)" if outcome == "fail_open" else None))
        res.latency_ms = self._ms(t0)
        self._record_live(c, res, state["embedded"])
        return res

    def _record_live(self, c, res: Result, embedded: bool, ttft_ms=None, streamed=False):
        self.tm.record_request(
            outcome=res.outcome, tenant=c.tenant_id, unit=c.extra.get("unit", "default"),
            key_id=c.extra.get("key_id", "anon"), feature=c.feature, model=c.model,
            total_ms=res.latency_ms, lookup_ms=res.lookup_ms, ttft_ms=ttft_ms, streamed=streamed,
            spent_in=res.in_tokens, spent_out=res.out_tokens, spent_usd=res.cost_usd, embedded=embedded)
        self._event(res.outcome, c, res.request_id, res.similarity, res.latency_ms)

    # ------------------------------------------------------------------ tools passthrough (never cached)
    def _passthrough(self, c, rid, t0) -> Result:
        bstate = self._budget_gate(c)
        provider = self.router.resolve(c.model or self.cfg.default_model)
        try:
            raw = provider.passthrough(c)
        except ProviderError as e:
            raise GatewayError(e.status if e.status in (400, 401, 403, 404, 429) else 502, str(e), e.retry_after)
        u = raw.get("usage") or {}
        in_tok = u.get("prompt_tokens", u.get("input_tokens")) or 0
        out_tok = u.get("completion_tokens", u.get("output_tokens")) or 0
        cost = self._account_live(c, in_tok, out_tok)
        res = Result(request_id=rid, text="", model=c.model, outcome="bypass", in_tokens=in_tok,
                     out_tokens=out_tok, cost_usd=cost, raw=raw, budget_state=bstate,
                     note="tool-calling request: forwarded, never cached")
        res.latency_ms = self._ms(t0)
        self._record_live(c, res, False)
        return res

    def handle_passthrough_stream(self, c: CanonicalRequest) -> Iterator[str]:
        t0, rid = self._clock(), uuid.uuid4().hex
        self._budget_gate(c)
        provider = self.router.resolve(c.model or self.cfg.default_model)
        usage = {"in": None, "out": None}
        first = None
        try:
            for line in provider.passthrough_stream(c):
                provider._capture_usage(line.strip(), usage)
                if first is None:
                    first = self._ms(t0)
                yield line
        except ProviderError as e:
            raise GatewayError(e.status if e.status in (400, 401, 403, 404, 429) else 502, str(e), e.retry_after)
        finally:
            in_tok, out_tok = usage["in"] or 0, usage["out"] or 0
            res = Result(request_id=rid, text="", model=c.model, outcome="bypass", in_tokens=in_tok,
                         out_tokens=out_tok, cost_usd=self._account_live(c, in_tok, out_tok))
            res.latency_ms = self._ms(t0)
            self._record_live(c, res, False, ttft_ms=first, streamed=True)

    # ------------------------------------------------------------------ streaming
    def handle_stream(self, c: CanonicalRequest):
        """Returns (info: Result-with-meta, iterator of text deltas).
        Cache hits are replayed as chunks; misses proxy the live stream and
        store the assembled answer once it completes. Partial streams (client
        disconnects, upstream errors) are never cached."""
        if c.tools:
            raise GatewayError(400, "stream+tools goes through handle_passthrough_stream")
        t0, rid = self._clock(), uuid.uuid4().hex
        policy, ns, ekey, cache_ok, note = self._prep(c)
        state = {"vec": None, "sim": None, "embedded": False, "failed": False, "lookup_ms": None}
        if cache_ok:
            hit, state = self._try_cache(c, policy, ns, ekey, rid, t0)
            if hit:
                return hit, self._replay_iter(c, hit, t0)
        bstate = self._budget_gate(c)
        provider = self.router.resolve(c.model)
        outcome = "fail_open" if (cache_ok and state["failed"]) else ("miss" if cache_ok else "bypass")
        info = Result(request_id=rid, text="", model=c.model, outcome=outcome, budget_state=bstate,
                      similarity=round(state["sim"], 4) if state["sim"] is not None else None,
                      lookup_ms=state["lookup_ms"], note=note)
        return info, self._live_iter(c, provider, info, policy, ns, ekey, cache_ok and not state["failed"],
                                     state, t0)

    def _replay_iter(self, c, hit: Result, t0) -> Iterator[str]:
        chunks = chunk_text(hit.text, self.cfg.stream_chunk_chars)
        ttft = None
        try:
            for i, ch in enumerate(chunks):
                if ttft is None:
                    ttft = self._ms(t0)
                elif self.cfg.replay_delay_ms:
                    time.sleep(self.cfg.replay_delay_ms / 1000.0)
                yield ch
        finally:
            self._finish_hit(c, hit, t0, ttft_ms=ttft, streamed=True)

    def _live_iter(self, c, provider, info: Result, policy, ns, ekey, store_ok, state, t0) -> Iterator[str]:
        parts: List[str] = []
        ttft = None
        completed = False
        stream = None
        try:
            stream = provider.stream(c)
            for delta in stream:
                if ttft is None:
                    ttft = self._ms(t0)
                parts.append(delta)
                yield delta
            completed = True
        except ProviderError as e:
            raise GatewayError(e.status if e.status in (400, 401, 403, 404, 429) else 502, str(e), e.retry_after)
        finally:
            text = "".join(parts)
            usage = stream.usage if (completed and stream is not None) else {"in": None, "out": None}
            in_tok = usage["in"] if usage["in"] is not None else estimate_tokens(c.input_text())
            out_tok = usage["out"] if usage["out"] is not None else estimate_tokens(text)
            info.text, info.in_tokens, info.out_tokens = text, in_tok, out_tok
            info.cost_usd = self._account_live(c, in_tok, out_tok)
            if completed and store_ok and text:
                self._store(c, policy, ns, ekey, text, in_tok, out_tok, state["vec"])
            info.latency_ms = self._ms(t0)
            self._record_live(c, info, state["embedded"], ttft_ms=ttft, streamed=True)


def make_llm_verifier(router: ProviderRouter, model: str) -> Callable[[str, str, str], bool]:
    """Borderline-match verifier: asks a small LLM whether the cached answer
    also answers the new question. Any failure -> False (never reuse)."""
    def verify(question: str, candidate_answer: str, matched_question: str) -> bool:
        c = CanonicalRequest(dialect="native", model=model, system="", temperature=0.0,
                             messages=[{"role": "user", "content": VERIFY_PROMPT.format(
                                 matched_question=matched_question, question=question,
                                 candidate_answer=candidate_answer)}], max_tokens=5)
        try:
            return router.resolve(model).complete(c).text.strip().upper().startswith("YES")
        except Exception as e:
            log.warning("verification failed, defaulting to no-reuse: %s", e)
            return False
    return verify
