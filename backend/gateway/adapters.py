"""
adapters.py - dual-engine contract layer.

Mirrors the OpenAI /v1/chat/completions and Anthropic /v1/messages schemas
and maps both onto one internal request (CanonicalRequest), so a single
cache serves both SDKs. Also builds responses and Server-Sent-Event streams
for each dialect, which is what makes streaming-aware cache replay possible:
a cached *text* answer can be re-emitted as a native stream in either format.
"""
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class CanonicalRequest:
    dialect: str                       # "openai" | "anthropic" | "native"
    model: str
    system: str
    messages: List[Dict[str, Any]]     # [{"role","content"(str)}]
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[list] = None
    feature: str = "default"
    tag: str = "default"
    user_id: Optional[str] = None
    tenant_id: str = "public"          # filled from API key by auth layer
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m["role"] == "user":
                return m["content"]
        return ""

    @property
    def context_fingerprint_text(self) -> str:
        """Everything before the last user turn. Folded into the cache
        namespace so a multi-turn conversation never matches a different
        conversation that happens to end with the same last sentence."""
        idx = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=-1)
        return json.dumps(self.messages[:idx], sort_keys=True) if idx > 0 else ""

    def input_text(self) -> str:
        return self.system + "\n" + "\n".join(m["content"] for m in self.messages)


def _flatten_content(content) -> str:
    """OpenAI/Anthropic both allow content to be a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text"):
                out.append(part.get("text", ""))
        return "".join(out)
    return "" if content is None else str(content)


def _meta(body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    h = {k.lower(): v for k, v in (headers or {}).items()}
    md = body.get("metadata") or {}
    return {
        "feature": h.get("x-cache-feature") or md.get("feature") or "default",
        "tag": h.get("x-cache-tag") or md.get("tag") or "default",
        "user_id": h.get("x-user-id") or body.get("user") or md.get("user_id"),
    }


def from_openai(body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> CanonicalRequest:
    system_parts, msgs = [], []
    for m in body.get("messages", []):
        role, text = m.get("role"), _flatten_content(m.get("content"))
        if role in ("system", "developer"):
            system_parts.append(text)
        elif role in ("user", "assistant", "tool"):
            msgs.append({"role": role, "content": text})
    if not any(m["role"] == "user" for m in msgs):
        raise ValueError("messages must contain at least one user message")
    return CanonicalRequest(
        dialect="openai", model=body.get("model", ""), system="\n".join(system_parts),
        messages=msgs, temperature=float(body.get("temperature", 1.0) or 0.0),
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        stream=bool(body.get("stream", False)), tools=body.get("tools"), **_meta(body, headers or {}),
    )


def from_anthropic(body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> CanonicalRequest:
    system = _flatten_content(body.get("system"))
    msgs = [{"role": m["role"], "content": _flatten_content(m.get("content"))}
            for m in body.get("messages", []) if m.get("role") in ("user", "assistant")]
    if not any(m["role"] == "user" for m in msgs):
        raise ValueError("messages must contain at least one user message")
    if not body.get("max_tokens"):
        raise ValueError("max_tokens is required by the Anthropic Messages API")
    return CanonicalRequest(
        dialect="anthropic", model=body.get("model", ""), system=system, messages=msgs,
        temperature=float(body.get("temperature", 1.0) or 0.0), max_tokens=body.get("max_tokens"),
        stream=bool(body.get("stream", False)), tools=body.get("tools"), **_meta(body, headers or {}),
    )


# ------------------------- non-streaming responses -------------------------

def to_openai_response(text: str, model: str, in_tok: int, out_tok: int) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                  "total_tokens": in_tok + out_tok},
    }


def to_anthropic_response(text: str, model: str, in_tok: int, out_tok: int) -> Dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "model": model, "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


# ------------------------- SSE: chunking + replay --------------------------

def chunk_text(text: str, size: int = 12) -> List[str]:
    """Split into word-boundary chunks of roughly `size` chars so a replayed
    answer looks like a live stream. "".join(chunks) == text, always."""
    words = re.findall(r"\S+\s*|\s+", text)
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) > size:
            chunks.append(cur)
            cur = w
        else:
            cur += w
    if cur:
        chunks.append(cur)
    return chunks


def _sse(data: Dict[str, Any], event: Optional[str] = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, separators=(',', ':'))}\n\n"


def openai_stream(chunks: List[str], model: str, in_tok: int = 0, out_tok: int = 0) -> Iterator[str]:
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                     "finish_reason": None}]})
    for c in chunks:
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                          "total_tokens": in_tok + out_tok}})
    yield "data: [DONE]\n\n"


def anthropic_stream(chunks: List[str], model: str, in_tok: int = 0, out_tok: int = 0) -> Iterator[str]:
    mid = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse({"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": 0}}}, "message_start")
    yield _sse({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}, "content_block_start")
    yield _sse({"type": "ping"}, "ping")
    for c in chunks:
        yield _sse({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": c}}, "content_block_delta")
    yield _sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
    yield _sse({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": out_tok}}, "message_delta")
    yield _sse({"type": "message_stop"}, "message_stop")


def replay_stream(dialect: str, chunks: List[str], model: str, in_tok: int, out_tok: int) -> Iterator[str]:
    fn = anthropic_stream if dialect == "anthropic" else openai_stream
    return fn(chunks, model, in_tok, out_tok)


# ------------------------- parsing live upstream SSE -----------------------

def parse_sse_text_delta(line: str, dialect: str) -> Optional[str]:
    """Extract the text delta from one upstream SSE 'data:' line, else None.
    Used while proxying a live stream so the full answer can be assembled
    and stored for later replay."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    if dialect == "anthropic":
        if obj.get("type") == "content_block_delta":
            return obj.get("delta", {}).get("text")
        return None
    try:
        return obj["choices"][0]["delta"].get("content")
    except (KeyError, IndexError, AttributeError, TypeError):
        return None
