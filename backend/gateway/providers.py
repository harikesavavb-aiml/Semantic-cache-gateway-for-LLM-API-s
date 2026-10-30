"""
providers.py - upstream LLM providers (OpenAI, Anthropic, Groq / any
OpenAI-compatible endpoint) behind one interface, plus model->provider routing.

    complete(canon)        -> ProviderResult(text, in_tokens, out_tokens)
    stream(canon)          -> ProviderStream: iterate text deltas; .usage after
    passthrough(canon)     -> raw JSON dict   (tool-calling requests; never cached)
    passthrough_stream(c)  -> iterator of raw SSE lines

httpx is imported lazily so the pure-logic modules/tests load without it.
Transient failures (429 / 5xx / timeouts) are retried with exponential
backoff, honouring Retry-After.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .adapters import CanonicalRequest, parse_sse_text_delta


class ProviderError(Exception):
    def __init__(self, status: int, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


class DialectMismatch(ProviderError):
    def __init__(self):
        super().__init__(400, "Requests with tools must target a model whose provider speaks the same "
                              "API dialect as the endpoint used (e.g. /v1/messages -> claude-*).")


@dataclass
class ProviderResult:
    text: str
    in_tokens: Optional[int] = None
    out_tokens: Optional[int] = None


class ProviderStream:
    """Iterable of text deltas. `usage` is populated once exhausted."""
    def __init__(self, gen: Iterator[str]):
        self._gen = gen
        self.usage: Dict[str, Optional[int]] = {"in": None, "out": None}

    def __iter__(self):
        return self._gen


def _retry_sleep(attempt: int, retry_after: Optional[int]):
    time.sleep(min(retry_after if retry_after else 0.5 * (2 ** attempt), 8))


class BaseProvider:
    name = "base"
    dialect = "openai"
    max_retries = 3

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0):
        self.api_key, self.base_url, self.timeout = api_key, base_url.rstrip("/"), timeout

    # subclass hooks
    def _url(self) -> str: ...
    def _headers(self) -> Dict[str, str]: ...
    def _payload(self, c: CanonicalRequest, stream: bool) -> dict: ...
    def _parse(self, data: dict) -> ProviderResult: ...

    def _post(self, payload: dict) -> dict:
        import httpx
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(self._url(), headers=self._headers(), json=payload, timeout=self.timeout)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = ProviderError(504, f"{self.name} unreachable: {e}")
            else:
                if r.status_code < 400:
                    return r.json()
                ra = r.headers.get("retry-after")
                last = ProviderError(r.status_code, f"{self.name}: {r.text[:300]}",
                                     int(ra) if ra and ra.isdigit() else None)
                if r.status_code not in (429, 500, 502, 503, 504, 529):
                    raise last
            if attempt < self.max_retries:
                _retry_sleep(attempt, last.retry_after)
        raise last  # type: ignore

    def complete(self, c: CanonicalRequest) -> ProviderResult:
        return self._parse(self._post(self._payload(c, False)))

    def passthrough(self, c: CanonicalRequest) -> dict:
        if c.dialect != self.dialect or "raw_body" not in c.extra:
            raise DialectMismatch()
        body = dict(c.extra["raw_body"])
        body["model"] = c.model
        return self._post(body)

    def _open_stream(self, payload: dict):
        import httpx
        return httpx.stream("POST", self._url(), headers=self._headers(), json=payload, timeout=self.timeout)

    def passthrough_stream(self, c: CanonicalRequest) -> Iterator[str]:
        if c.dialect != self.dialect or "raw_body" not in c.extra:
            raise DialectMismatch()
        body = dict(c.extra["raw_body"])
        body["model"] = c.model
        with self._open_stream(body) as r:
            if r.status_code >= 400:
                r.read()
                raise ProviderError(r.status_code, f"{self.name}: {r.text[:300]}")
            for line in r.iter_lines():
                yield line + "\n"

    def stream(self, c: CanonicalRequest) -> ProviderStream:
        payload = self._payload(c, True)
        holder: ProviderStream

        def gen():
            with self._open_stream(payload) as r:
                if r.status_code >= 400:
                    r.read()
                    ra = r.headers.get("retry-after")
                    raise ProviderError(r.status_code, f"{self.name}: {r.text[:300]}",
                                        int(ra) if ra and ra.isdigit() else None)
                for line in r.iter_lines():
                    self._capture_usage(line, holder.usage)
                    delta = parse_sse_text_delta(line, self.dialect)
                    if delta:
                        yield delta
        holder = ProviderStream(gen())
        return holder

    def _capture_usage(self, line: str, usage: dict):
        pass


class OpenAICompatProvider(BaseProvider):
    """OpenAI, Groq, and any other OpenAI-compatible endpoint."""
    dialect = "openai"

    def __init__(self, name: str, api_key: str, base_url: str, **kw):
        super().__init__(api_key, base_url, **kw)
        self.name = name

    def _url(self):
        return f"{self.base_url}/chat/completions"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, c, stream):
        msgs = ([{"role": "system", "content": c.system}] if c.system else []) + \
               [{"role": m["role"], "content": m["content"]} for m in c.messages]
        p = {"model": c.model, "messages": msgs, "temperature": c.temperature, "stream": stream}
        if c.max_tokens:
            p["max_tokens"] = c.max_tokens
        if stream:
            p["stream_options"] = {"include_usage": True}
        return p

    def _parse(self, data):
        u = data.get("usage") or {}
        return ProviderResult(data["choices"][0]["message"].get("content") or "",
                              u.get("prompt_tokens"), u.get("completion_tokens"))

    def _capture_usage(self, line, usage):
        if line.startswith("data:") and "usage" in line:
            try:
                u = json.loads(line[5:]).get("usage") or {}
                if u:
                    usage["in"], usage["out"] = u.get("prompt_tokens"), u.get("completion_tokens")
            except ValueError:
                pass


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    dialect = "anthropic"

    def _url(self):
        return f"{self.base_url}/v1/messages"

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}

    def _payload(self, c, stream):
        msgs = [{"role": m["role"], "content": m["content"]} for m in c.messages
                if m["role"] in ("user", "assistant")]
        p = {"model": c.model, "max_tokens": c.max_tokens or 1024, "messages": msgs,
             "temperature": min(max(c.temperature, 0.0), 1.0), "stream": stream}
        if c.system:
            p["system"] = c.system
        return p

    def _parse(self, data):
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        return ProviderResult(text, u.get("input_tokens"), u.get("output_tokens"))

    def _capture_usage(self, line, usage):
        if not line.startswith("data:"):
            return
        try:
            o = json.loads(line[5:])
        except ValueError:
            return
        if o.get("type") == "message_start":
            usage["in"] = (o.get("message", {}).get("usage") or {}).get("input_tokens")
        elif o.get("type") == "message_delta":
            usage["out"] = (o.get("usage") or {}).get("output_tokens")


class ProviderRouter:
    """model name -> provider. Prefix rules are checked in order; the last
    registered provider is the default (Groq, matching the original gateway)."""

    def __init__(self):
        self.rules: List[tuple] = []     # (prefix, provider)
        self.default: Optional[BaseProvider] = None

    def add(self, prefixes, provider: BaseProvider, default: bool = False):
        for p in prefixes:
            self.rules.append((p, provider))
        if default:
            self.default = provider

    def resolve(self, model: str) -> BaseProvider:
        m = (model or "").lower()
        for prefix, prov in self.rules:
            if m.startswith(prefix):
                return prov
        if self.default is None:
            raise ProviderError(400, f"No provider configured for model '{model}'")
        return self.default


def build_router_from_env() -> ProviderRouter:
    env = os.environ.get
    router = ProviderRouter()
    if env("ANTHROPIC_API_KEY"):
        router.add(["claude"], AnthropicProvider(env("ANTHROPIC_API_KEY"),
                                                 env("ANTHROPIC_BASE_URL", "https://api.anthropic.com")))
    if env("OPENAI_API_KEY"):
        router.add(["gpt", "o1", "o3", "o4", "chatgpt"],
                   OpenAICompatProvider("openai", env("OPENAI_API_KEY"),
                                        env("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    if env("GROQ_API_KEY"):
        router.add(["llama", "gemma", "mixtral", "openai/", "qwen", "deepseek"],
                   OpenAICompatProvider("groq", env("GROQ_API_KEY"),
                                        env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")),
                   default=True)
    return router
