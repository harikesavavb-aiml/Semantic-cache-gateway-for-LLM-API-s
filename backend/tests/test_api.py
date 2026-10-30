"""
HTTP-level tests using FastAPI's TestClient with a fake engine (no Redis, no
models, no network). Skipped automatically if fastapi/httpx aren't installed.

Run:  pytest tests/test_api.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pytest
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
except ImportError:           # plain-python runners: nothing to do
    TestClient = None

from helpers import make_engine


if TestClient is not None:

    @pytest.fixture(autouse=True)
    def _engine(monkeypatch):
        global eng
        eng = make_engine()
        import main
        main._state["redis"] = eng.store
        monkeypatch.setattr(main, "TENANTS", {})
        monkeypatch.setattr(main, "AUTH_REQUIRED", False)
        monkeypatch.setattr(main, "ADMIN_API_KEY", "adm")
        monkeypatch.setattr(main, "get_engine", lambda: eng)
        monkeypatch.setattr(main.app.router, "on_startup", [])      # skip model preload

    def test_openai_endpoint_miss_then_hit_with_headers():
        import main
        c = TestClient(main.app)
        body = {"model": "m", "messages": [{"role": "user", "content": "hello cache"}]}
        r1 = c.post("/v1/chat/completions", json=body)
        r2 = c.post("/v1/chat/completions", json=body)
        assert r1.headers["x-cache"] == "MISS" and r2.headers["x-cache"] == "HIT"
        assert r2.json()["choices"][0]["message"]["content"] == "ANSWER[hello cache]"
        assert r2.json()["object"] == "chat.completion"

    def test_anthropic_endpoint_shape():
        import main
        c = TestClient(main.app)
        r = c.post("/v1/messages", json={"model": "m", "max_tokens": 20,
                                         "messages": [{"role": "user", "content": "hello cache"}]})
        j = r.json()
        assert r.status_code == 200 and j["type"] == "message" and j["content"][0]["type"] == "text"

    def test_openai_streaming_sse():
        import main
        c = TestClient(main.app)
        body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "stream please"}]}
        for expected in ("MISS", "HIT"):
            with c.stream("POST", "/v1/chat/completions", json=body) as r:
                assert r.headers["x-cache"] == expected
                lines = [l for l in r.iter_lines() if l.startswith("data:")]
            assert lines[-1] == "data: [DONE]"
            text = "".join(json.loads(l[5:])["choices"][0]["delta"].get("content", "")
                           for l in lines[:-1] if json.loads(l[5:])["choices"][0]["delta"])
            assert text == "ANSWER[stream please]"

    def test_auth_required_rejects_missing_key(monkeypatch):
        import main
        from gateway.tenants import Tenant, hash_key
        monkeypatch.setattr(main, "TENANTS", {hash_key("sk-ok"): Tenant(tenant="acme", key_id="k")})
        monkeypatch.setattr(main, "AUTH_REQUIRED", True)
        c = TestClient(main.app)
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        assert c.post("/v1/chat/completions", json=body).status_code == 401
        assert c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-ok"}).status_code == 200
        assert c.post("/v1/messages", json={"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
                      headers={"x-api-key": "sk-ok"}).status_code == 200

    def test_budget_429_has_retry_after(monkeypatch):
        import main
        from gateway.tenants import Tenant, hash_key
        t = Tenant(tenant="acme", key_id="k", rpm=1)
        monkeypatch.setattr(main, "TENANTS", {hash_key("sk"): t})
        monkeypatch.setattr(main, "AUTH_REQUIRED", True)
        c = TestClient(main.app)
        h = {"Authorization": "Bearer sk"}
        mk = lambda q: {"model": "m", "messages": [{"role": "user", "content": q}]}   # noqa: E731
        assert c.post("/v1/chat/completions", json=mk("one"), headers=h).status_code == 200
        r = c.post("/v1/chat/completions", json=mk("two"), headers=h)
        assert r.status_code == 429 and int(r.headers["retry-after"]) > 0 and r.json()["error"]["type"]

    def test_admin_endpoints_require_key():
        import main
        c = TestClient(main.app)
        assert c.get("/admin/report").status_code == 401
        assert c.delete("/cache").status_code == 401
        assert c.get("/admin/report", headers={"X-Admin-Key": "adm"}).status_code == 200

    def test_feedback_updates_false_positive_rate():
        import main
        c = TestClient(main.app)
        body = {"model": "m", "messages": [{"role": "user", "content": "feedback test"}]}
        c.post("/v1/chat/completions", json=body)
        hit = c.post("/v1/chat/completions", json=body)
        rid = hit.headers["x-request-id"]
        assert c.post("/v1/feedback", json={"request_id": rid, "bad": True}).status_code == 200
        rep = c.get("/admin/report", headers={"X-Admin-Key": "adm"}).json()
        assert rep["quality"]["false_positive_rate"] == 1.0

    def test_invalidate_by_feature():
        import main
        c = TestClient(main.app)
        body = {"model": "m", "messages": [{"role": "user", "content": "to be removed"}]}
        c.post("/v1/chat/completions", json=body, headers={"X-Cache-Feature": "billing"})
        r = c.delete("/cache?feature=billing", headers={"X-Admin-Key": "adm"})
        assert r.json()["cleared"] == 1

    def test_legacy_chat_endpoint_still_works():
        import main
        c = TestClient(main.app)
        r = c.post("/chat", json={"message": "legacy call"})
        assert r.status_code == 200 and r.json()["answer"] == "ANSWER[legacy call]" and r.json()["cache_hit"] is False
