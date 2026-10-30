"""
loadtest.py - mixed workload against the gateway's OpenAI-compatible endpoint:
exact repeats, paraphrases, and brand-new questions, in streaming and
non-streaming modes. Measure real savings, latency and false-hit rate; do not
quote numbers you did not measure.

    locust -f loadtest.py --host=http://127.0.0.1:8000
    # with auth enabled:  GATEWAY_API_KEY=sk-... locust -f loadtest.py --host=...
    # then open http://localhost:8089

Compare against a baseline run with the cache bypassed (e.g. send
`X-Cache-Tag: user_specific` + a user id, or point locust straight at the
provider) to compute P95/P99 reduction. After a run, read the gateway's own
numbers from GET /admin/report (X-Admin-Key header).
"""
import os
import random
import uuid

from locust import HttpUser, between, task

CLUSTERS = [
    ["What is the capital of India?", "Which city is the capital of India?", "Tell me India's capital city"],
    ["How do I reset my password?", "I forgot my password, what do I do?", "Password reset steps please"],
    ["What are your business hours?", "When are you open?", "What time do you open and close?"],
    ["Can I get a refund?", "Is a refund possible for this?", "How do refunds work here?"],
    ["Is there a free trial?", "Can I try this for free?", "Do you offer a trial period?"],
]
MODEL = os.environ.get("LOAD_MODEL", "llama-3.1-8b-instant")
HEADERS = {"Authorization": f"Bearer {os.environ['GATEWAY_API_KEY']}"} if os.environ.get("GATEWAY_API_KEY") else {}


class GatewayUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def _question(self):
        if random.random() < 0.15:                       # ~15% genuinely new questions
            return f"Explain topic number {uuid.uuid4().hex[:8]} in one sentence."
        return random.choice(random.choice(CLUSTERS))

    @task(4)
    def non_streaming(self):
        self.client.post("/v1/chat/completions", name="chat (non-stream)", headers=HEADERS, json={
            "model": MODEL, "temperature": 0,
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": self._question()}]})

    @task(2)
    def streaming(self):
        with self.client.post("/v1/chat/completions", name="chat (stream)", headers=HEADERS, stream=True,
                              catch_response=True, json={
                                  "model": MODEL, "temperature": 0, "stream": True,
                                  "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                               {"role": "user", "content": self._question()}]}) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
                return
            for _ in r.iter_lines():                      # consume the stream
                pass
            r.success()

    @task(1)
    def anthropic_dialect(self):
        self.client.post("/v1/messages", name="messages (anthropic)", headers=HEADERS, json={
            "model": MODEL, "max_tokens": 200, "temperature": 0,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": self._question()}]})
