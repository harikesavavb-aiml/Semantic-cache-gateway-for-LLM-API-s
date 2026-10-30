"""Dual-engine contract layer + SSE replay (spec section 5: SDK compatibility, streaming)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import *  # noqa
from gateway.adapters import (chunk_text, from_anthropic, from_openai, openai_stream, anthropic_stream,
                              parse_sse_text_delta, replay_stream, to_anthropic_response, to_openai_response)


def test_openai_request_maps_system_and_messages():
    c = from_openai({"model": "gpt-4o-mini", "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]}],
        "temperature": 0, "stream": True, "user": "u7"},
        {"X-Cache-Feature": "support", "X-Cache-Tag": "faq"})
    assert c.system == "Be brief." and c.last_user_message == "Hi there"
    assert c.stream and c.user_id == "u7" and c.feature == "support" and c.tag == "faq"


def test_anthropic_request_maps_top_level_system():
    c = from_anthropic({"model": "claude-sonnet-4-5", "max_tokens": 100, "system": [{"type": "text", "text": "Sys"}],
                        "messages": [{"role": "user", "content": "Hello"}]})
    assert c.system == "Sys" and c.last_user_message == "Hello" and c.dialect == "anthropic"


def test_anthropic_requires_max_tokens_and_a_user_turn():
    for body in ({"model": "c", "messages": [{"role": "user", "content": "x"}]},
                 {"model": "c", "max_tokens": 5, "messages": []}):
        try:
            from_anthropic(body)
            assert False, "should have raised"
        except ValueError:
            pass


def test_openai_requires_user_message():
    try:
        from_openai({"model": "m", "messages": [{"role": "system", "content": "x"}]})
        assert False
    except ValueError:
        pass


def test_response_shapes_match_native_sdk_schemas():
    o = to_openai_response("hello", "gpt-x", 3, 4)
    assert o["object"] == "chat.completion" and o["choices"][0]["message"]["content"] == "hello"
    assert o["usage"]["total_tokens"] == 7
    a = to_anthropic_response("hello", "claude-x", 3, 4)
    assert a["type"] == "message" and a["content"][0] == {"type": "text", "text": "hello"}
    assert a["usage"] == {"input_tokens": 3, "output_tokens": 4}


def test_chunk_text_is_lossless():
    for t in ["Hello world, this is a cached answer.\nSecond line.", "x", "  pad  ", "", "a b c d e f g h i j"]:
        assert "".join(chunk_text(t, 10)) == t


def _collect(stream, dialect):
    return "".join(d for d in (parse_sse_text_delta(l, dialect)
                               for blob in stream for l in blob.splitlines()) if d)


def test_openai_stream_roundtrip_and_terminator():
    text = "The capital of India is New Delhi."
    frames = list(openai_stream(chunk_text(text), "m", 5, 7))
    assert frames[-1] == "data: [DONE]\n\n"
    assert _collect(frames, "openai") == text
    finish = json.loads(frames[-2][5:])
    assert finish["choices"][0]["finish_reason"] == "stop"


def test_anthropic_stream_event_order_and_roundtrip():
    text = "The capital of India is New Delhi."
    frames = list(anthropic_stream(chunk_text(text), "claude-x", 5, 7))
    events = [f.split("\n")[0] for f in frames]
    assert events[0] == "event: message_start" and events[-1] == "event: message_stop"
    assert events.index("event: content_block_start") < events.index("event: content_block_delta") \
        < events.index("event: content_block_stop") < events.index("event: message_delta")
    assert _collect(frames, "anthropic") == text


def test_replay_stream_dispatches_by_dialect():
    assert next(iter(replay_stream("anthropic", ["a"], "m", 1, 1))).startswith("event: message_start")
    assert next(iter(replay_stream("openai", ["a"], "m", 1, 1))).startswith("data:")


def test_multiturn_context_excludes_last_user_turn():
    c = from_openai({"model": "m", "messages": [
        {"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"}]})
    assert "A" in c.context_fingerprint_text and "C" not in c.context_fingerprint_text
    single = from_openai({"model": "m", "messages": [{"role": "user", "content": "C"}]})
    assert single.context_fingerprint_text == ""
