"""
tests/test_cache_logic.py

Unit tests for the pure decision logic in cache_logic.py. These run
instantly with no network access, no Redis, no embedding model, and no
LLM calls - satisfying NFR8 (Testability) by keeping the core business
logic decoupled from I/O and heavy dependencies.

Run with:  pytest tests/
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_logic import (
    make_namespace,
    exact_hash,
    classify_similarity,
    load_policies,
    get_policy,
    DEFAULT_POLICIES,
)


# ---------- Request Normalizer (FR2) ----------

def test_same_inputs_produce_same_namespace():
    ns1 = make_namespace("llama-3.1-8b-instant", "You are helpful.", 0.0, "default", "default", None)
    ns2 = make_namespace("llama-3.1-8b-instant", "You are helpful.", 0.0, "default", "default", None)
    assert ns1 == ns2


def test_different_system_prompt_produces_different_namespace():
    ns1 = make_namespace("llama-3.1-8b-instant", "You are helpful.", 0.0, "default", "default", None)
    ns2 = make_namespace("llama-3.1-8b-instant", "You are a pirate.", 0.0, "default", "default", None)
    assert ns1 != ns2


def test_different_temperature_produces_different_namespace():
    ns1 = make_namespace("m", "p", 0.0, "default", "default", None)
    ns2 = make_namespace("m", "p", 0.7, "default", "default", None)
    assert ns1 != ns2


def test_different_feature_produces_different_namespace():
    ns1 = make_namespace("m", "p", 0.0, "support", "default", None)
    ns2 = make_namespace("m", "p", 0.0, "billing", "default", None)
    assert ns1 != ns2


# ---------- Cross-user isolation (FR11 / NFR3) ----------

def test_user_specific_tag_isolates_different_users():
    ns_alice = make_namespace("m", "p", 0.0, "support", "user_specific", "alice")
    ns_bob = make_namespace("m", "p", 0.0, "support", "user_specific", "bob")
    assert ns_alice != ns_bob, "Two different users must never share a namespace"


def test_default_tag_ignores_user_id():
    # Non-user_specific tags should NOT fold in user_id, since sharing
    # is intended for general-purpose questions.
    ns1 = make_namespace("m", "p", 0.0, "support", "default", "alice")
    ns2 = make_namespace("m", "p", 0.0, "support", "default", "bob")
    assert ns1 == ns2


def test_user_specific_missing_user_id_does_not_crash():
    ns = make_namespace("m", "p", 0.0, "support", "user_specific", None)
    assert isinstance(ns, str) and len(ns) > 0


# ---------- Exact-match hashing (FR3) ----------

def test_exact_hash_deterministic():
    h1 = exact_hash("ns1", "What is the capital of India?")
    h2 = exact_hash("ns1", "What is the capital of India?")
    assert h1 == h2


def test_exact_hash_sensitive_to_namespace():
    h1 = exact_hash("ns1", "same text")
    h2 = exact_hash("ns2", "same text")
    assert h1 != h2, "Same text in different namespaces must not collide"


def test_exact_hash_sensitive_to_message():
    h1 = exact_hash("ns1", "What is the capital of India?")
    h2 = exact_hash("ns1", "What is the capital of Japan?")
    assert h1 != h2


# ---------- Three-zone similarity classifier (FR5/FR6) ----------

def test_score_above_auto_hit_is_auto_hit():
    policy = {"auto_hit": 0.93, "borderline_low": 0.70}
    assert classify_similarity(0.95, policy) == "auto_hit"


def test_score_exactly_at_auto_hit_boundary_is_auto_hit():
    policy = {"auto_hit": 0.93, "borderline_low": 0.70}
    assert classify_similarity(0.93, policy) == "auto_hit"


def test_score_in_borderline_zone_is_borderline():
    policy = {"auto_hit": 0.93, "borderline_low": 0.70}
    assert classify_similarity(0.80, policy) == "borderline"


def test_score_exactly_at_borderline_low_boundary_is_borderline():
    policy = {"auto_hit": 0.93, "borderline_low": 0.70}
    assert classify_similarity(0.70, policy) == "borderline"


def test_score_below_borderline_low_is_miss():
    policy = {"auto_hit": 0.93, "borderline_low": 0.70}
    assert classify_similarity(0.50, policy) == "miss"


def test_user_specific_policy_has_no_effective_borderline_zone():
    # user_specific sets auto_hit == borderline_low == 0.99, so nothing
    # below a near-perfect match should ever be reused or verified.
    policy = DEFAULT_POLICIES["user_specific"]
    assert classify_similarity(0.95, policy) == "miss"
    assert policy["cache_allowed"] is False


# ---------- Policy loading / configurability (NFR4) ----------

def test_load_policies_falls_back_to_defaults_when_file_missing():
    policies = load_policies(config_path="/tmp/does_not_exist_12345.json")
    assert policies == DEFAULT_POLICIES


def test_load_policies_falls_back_on_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ this is not valid json")
        path = f.name
    try:
        policies = load_policies(config_path=path)
        assert policies == DEFAULT_POLICIES
    finally:
        os.remove(path)


def test_load_policies_merges_custom_file_over_defaults():
    custom = {"default": {"auto_hit": 0.99, "borderline_low": 0.80, "ttl": 100, "cache_allowed": True}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        path = f.name
    try:
        policies = load_policies(config_path=path)
        assert policies["default"]["auto_hit"] == 0.99
        # Other tags not present in the custom file should still exist
        assert "user_specific" in policies
    finally:
        os.remove(path)


def test_get_policy_returns_default_for_unknown_tag():
    policies = DEFAULT_POLICIES
    result = get_policy(policies, "some_tag_that_does_not_exist")
    assert result == policies["default"]


def test_get_policy_returns_matching_tag():
    policies = DEFAULT_POLICIES
    result = get_policy(policies, "legal_sensitive")
    assert result == policies["legal_sensitive"]


# ---------- Sanity check on default policy shape ----------

def test_all_default_policies_have_required_keys():
    required = {"auto_hit", "borderline_low", "ttl", "cache_allowed"}
    for tag, policy in DEFAULT_POLICIES.items():
        assert required.issubset(policy.keys()), f"Policy '{tag}' missing required keys"


def test_all_default_policies_have_sane_threshold_ordering():
    for tag, policy in DEFAULT_POLICIES.items():
        assert policy["borderline_low"] <= policy["auto_hit"], \
            f"Policy '{tag}' has borderline_low > auto_hit, which makes the borderline zone inverted"
