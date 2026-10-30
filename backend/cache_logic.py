"""
cache_logic.py

Pure functions with zero heavy dependencies (no redis, no torch, no network
calls). This is deliberately separated from main.py so the core decision
logic - normalization, hashing, and the three-zone similarity classifier -
can be unit tested quickly and in isolation (NFR8: testability).
"""

import hashlib
import json
import os
from typing import Optional


DEFAULT_POLICIES = {
    "default":         {"auto_hit": 0.93, "borderline_low": 0.70, "ttl": 3600,  "cache_allowed": True},
    "faq":             {"auto_hit": 0.90, "borderline_low": 0.65, "ttl": 86400, "cache_allowed": True},
    "user_specific":   {"auto_hit": 0.99, "borderline_low": 0.99, "ttl": 60,    "cache_allowed": False},
    "current_events":  {"auto_hit": 0.97, "borderline_low": 0.85, "ttl": 300,   "cache_allowed": True},
    "legal_sensitive": {"auto_hit": 0.97, "borderline_low": 0.85, "ttl": 1800,  "cache_allowed": True},
}


def load_policies(config_path: Optional[str] = None) -> dict:
    """
    NFR4 (Configurability): policies are loaded from an external JSON file
    when available, so thresholds/TTLs can change without a code deploy.
    Falls back to DEFAULT_POLICIES if the file is missing or invalid, so a
    bad/missing config never crashes the gateway.
    """
    path = config_path or os.environ.get("POLICY_CONFIG_PATH", "config.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            merged = dict(DEFAULT_POLICIES)
            merged.update(loaded)
            return merged
        except (json.JSONDecodeError, OSError):
            return dict(DEFAULT_POLICIES)
    return dict(DEFAULT_POLICIES)


def get_policy(policies: dict, tag: str) -> dict:
    return policies.get(tag, policies["default"])


def make_namespace(model: str, system_prompt: str, temperature: float,
                    feature: str, tag: str, user_id: Optional[str],
                    *, tenant_id: str = "public", context: str = "", tools: Optional[list] = None) -> str:
    """
    Request Normalizer (FR2). Two requests can only ever share a cache
    entry if they agree on model, system prompt, temperature, feature,
    and tag. If tag == user_specific, user_id is folded in so two users
    can never share an answer (FR11 / NFR3: cross-user isolation).

    Production additions (keyword-only, so existing callers and namespaces
    are unchanged when they are left at their defaults):
      tenant_id - authenticated tenant; tenants never share entries
      context   - serialized earlier turns of a multi-turn conversation
      tools     - tool definitions; different tool sets never share entries
    """
    parts = [model, system_prompt, str(temperature), feature, tag]
    if tag == "user_specific":
        parts.append(user_id or "anonymous")
    if tenant_id and tenant_id != "public":
        parts.append("tenant=" + tenant_id)
    if context:
        parts.append("ctx=" + hashlib.sha256(context.encode()).hexdigest()[:16])
    if tools:
        parts.append("tools=" + hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:16])
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def exact_hash(namespace: str, message: str) -> str:
    return hashlib.sha256(f"{namespace}:{message}".encode()).hexdigest()[:16]


def classify_similarity(score: float, policy: dict) -> str:
    """
    Three-zone classifier (FR5/FR6), replacing a single fixed threshold.
    Calibration data (see calibrate_threshold.py) showed real paraphrase
    and non-paraphrase scores overlap for small embedding models, so a
    single cutoff cannot cleanly separate them. Instead:

      score >= auto_hit          -> "auto_hit"   (reuse immediately)
      borderline_low <= score    -> "borderline" (verify before reuse)
      score <  borderline_low    -> "miss"       (don't even verify)
    """
    if score >= policy["auto_hit"]:
        return "auto_hit"
    if score >= policy["borderline_low"]:
        return "borderline"
    return "miss"
