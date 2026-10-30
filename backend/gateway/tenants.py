"""
tenants.py - API-key auth and per-tenant limits (multi-tenant attribution).

tenants.json maps SHA-256(api_key) -> tenant config, so raw keys never sit on
disk:
  {"<sha256 hex>": {"tenant": "acme", "unit": "support-team",
                    "rpm": 120, "daily_tokens": 500000,
                    "daily_usd": 25.0, "monthly_usd": 400.0}}
Generate a hash with:  python -m gateway.tenants hash <api-key>
"""
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Tenant:
    tenant: str
    unit: str = "default"
    rpm: Optional[int] = None
    daily_tokens: Optional[int] = None
    daily_usd: Optional[float] = None
    monthly_usd: Optional[float] = None
    key_id: str = ""          # first 8 hex of the key hash; safe to log


PUBLIC = Tenant(tenant="public", key_id="anon")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_tenants(path: Optional[str] = None) -> Dict[str, Tenant]:
    path = path or os.environ.get("TENANTS_CONFIG_PATH", "tenants.json")
    out: Dict[str, Tenant] = {}
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        for h, cfg in raw.items():
            cfg = dict(cfg)
            out[h.lower()] = Tenant(key_id=h[:8].lower(), **cfg)
    return out


def authenticate(tenants: Dict[str, Tenant], presented_key: Optional[str],
                 auth_required: bool) -> Optional[Tenant]:
    """Returns the Tenant, PUBLIC (dev mode: no tenants configured and auth
    not required), or None (reject with 401)."""
    if not tenants and not auth_required:
        return PUBLIC
    if not presented_key:
        return None
    h = hash_key(presented_key)
    for known, t in tenants.items():          # constant-time compare per entry
        if hmac.compare_digest(known, h):
            return t
    return None


def extract_key(headers: Dict[str, str]) -> Optional[str]:
    h = {k.lower(): v for k, v in headers.items()}
    if h.get("x-api-key"):
        return h["x-api-key"]
    auth = h.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "hash":
        print(hash_key(sys.argv[2]))
    else:
        print("usage: python -m gateway.tenants hash <api-key>")
