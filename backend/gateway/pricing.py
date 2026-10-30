"""
pricing.py - provider price table, token estimation and savings math.

Savings = sum(saved_input_tokens * price_in + saved_output_tokens * price_out)
Net_ROI = total_llm_dollars_saved - (embedding_cost + vector_db_hosting_cost)

Prices are USD per 1M tokens and are overridable at runtime through the
PRICING_CONFIG_PATH JSON file so finance can track live provider pricing
without a deploy. The defaults below are illustrative placeholders: verify
them against your provider's current price sheet before relying on the
dollar figures.
"""
import json
import math
import os
from typing import Dict, Optional, Tuple

# model-prefix -> (input $/1M tokens, output $/1M tokens)
DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
    "llama-3.1-8b": (0.05, 0.08),
    "llama-3.3-70b": (0.59, 0.79),
    "gemma2-9b": (0.20, 0.20),
    "default": (1.00, 3.00),
}


def load_prices(path: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    path = path or os.environ.get("PRICING_CONFIG_PATH", "pricing.json")
    prices = dict(DEFAULT_PRICES)
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                for k, v in json.load(f).items():
                    prices[k] = (float(v["input"]), float(v["output"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass  # bad file never crashes the gateway
    return prices


def price_for(model: str, prices: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Longest-prefix match so 'claude-sonnet-4-5-2025' resolves to 'claude-sonnet'."""
    m = (model or "").lower()
    best = None
    for prefix in prices:
        if prefix != "default" and m.startswith(prefix):
            if best is None or len(prefix) > len(best):
                best = prefix
    return prices[best or "default"]


def estimate_tokens(text: str) -> int:
    """Provider-agnostic estimate (~4 chars/token). Used only when the upstream
    response carries no usage block; real usage always wins when present."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def call_cost(model: str, in_tokens: int, out_tokens: int, prices) -> float:
    p_in, p_out = price_for(model, prices)
    return (in_tokens * p_in + out_tokens * p_out) / 1_000_000


def net_roi(total_saved: float, embedding_cost: float, vector_db_cost: float) -> float:
    return total_saved - (embedding_cost + vector_db_cost)
