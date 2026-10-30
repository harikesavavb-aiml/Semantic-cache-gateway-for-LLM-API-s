"""
index.py - cache entry store + per-namespace vector index.

Replaces the original KEYS-scan-and-json.loads-everything search (O(N) and
blocking on Redis) with:
  * entry:{ns}:{id}  JSON entry with TTL. The id is the exact-match hash, so
                     an exact hit is one GET on the same record.
  * idx:{ns}         ZSET id -> expiry time; expired ids are pruned on read
  * idxv:{ns}        version counter bumped on every write / invalidation
  * an in-process numpy matrix per namespace, rebuilt only when the version
    changes, so a semantic lookup is one matrix-vector product (sub-ms for
    thousands of entries) rather than N Redis round trips.
Embeddings are L2-normalised, so cosine similarity == dot product.
Each namespace is capped (MAX_ENTRIES_PER_NS); the soonest-to-expire entry is
evicted first.
"""
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


class CacheIndex:
    def __init__(self, store, max_per_ns: int = 5000, clock=time.time, max_cached_ns: int = 256):
        self.r, self.max_per_ns, self._clock = store, max_per_ns, clock
        self._mem: Dict[str, Tuple[str, List[str], np.ndarray, List[dict]]] = {}
        self._max_cached_ns = max_cached_ns

    # ---- exact ----
    def get_entry(self, ns: str, eid: str) -> Optional[dict]:
        raw = self.r.get(f"entry:{ns}:{eid}")
        return json.loads(raw) if raw else None

    # ---- write ----
    def put(self, ns: str, eid: str, entry: dict, ttl: int):
        now = self._clock()
        entry = {**entry, "created": now, "ttl": ttl}
        self.r.set(f"entry:{ns}:{eid}", json.dumps(entry), ex=ttl)
        self.r.zadd(f"idx:{ns}", {eid: now + ttl})
        overflow = self.r.zcard(f"idx:{ns}") - self.max_per_ns
        if overflow > 0:
            victims = self.r.zrange(f"idx:{ns}", 0, overflow - 1)
            self.r.zrem(f"idx:{ns}", *victims)
            self.r.delete(*[f"entry:{ns}:{v}" for v in victims])
        self.r.incr(f"idxv:{ns}")

    # ---- semantic ----
    def _load(self, ns: str):
        ver = self.r.get(f"idxv:{ns}") or "0"
        cached = self._mem.get(ns)
        if cached and cached[0] == ver:
            return cached
        self.r.zremrangebyscore(f"idx:{ns}", 0, self._clock())
        ids = self.r.zrange(f"idx:{ns}", 0, -1)
        entries, keep, vecs = [], [], []
        if ids:
            for eid, raw in zip(ids, self.r.mget([f"entry:{ns}:{i}" for i in ids])):
                if raw is None:
                    continue
                e = json.loads(raw)
                if not e.get("embedding"):
                    continue
                keep.append(eid)
                entries.append(e)
                vecs.append(e["embedding"])
        mat = np.asarray(vecs, dtype=np.float32) if vecs else np.zeros((0, 1), dtype=np.float32)
        if len(self._mem) >= self._max_cached_ns:
            self._mem.pop(next(iter(self._mem)))
        self._mem[ns] = (ver, keep, mat, entries)
        return self._mem[ns]

    def top_k(self, ns: str, query_vec, k: int = 5) -> List[Tuple[float, dict]]:
        _, _, mat, entries = self._load(ns)
        if mat.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q)
        q = q / n if n else q
        scores = mat @ q
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [(float(scores[i]), entries[i]) for i in idx]

    # ---- admin ----
    def invalidate(self, *, feature=None, tag=None, model=None, tenant=None,
                   created_before=None, created_after=None) -> int:
        """Rule-based removal by model / feature / tag / tenant / time range."""
        removed = 0
        touched = set()
        for key in list(self.r.scan_iter(match="entry:*")):
            raw = self.r.get(key)
            if raw is None:
                continue
            e = json.loads(raw)
            if feature and e.get("feature") != feature:
                continue
            if tag and e.get("tag") != tag:
                continue
            if model and e.get("model") != model:
                continue
            if tenant and e.get("tenant") != tenant:
                continue
            if created_before is not None and e.get("created", 0) >= created_before:
                continue
            if created_after is not None and e.get("created", 0) <= created_after:
                continue
            _, ns, eid = key.split(":", 2)
            self.r.delete(key)
            self.r.zrem(f"idx:{ns}", eid)
            touched.add(ns)
            removed += 1
        for ns in touched:
            self.r.incr(f"idxv:{ns}")
        return removed

    def count(self) -> int:
        return sum(1 for _ in self.r.scan_iter(match="entry:*"))
