"""
memstore.py - in-memory stand-in exposing the subset of the redis-py API the
gateway uses (strings, hashes, lists, sorted sets, counters, TTL, scan).
Used by the unit tests and by single-process dev mode; production uses Redis.
Both are decode_responses=True style (str in / str out).
"""
import fnmatch
import time
from typing import Dict, Iterator, List, Optional


class MemoryStore:
    def __init__(self, clock=time.time):
        self._d: Dict[str, object] = {}
        self._exp: Dict[str, float] = {}
        self._clock = clock

    def _alive(self, k: str) -> bool:
        e = self._exp.get(k)
        if e is not None and e <= self._clock():
            self._d.pop(k, None)
            self._exp.pop(k, None)
            return False
        return k in self._d

    # strings
    def get(self, k):
        return self._d[k] if self._alive(k) else None

    def set(self, k, v, ex: Optional[int] = None):
        self._d[k] = v
        if ex:
            self._exp[k] = self._clock() + ex
        else:
            self._exp.pop(k, None)
        return True

    def mget(self, keys: List[str]):
        return [self.get(k) for k in keys]

    def delete(self, *keys):
        n = 0
        for k in keys:
            if self._alive(k):
                n += 1
            self._d.pop(k, None)
            self._exp.pop(k, None)
        return n

    def expire(self, k, seconds):
        if self._alive(k):
            self._exp[k] = self._clock() + seconds
            return True
        return False

    def incr(self, k, amount=1):
        v = int(self.get(k) or 0) + amount
        ex = self._exp.get(k)
        self._d[k] = str(v)
        if ex is not None:
            self._exp[k] = ex
        return v

    def ping(self):
        return True

    # hashes
    def _h(self, k) -> dict:
        if not self._alive(k):
            self._d[k] = {}
        return self._d[k]  # type: ignore

    def hincrby(self, k, f, amount=1):
        h = self._h(k)
        h[f] = str(int(float(h.get(f, 0))) + amount)
        return int(h[f])

    def hincrbyfloat(self, k, f, amount):
        h = self._h(k)
        h[f] = repr(float(h.get(f, 0)) + float(amount))
        return float(h[f])

    def hset(self, k, f, v):
        self._h(k)[f] = str(v)

    def hget(self, k, f):
        return self._h(k).get(f) if self._alive(k) else None

    def hgetall(self, k):
        return dict(self._d[k]) if self._alive(k) else {}  # type: ignore

    # lists (lpush = newest first, like redis)
    def lpush(self, k, v):
        if not self._alive(k):
            self._d[k] = []
        self._d[k].insert(0, v)  # type: ignore
        return len(self._d[k])  # type: ignore

    def ltrim(self, k, start, stop):
        if self._alive(k):
            self._d[k] = self._d[k][start: stop + 1]  # type: ignore

    def lrange(self, k, start, stop):
        if not self._alive(k):
            return []
        lst = self._d[k]
        return list(lst[start:] if stop == -1 else lst[start: stop + 1])  # type: ignore

    # sorted sets
    def zadd(self, k, mapping: Dict[str, float]):
        if not self._alive(k):
            self._d[k] = {}
        self._d[k].update({m: float(s) for m, s in mapping.items()})  # type: ignore

    def zrem(self, k, *members):
        if self._alive(k):
            for m in members:
                self._d[k].pop(m, None)  # type: ignore

    def zcard(self, k):
        return len(self._d[k]) if self._alive(k) else 0  # type: ignore

    def zrange(self, k, start, stop):
        if not self._alive(k):
            return []
        items = sorted(self._d[k].items(), key=lambda kv: kv[1])  # type: ignore
        ids = [m for m, _ in items]
        return ids[start:] if stop == -1 else ids[start: stop + 1]

    def zremrangebyscore(self, k, lo, hi):
        if not self._alive(k):
            return 0
        d = self._d[k]
        dead = [m for m, s in d.items() if lo <= s <= hi]  # type: ignore
        for m in dead:
            d.pop(m)  # type: ignore
        return len(dead)

    def scan_iter(self, match="*", count=500) -> Iterator[str]:
        for k in list(self._d):
            if self._alive(k) and fnmatch.fnmatch(k, match):
                yield k
