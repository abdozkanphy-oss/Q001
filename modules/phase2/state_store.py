from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from modules.phase2.frame_builder import EventTimeBucketizer


@dataclass
class Phase2KeyState:
    key: str
    bucketizer: EventTimeBucketizer
    last_seen_wall: float
    user_state: Dict[str, Any] = field(default_factory=dict)


class Phase2StateStore:
    """Bounded per-key state store with TTL eviction and LRU trimming.

    Notes:
    - Uses wall time for TTL/LRU management (independent of event-time).
    - Single-threaded access is assumed (Phase2 worker loop).
    """

    def __init__(
        self,
        *,
        max_active_keys: int = 5000,
        key_ttl_sec: int = 3600,
    ) -> None:
        self.max_active_keys = max(1, int(max_active_keys))
        self.key_ttl_sec = max(60, int(key_ttl_sec))
        self._od: "OrderedDict[str, Phase2KeyState]" = OrderedDict()

    def size(self) -> int:
        return len(self._od)

    def _evict_expired(self) -> int:
        now = time.time()
        removed = 0
        # ordered by LRU; stop early once fresh enough
        for k in list(self._od.keys()):
            st = self._od.get(k)
            if st is None:
                continue
            if (now - float(st.last_seen_wall)) > float(self.key_ttl_sec):
                self._od.pop(k, None)
                removed += 1
            else:
                # because LRU order, if oldest isn't expired we can break
                break
        return removed

    def _trim_lru(self) -> int:
        removed = 0
        while len(self._od) > self.max_active_keys:
            self._od.popitem(last=False)
            removed += 1
        return removed

    def get_or_create(
        self,
        *,
        key: str,
        bucketizer_factory,
    ) -> Tuple[Phase2KeyState, int, int]:
        """Return state; also evict (expired, lru_removed)."""
        expired = self._evict_expired()

        st = self._od.get(key)
        if st is None:
            b = bucketizer_factory()
            st = Phase2KeyState(key=key, bucketizer=b, last_seen_wall=time.time())
            self._od[key] = st
        else:
            st.last_seen_wall = time.time()
            # refresh LRU
            self._od.move_to_end(key, last=True)

        lru_removed = self._trim_lru()
        return st, expired, lru_removed
