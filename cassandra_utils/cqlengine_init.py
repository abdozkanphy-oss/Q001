# cassandra_utils/cqlengine_init.py
# Centralized Cassandra cqlengine bootstrap to avoid "setup() required" runtime failures.

from __future__ import annotations

import threading
from typing import Any, Optional

from cassandra.auth import PlainTextAuthProvider
from cassandra.cqlengine import connection
from cassandra.cqlengine.connection import CQLEngineException

from utils.config_reader import ConfigReader


_LOCK = threading.Lock()
_READY = False


def _parse_hosts(h: Any) -> list[str]:
    if isinstance(h, (list, tuple)):
        return [str(x).strip() for x in h if str(x).strip()]
    if isinstance(h, str):
        return [x.strip() for x in h.split(",") if x.strip()]
    return []


def ensure_cqlengine_setup(cfg: Optional[Any] = None) -> None:
    """Ensure cassandra.cqlengine has a default connection.

    Safe to call multiple times; will only initialize once per process.
    Raises RuntimeError with a clear message if config is incomplete or connection cannot be established.
    """
    global _READY
    if _READY:
        return

    with _LOCK:
        if _READY:
            return

        # If already set up, this should succeed.
        try:
            _ = connection.get_session()
            _READY = True
            return
        except Exception:
            pass

        if cfg is None:
            cfg = ConfigReader()

        cass = None
        try:
            cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}
        except Exception:
            # ConfigReader supports dict-like access; if not, fall back.
            try:
                cass = getattr(cfg, "cassandra_props", None) or getattr(cfg, "cassandra", None) or {}
            except Exception:
                cass = {}

        host = cass.get("host")
        keyspace = cass.get("keyspace")
        username = cass.get("username")
        password = cass.get("password")

        hosts = _parse_hosts(host)
        if not hosts or not keyspace:
            raise RuntimeError(
                f"[cqlengine_init] Missing Cassandra config. host={host!r} keyspace={keyspace!r}. "
                "Please set cassandra_props.host and cassandra_props.keyspace in utils/config.json."
            )

        auth_provider = None
        if username is not None and password is not None:
            auth_provider = PlainTextAuthProvider(username=str(username), password=str(password))

        # Register default connection for cqlengine
        connection.setup(
            hosts=hosts,
            default_keyspace=str(keyspace),
            auth_provider=auth_provider,
            protocol_version=4,
            lazy_connect=True,
            retry_connect=True,
            connect_timeout=20,
        )

        # Force initialization so failures are obvious at startup.
        try:
            _ = connection.get_session()
        except CQLEngineException as e:
            raise RuntimeError(f"[cqlengine_init] Cassandra cqlengine setup failed: {e}") from e

        _READY = True
