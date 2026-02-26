import json
import os
from pathlib import Path
from copy import deepcopy

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


class ConfigReader:
    """
    Loads utils/config.json and applies environment variable overrides.

    Env vars (recommended):
      - MSF_CASSANDRA_HOST, MSF_CASSANDRA_USERNAME, MSF_CASSANDRA_PASSWORD, MSF_CASSANDRA_KEYSPACE
      - MSF_KAFKA_BOOTSTRAP_SERVERS, MSF_KAFKA_SASL_USERNAME, MSF_KAFKA_SASL_PASSWORD
    """

    def __init__(self):
        self.BASE_DIR = BASE_DIR
        with open(f"{BASE_DIR}/config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Normalize + deep copy
        self._cfg = deepcopy(cfg)

        # --- Environment mode switch (live vs test) ---
        # Single config file contains both profiles:
        #   - consumer_props/producer_props: LIVE (real Kafka)
        #   - test_consumer_props/test_producer_props: TEST (localhost Kafka)
        # Switch by config "environment_mode" or env var MSF_ENVIRONMENT_MODE.
        mode = str(_env("MSF_ENVIRONMENT_MODE", self._cfg.get("environment_mode", "live"))).strip().lower()
        self._cfg["environment_mode"] = mode

        # Snapshot live props from file (canonical)
        # Prefer explicit live_* snapshots if present; fall back to legacy consumer_props/producer_props.
        live_cons = deepcopy(self._cfg.get("live_consumer_props") or self._cfg.get("consumer_props") or {})
        live_prod = deepcopy(self._cfg.get("live_producer_props") or self._cfg.get("producer_props") or {})

        # Keep explicit snapshots available for debugging/inspection.
        self._cfg["live_consumer_props"] = deepcopy(live_cons)
        self._cfg["live_producer_props"] = deepcopy(live_prod)
        test_cons = deepcopy(self._cfg.get("test_consumer_props") or {})
        test_prod = deepcopy(self._cfg.get("test_producer_props") or {})
        test_overrides = deepcopy(self._cfg.get("test_overrides") or {})

        if mode in ("test", "local"):
            # Activate test kafka props (fallback to live if missing)
            self._cfg["consumer_props"] = deepcopy(test_cons or live_cons)
            self._cfg["producer_props"] = deepcopy(test_prod or live_prod)

            # Apply test-only non-kafka knobs (baseline intervals, raw persist disable, etc.)
            for k, v in (test_overrides or {}).items():
                self._cfg[k] = v
        else:
            # Force live props active
            self._cfg["consumer_props"] = deepcopy(live_cons)
            self._cfg["producer_props"] = deepcopy(live_prod)


        # --- Validate Kafka consumer config ---
        # confluent_kafka.Consumer requires group.id for subscribe(); missing group.id yields 'Local: Unknown group'.
        try:
            gid = (self._cfg.get("consumer_props") or {}).get("group.id")
            if gid in (None, "", "None"):
                raise ValueError("Missing required Kafka consumer property: group.id")
        except Exception as e:
            raise RuntimeError(
                "Invalid Kafka consumer config (group.id). Ensure config.json has live_consumer_props/test_consumer_props with group.id. "
                "This project runs in '" + str(mode) + "' mode."
            ) from e

        # --- Normalize / alias keys to reduce config.json complexity ---
        # Phase3: keep legacy keys consistent with the canonical ones used in code.
        # baseline interval: prefer explicit phase3_heavy_every_sec; otherwise fall back to legacy mode_b name.
        if "phase3_heavy_every_sec" not in self._cfg and "phase3_modeb_baseline_every_sec" in self._cfg:
            self._cfg["phase3_heavy_every_sec"] = self._cfg.get("phase3_modeb_baseline_every_sec")

        # WS-only gating: canonical key is phase3_mode_b_ws_only; legacy key was phase3_modeb_disable_pid.
        if "phase3_mode_b_ws_only" not in self._cfg and "phase3_modeb_disable_pid" in self._cfg:
            try:
                self._cfg["phase3_mode_b_ws_only"] = bool(self._cfg.get("phase3_modeb_disable_pid"))
            except Exception:
                self._cfg["phase3_mode_b_ws_only"] = True

        # Always expose both legacy+canonical keys (kept in sync) so older modules don't diverge.
        self._cfg["phase3_modeb_disable_pid"] = bool(self._cfg.get("phase3_mode_b_ws_only", True))
        self._cfg["phase3_modeb_baseline_every_sec"] = int(self._cfg.get("phase3_heavy_every_sec", 0) or 0)

        # --- Stage0 durability boundary controls (canonical) ---
        # IMPORTANT: Stage0 is the durability boundary: Kafka offsets must be committed only after
        # RAW/projection persistence succeeds (commit-after-raw).
        #
        # Historically, this project used the name 'phase3_raw_persist_enabled' even though it gates
        # Stage0 RAW writes. We keep the legacy key for backwards compatibility, but the canonical keys are:
        #   - stage0_raw_persist_enabled
        #   - stage0_commit_enabled
        #   - stage0_commit_requires_raw_persist
        #   - stage0_allow_commit_without_raw   (UNSAFE; requires explicit opt-in)

        if "stage0_raw_persist_enabled" not in self._cfg:
            self._cfg["stage0_raw_persist_enabled"] = bool(self._cfg.get("phase3_raw_persist_enabled", True))

        if "stage0_commit_enabled" not in self._cfg:
            self._cfg["stage0_commit_enabled"] = True

        if "stage0_commit_requires_raw_persist" not in self._cfg:
            self._cfg["stage0_commit_requires_raw_persist"] = True

        if "stage0_allow_commit_without_raw" not in self._cfg:
            self._cfg["stage0_allow_commit_without_raw"] = False

        # Mirror legacy key so older modules remain consistent (do NOT let these diverge)
        self._cfg["phase3_raw_persist_enabled"] = bool(self._cfg.get("stage0_raw_persist_enabled", True))


        # --- Cassandra overrides ---
        cass = self._cfg.get("cassandra_props") or self._cfg.get("cassandra") or {}
        cass["host"] = _env("MSF_CASSANDRA_HOST", cass.get("host"))
        cass["username"] = _env("MSF_CASSANDRA_USERNAME", cass.get("username"))
        cass["password"] = _env("MSF_CASSANDRA_PASSWORD", cass.get("password"))
        cass["keyspace"] = _env("MSF_CASSANDRA_KEYSPACE", cass.get("keyspace"))

        # Keep both keys consistent
        self._cfg["cassandra_props"] = cass
        self._cfg["cassandra"] = {
            "host": cass.get("host"),
            "username": cass.get("username"),
            "password": cass.get("password"),
            "keyspace": cass.get("keyspace"),
        }

        # --- Kafka overrides (consumer/producer props) ---
        for props_key in ("consumer_props", "producer_props"):
            props = self._cfg.get(props_key) or {}
            props["bootstrap.servers"] = _env("MSF_KAFKA_BOOTSTRAP_SERVERS", props.get("bootstrap.servers"))

            # SASL auth is optional (only override if env provided)
            sasl_user = _env("MSF_KAFKA_SASL_USERNAME", None)
            sasl_pass = _env("MSF_KAFKA_SASL_PASSWORD", None)
            if sasl_user is not None:
                props["sasl.username"] = sasl_user
            if sasl_pass is not None:
                props["sasl.password"] = sasl_pass

            self._cfg[props_key] = props

        # Expose keys as attributes for backward compatibility with code that does cfg["x"]
        for k, v in self._cfg.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
