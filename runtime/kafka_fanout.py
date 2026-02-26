# runtime/kafka_fanout.py
# Thread 1: Kafka ingest + durability boundary + fanout into Phase2 and Phase3 queues.

from __future__ import annotations

import json
import time
from typing import Optional

from confluent_kafka import TopicPartition

from modules.kafka_modules import kafka_consumer3
from modules.batching.batch_assigner import BatchAssigner
from modules.planning.training_planner import TrainingPlanner
from runtime.event_bus import enqueue_phase2, enqueue_phase3, qsize
from utils.config_reader import ConfigReader
from utils.identity import get_workstation_uid
from utils.logger_2 import setup_logger
from utils.keypoint_recorder import KP

log = setup_logger("kafka_fanout", "logs/kafka_fanout.log")

cfg = ConfigReader()

# Stage0 durability boundary toggles (canonical; see utils/config_reader.py)
STAGE0_RAW_PERSIST_ENABLED = bool(getattr(cfg, "stage0_raw_persist_enabled", getattr(cfg, "phase3_raw_persist_enabled", True)))
STAGE0_COMMIT_ENABLED = bool(getattr(cfg, "stage0_commit_enabled", True))
STAGE0_COMMIT_REQUIRES_RAW = bool(getattr(cfg, "stage0_commit_requires_raw_persist", True))
STAGE0_ALLOW_UNSAFE_COMMIT = bool(getattr(cfg, "stage0_allow_commit_without_raw", False))

# Optional plant filter:
# If stage0_plid_allowlist is set (e.g. [149]), Stage0 will only persist+fanout messages whose plId is in allowlist.
# Messages with missing/invalid plId are not filtered (pass-through) to preserve legacy behavior.
def _parse_int_or_none(x) -> Optional[int]:
    try:
        if x in (None, "", "None"):
            return None
        return int(x)
    except Exception:
        return None


def _parse_plid_allowlist(raw) -> Optional[set[int]]:
    if raw in (None, "", "None", [], {}):
        return None
    vals = []
    if isinstance(raw, (list, tuple, set)):
        vals = list(raw)
    elif isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        vals = [p for p in parts if p]
    else:
        vals = [raw]
    out: set[int] = set()
    for v in vals:
        i = _parse_int_or_none(v)
        if i is not None:
            out.add(i)
    return out or None


STAGE0_PLID_ALLOWLIST = _parse_plid_allowlist(getattr(cfg, "stage0_plid_allowlist", None))


# Optional Stage0 Kafka entry logging (for debugging / audit).
# NOTE: Logging full payloads can be very large. Prefer sampling (every_n>1) and truncation.
def _parse_str_list(raw) -> list[str]:
    if raw in (None, "", "None", [], {}):
        return []
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [p.strip() for p in str(raw).replace(";", ",").split(",")]
    out: list[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return out


STAGE0_LOG_KAFKA_ENTRY_EVERY_N = int(getattr(cfg, "stage0_log_kafka_entry_every_n", 0) or 0)
STAGE0_LOG_KAFKA_ENTRY_MAX_CHARS = int(getattr(cfg, "stage0_log_kafka_entry_max_chars", 8000) or 8000)
STAGE0_LOG_KAFKA_ENTRY_MODE = str(getattr(cfg, "stage0_log_kafka_entry_mode", "raw") or "raw").strip().lower()
STAGE0_LOG_KAFKA_ENTRY_REDACT_KEYS = set(_parse_str_list(getattr(cfg, "stage0_log_kafka_entry_redact_keys", None)))
STAGE0_LOG_KAFKA_ENTRY_INCLUDE_HEADERS = bool(getattr(cfg, "stage0_log_kafka_entry_include_headers", False))


def _decode_key_safe(k) -> Optional[str]:
    try:
        if k is None:
            return None
        if isinstance(k, (bytes, bytearray)):
            s = k.decode("utf-8", errors="ignore")
        else:
            s = str(k)
        s = (s or "").strip()
        return s or None
    except Exception:
        return None


def _truncate(s: str, max_chars: int) -> str:
    try:
        if max_chars <= 0:
            return ""
        if s is None:
            return ""
        if len(s) <= max_chars:
            return s
        cut = max_chars
        return s[:cut] + f"...<truncated {len(s) - cut} chars>"
    except Exception:
        return ""


def _redact_top_level(message: dict) -> dict:
    if not STAGE0_LOG_KAFKA_ENTRY_REDACT_KEYS:
        return message
    try:
        out = dict(message)
        for k in list(out.keys()):
            if k in STAGE0_LOG_KAFKA_ENTRY_REDACT_KEYS:
                out[k] = "***REDACTED***"
        return out
    except Exception:
        return message
    
def _drop_nones(obj):
    if isinstance(obj, dict):
        return {
            k: _drop_nones(v)
            for k, v in obj.items()
            if v is not None
        }
    elif isinstance(obj, list):
        return [
            _drop_nones(v)
            for v in obj
            if v is not None
        ]
    else:
        return obj

def _format_payload_for_log(raw_value: str, message: dict) -> str:
    mode = STAGE0_LOG_KAFKA_ENTRY_MODE
    if mode == "raw":
        return _truncate(raw_value, STAGE0_LOG_KAFKA_ENTRY_MAX_CHARS)
    if mode == "compact":
        try:
            m = _redact_top_level(message)
            compact = {
                k: v
                for k, v in m.items() 
                if k not in ("outVals", "inVars", "inVals")
            }
            # compact = _drop_nones(compact)
            # add sizes for big fields
            for k in ("outVals", "inVars", "inVals"):
                v = m.get(k)
                if isinstance(v, list):
                    compact[k + "_len"] = len(v)
                elif v is not None:
                    compact[k + "_type"] = type(v).__name__
            s = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            return _truncate(s, STAGE0_LOG_KAFKA_ENTRY_MAX_CHARS)
        except Exception:
            return _truncate(raw_value, STAGE0_LOG_KAFKA_ENTRY_MAX_CHARS)
    # default: json (redacted)
    try:
        m = _redact_top_level(message)
        s = json.dumps(m, ensure_ascii=False, separators=(",", ":"))
        return _truncate(s, STAGE0_LOG_KAFKA_ENTRY_MAX_CHARS)
    except Exception:
        return _truncate(raw_value, STAGE0_LOG_KAFKA_ENTRY_MAX_CHARS)


PHASE2_ENABLED = bool(getattr(cfg, "phase2_enabled", True))
PHASE3_ENABLED = True  # phase3 worker thread may be disabled separately; fanout keeps both by default

# Optional: only enqueue to workers for allowlisted workstations (raw persist still occurs)
def _parse_allowlist(raw) -> set[str]:
    if not raw:
        return set()
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw).split(",")
    out = set()
    for s in items:
        s = (s or "").strip()
        if s:
            out.add(s)
    return out

PHASE_ALLOWLIST = _parse_allowlist(getattr(cfg, "phase3_workstation_allowlist", None))

# M1 shadow (optional)
M1_ENABLE_TRAINING_PLANNER = bool(getattr(cfg, "m1_enable_training_planner", False))
M1_LOG_BATCH_CONTEXT = bool(getattr(cfg, "m1_log_batch_context", False))


def _looks_like_customer_token(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False

    # Reject common workstation-identity keys used in some replays:
    # - "149|951|441165"
    parts = s.split("|")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return False

    # - "149_WC951_WS441165" (canonical workstation_uid shape)
    if "_WC" in s and "_WS" in s:
        try:
            pref = s.split("_WC", 1)[0]
            if pref.isdigit():
                return False
        except Exception:
            pass

    # Too short is unlikely to be a real customer id
    if len(s) < 3:
        return False

    return True


def _extract_customer(kafka_key, message: dict) -> Optional[str]:
    """Extract customer identifier.

    Production truth should come from the message payload when available
    (outVals[*].cust), because Kafka keys are sometimes repurposed for replay keys
    (e.g., workstation identifiers) in test environments.
    """

    # 1) Explicit message fields
    customer = (message.get("customer") or message.get("cust"))
    if isinstance(customer, str):
        customer = customer.strip()
    if not customer:
        customer = None

    # 2) Payload outVals[*].cust
    if not customer and isinstance(message.get("outVals"), list) and message["outVals"]:
        for ov in message["outVals"]:
            if not isinstance(ov, dict):
                continue
            c = ov.get("cust")
            if isinstance(c, str):
                c = c.strip()
            if c:
                customer = c
                break

    # 3) Kafka key fallback (only if it looks like a customer token)
    if not customer and kafka_key is not None:
        try:
            if isinstance(kafka_key, (bytes, bytearray)):
                k = kafka_key.decode("utf-8", errors="ignore")
            else:
                k = str(kafka_key)
            k = (k or "").strip()
            if k and _looks_like_customer_token(k):
                customer = k
        except Exception:
            pass

    return customer or None


def _fill_compat_fields(message: dict) -> None:
    # convenience fields used across code paths
    prod0 = None
    if isinstance(message.get("prodList"), list) and message["prodList"]:
        if isinstance(message["prodList"][0], dict):
            prod0 = message["prodList"][0]

    st_no = (prod0 or {}).get("stNo") or (prod0 or {}).get("stId") or message.get("stNo")
    st_nm = (prod0 or {}).get("stNm") or message.get("stNm")
    if st_no is not None:
        message["output_stock_no"] = st_no
    if st_nm is not None:
        message["output_stock_name"] = st_nm

    # operation aliases
    if "opNm" in message and "operationname" not in message:
        message["operationname"] = message.get("opNm")
    if "opNo" in message and "operationno" not in message:
        message["operationno"] = message.get("opNo")
    if "opTc" in message and "operationtaskcode" not in message:
        message["operationtaskcode"] = message.get("opTc")


def _should_skip_reason(message: dict) -> Optional[str]:
    # Keep legacy hard-skip for known bad plant, to preserve behavior.
    pl_id = _parse_int_or_none(message.get("plId"))

    # Optional allowlist: if configured, drop messages from other plants (but still commit offsets).
    if STAGE0_PLID_ALLOWLIST is not None and pl_id is not None and pl_id not in STAGE0_PLID_ALLOWLIST:
        return "plid_filter"

    return None


def _should_skip(message: dict) -> bool:
    return _should_skip_reason(message) is not None


def _message_valid_minimal(message: dict) -> bool:
    # Minimal guard to prevent non-dict / non-JSON payloads.
    if not isinstance(message, dict):
        return False
    if not message.get("outVals") and not message.get("inVars") and not message.get("inVals"):
        # still allow if it has process/identity fields, but this is usually junk
        return False
    return True


def execute_kafka_fanout(group_id_override: Optional[str] = None) -> None:
    """
    - polls Kafka
    - enriches message (customer + M1 batch context + compat fields)
    - persists RAW (durability boundary)
    - commits offset only after successful raw write
    - fans out to phase2 + phase3 queues
    """
    log.info("[kafka_fanout] init")

    # ---- Stage0 durability boundary safety rails ----
    # Refuse to start in a configuration that would ACK Kafka messages without durable RAW persistence,
    # unless the operator explicitly opts into UNSAFE behavior.
    log.info(
        f"[kafka_fanout] stage0 settings: raw_persist={STAGE0_RAW_PERSIST_ENABLED} commit_enabled={STAGE0_COMMIT_ENABLED} "
        f"commit_requires_raw={STAGE0_COMMIT_REQUIRES_RAW} allow_unsafe_commit_without_raw={STAGE0_ALLOW_UNSAFE_COMMIT}"
    )

    if STAGE0_COMMIT_ENABLED and (not STAGE0_RAW_PERSIST_ENABLED) and STAGE0_COMMIT_REQUIRES_RAW and (not STAGE0_ALLOW_UNSAFE_COMMIT):
        msg = (
            "[kafka_fanout] INVALID CONFIG: stage0_commit_enabled=true but stage0_raw_persist_enabled=false. "
            "This would commit Kafka offsets without durable RAW persistence (data loss). "
            "Fix config: enable stage0_raw_persist_enabled OR disable stage0_commit_enabled for dry runs. "
            "If you truly want UNSAFE at-most-once behavior, set stage0_allow_commit_without_raw=true explicitly."
        )
        log.error(msg)
        raise RuntimeError(msg)

    if STAGE0_COMMIT_ENABLED and (not STAGE0_RAW_PERSIST_ENABLED) and STAGE0_ALLOW_UNSAFE_COMMIT:
        log.warning(
            "[kafka_fanout] UNSAFE MODE: committing Kafka offsets while RAW persistence is disabled. "
            "This is at-most-once (can lose messages)."
        )

    # Lazy-import Cassandra models only if RAW persistence is enabled. This allows safe dry-run execution
    # in environments without Cassandra dependencies (no commit by default).
    dw_tbl_raw_data = None
    if STAGE0_RAW_PERSIST_ENABLED:
        from cassandra_utils.models.dw_single_data import dw_tbl_raw_data as _dw_tbl_raw_data
        dw_tbl_raw_data = _dw_tbl_raw_data


    batch_assigner = BatchAssigner()
    training_planner = TrainingPlanner() if M1_ENABLE_TRAINING_PLANNER else None

    consumer = kafka_consumer3(group_id_override=group_id_override)
    if consumer is None:
        log.error("[kafka_fanout] consumer init failed")
        return

    none_counter = 0
    stage0_msg_counter = 0  # decoded+valid messages seen by Stage0

    NONE_LIMIT = 100
    NONE_LOG_EVERY = 30
    NO_ASSIGN_SLEEP_SEC = 1.0

    def _on_no_msg():
        nonlocal none_counter
        KP.inc("stage0.poll.none", 1)
        none_counter += 1
        if none_counter % NONE_LOG_EVERY == 0:
            try:
                parts = consumer.assignment()
                def _fmt(parts_):
                    out = []
                    try:
                        for p in (parts_ or []):
                            out.append({
                                "topic": getattr(p, "topic", None),
                                "partition": getattr(p, "partition", None),
                                "offset": getattr(p, "offset", None),
                                "leader_epoch": getattr(p, "leader_epoch", None),
                                "error": (str(getattr(p, "error", None)) if getattr(p, "error", None) else None),
                            })
                        return out
                    except Exception:
                        return str(parts_)

                log.info(f"[kafka_fanout] poll=None x{none_counter}, assignment={_fmt(parts)}")
                if not parts:
                    time.sleep(NO_ASSIGN_SLEEP_SEC)
                    return
                try:
                    committed = consumer.committed(parts, timeout=10)
                    pos = [TopicPartition(p.topic, p.partition, consumer.position([p])[0].offset) for p in parts]
                    log.info(f"[kafka_fanout] committed={_fmt(committed)} pos={_fmt(pos)}")
                except Exception as e:
                    log.warning(f"[kafka_fanout] committed/position check failed: {e}")
            except Exception as e:
                log.warning(f"[kafka_fanout] assignment() failed: {e}")

        if none_counter >= NONE_LIMIT:
            # avoid tight spin
            time.sleep(0.2)

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            _on_no_msg()
            continue

        none_counter = 0

        if msg.error():
            KP.inc("stage0.poll.error", 1)
            log.warning(f"[kafka_fanout] msg error={msg.error()}")
            continue

        try:
            raw_value = msg.value().decode("utf-8", errors="ignore")
            message = json.loads(raw_value)
            KP.inc("stage0.decode.ok", 1)

            # Optional: log Kafka entry payload at a controlled cadence.
            stage0_msg_counter += 1
            if STAGE0_LOG_KAFKA_ENTRY_EVERY_N > 0 and (stage0_msg_counter % STAGE0_LOG_KAFKA_ENTRY_EVERY_N == 0):
                try:
                    key_s = _decode_key_safe(msg.key())
                    ts_ms = None
                    try:
                        _tstype, _ts = msg.timestamp()
                        ts_ms = _ts
                    except Exception:
                        pass
                    headers = None
                    if STAGE0_LOG_KAFKA_ENTRY_INCLUDE_HEADERS:
                        try:
                            headers = msg.headers()
                        except Exception:
                            headers = None
                    payload_s = _format_payload_for_log(raw_value, message)
                    log.info(
                        f"[kafka_fanout] KAFKA_ENTRY n={stage0_msg_counter} topic={msg.topic()} "
                        f"part={msg.partition()} off={msg.offset()} ts_ms={ts_ms} key={key_s} "
                        f"headers={headers} value={payload_s}"
                    )
                    KP.inc("stage0.log.kafka_entry", 1)
                except Exception:
                    KP.inc("stage0.log.kafka_entry.fail", 1)

        except Exception as e:
            KP.inc("stage0.decode.fail", 1)
            log.error(f"[kafka_fanout] json decode failed: {e}")
            continue

        if not _message_valid_minimal(message):
            KP.inc("stage0.msg.invalid", 1)
            log.info("[kafka_fanout] skip invalid/minimal message")
            continue

        # customer from kafka key (source of truth)
        customer = _extract_customer(msg.key(), message)
        if customer:
            message["customer"] = customer
            message["cust"] = customer  # backward compat

        # Workstation identity normalization (canonical)
        try:
            message["_workstation_uid"] = get_workstation_uid(message)
        except Exception:
            pass

        # M1: batching + workstation identity
        try:
            bctx = batch_assigner.assign(message)
            KP.inc("stage0.batch_ctx.ok", 1)
            message["_workstation_uid"] = bctx.workstation_uid
            message["_batch_id"] = bctx.batch_id
            message["_batch_strategy"] = bctx.strategy
            message["_batch_confidence"] = bctx.confidence
            message["_phase_id"] = bctx.phase_id
            message["_event_ts_ms"] = bctx.event_ts_ms
            try:
                if bctx.event_ts_ms:
                    KP.observe("stage0.event_lag_ms", (time.time() * 1000.0) - float(bctx.event_ts_ms))
            except Exception:
                pass
            if M1_LOG_BATCH_CONTEXT:
                log.info(f"[kafka_fanout] bctx ws={bctx.workstation_uid} batch={bctx.batch_id} strat={bctx.strategy} conf={bctx.confidence:.3f}")
        except Exception as e:
            KP.inc("stage0.batch_ctx.fail", 1)
            log.error(f"[kafka_fanout] batch_assigner failed: {e}", exc_info=True)

        if training_planner is not None:
            try:
                training_planner.observe(message)
                training_planner.maybe_log(message.get("_workstation_uid"), log)
            except Exception as e:
                log.error(f"[kafka_fanout] training_planner failed: {e}", exc_info=True)

        _fill_compat_fields(message)

        skip_reason = _should_skip_reason(message)
        if skip_reason:
            KP.inc(f"stage0.skip.{skip_reason}", 1)
            if skip_reason == "plid_filter":
                log.info(
                    f"[kafka_fanout] Skipping message due to stage0_plid_allowlist (plId={message.get('plId')})"
                )
            else:
                log.info(f"[kafka_fanout] skipped by skip rules ({skip_reason})")

            # Intentionally ignored message: do not persist RAW and do not fanout,
            # but do commit offsets if commit is enabled to avoid getting stuck.
            if STAGE0_COMMIT_ENABLED:
                try:
                    with KP.timeit("stage0.commit_sec"):
                        consumer.commit(message=msg, asynchronous=False)
                    KP.inc(f"stage0.commit.ok_skip.{skip_reason}", 1)
                except Exception as e:
                    KP.inc("stage0.commit.fail", 1)
                    log.error(
                        f"[kafka_fanout] commit failed for skipped message (reason={skip_reason}): {e}",
                        exc_info=True,
                    )
            else:
                KP.inc("stage0.commit.skip.disabled", 1)

            continue

        # RAW persist (durability boundary)
        if STAGE0_RAW_PERSIST_ENABLED:
            try:
                with KP.timeit("stage0.raw_write_sec"):
                    _ = dw_tbl_raw_data.saveData(message)
                KP.inc("stage0.raw.ok", 1)
            except Exception as e:
                KP.inc("stage0.raw.fail", 1)
                log.error(f"[kafka_fanout] RAW write failed (will NOT commit): {e}", exc_info=True)
                continue

        # Commit after raw write (durability boundary)
        if STAGE0_COMMIT_ENABLED:
            if STAGE0_RAW_PERSIST_ENABLED:
                # Normal durable mode: commit only after RAW persistence succeeded.
                try:
                    with KP.timeit("stage0.commit_sec"):
                        consumer.commit(message=msg, asynchronous=False)
                    KP.inc("stage0.commit.ok", 1)
                except Exception as e:
                    KP.inc("stage0.commit.fail", 1)
                    log.error(
                        f"[kafka_fanout] commit failed (will reprocess on restart): {e}",
                        exc_info=True,
                    )
            else:
                # RAW persistence is disabled. Only allow commit if explicitly opted into UNSAFE mode.
                if STAGE0_ALLOW_UNSAFE_COMMIT:
                    try:
                        with KP.timeit("stage0.commit_sec"):
                            consumer.commit(message=msg, asynchronous=False)
                        KP.inc("stage0.commit.ok_unsafe_no_raw", 1)
                    except Exception as e:
                        KP.inc("stage0.commit.fail", 1)
                        log.error(
                            f"[kafka_fanout] UNSAFE commit failed (no raw persist): {e}",
                            exc_info=True,
                        )
                else:
                    # Should be unreachable due to init validation. Keep safe behavior anyway.
                    KP.inc("stage0.commit.skip.no_raw", 1)
        else:
            KP.inc("stage0.commit.skip.disabled", 1)

        # Fanout to worker queues (allowlist optional)
        ws_uid = message.get("_workstation_uid") or ""
        if PHASE_ALLOWLIST and ws_uid and ws_uid not in PHASE_ALLOWLIST:
            # still persisted raw; skip compute fanout
            KP.inc("stage0.fanout.skip_allowlist", 1)
            continue

        if PHASE2_ENABLED:
            ok2 = enqueue_phase2(message)
            if ok2:
                KP.inc("stage0.fanout.phase2.ok", 1)
            else:
                KP.inc("stage0.fanout.phase2.drop", 1)
                log.warning(f"[kafka_fanout] phase2 queue full -> drop ws={ws_uid} q={qsize()}")

        ok3 = enqueue_phase3(message)
        if ok3:
            KP.inc("stage0.fanout.phase3.ok", 1)
        else:
            KP.inc("stage0.fanout.phase3.drop", 1)
            log.warning(f"[kafka_fanout] phase3 queue full -> drop ws={ws_uid} q={qsize()}")
