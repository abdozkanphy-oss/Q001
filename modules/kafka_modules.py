from confluent_kafka import Consumer, OFFSET_BEGINNING
from utils.config_reader import ConfigReader
from utils.logger import logger, logger_agg
import time

cfg = ConfigReader()
topic = cfg["consume_topic_phase3"]

consumer_props = cfg["consumer_props"].copy()
consumer_props['group.id'] = consumer_props.get('group.id')
consumer_props['auto.offset.reset'] = 'earliest'

# Optional: force a seek to beginning on partition assignment (replay without changing group.id).
# WARNING: if enabled, this will replay from earliest on every restart until disabled.
STAGE0_SEEK_TO_EARLIEST = bool(getattr(cfg, 'stage0_seek_to_earliest_on_start', False))

from confluent_kafka import TopicPartition

def _fmt_partitions(partitions):
    """Render TopicPartition lists without relying on confluent_kafka's repr.

    Some Windows builds display raw printf placeholders (e.g. %I32d, %s) in repr.
    This keeps logs readable and stable.
    """
    out = []
    try:
        for p in (partitions or []):
            out.append({
                "topic": getattr(p, "topic", None),
                "partition": getattr(p, "partition", None),
                "offset": getattr(p, "offset", None),
                "leader_epoch": getattr(p, "leader_epoch", None),
                "error": (str(getattr(p, "error", None)) if getattr(p, "error", None) else None),
            })
    except Exception:
        return str(partitions)
    return out

def _on_assign(consumer, partitions):
    logger_agg.info(f"[kafka_consumer3] ASSIGNED partitions={_fmt_partitions(partitions)}")
    # If requested, override committed offsets and replay from beginning.
    if STAGE0_SEEK_TO_EARLIEST:
        try:
            for p in partitions:
                p.offset = OFFSET_BEGINNING
            consumer.assign(partitions)
            logger_agg.warning("[kafka_consumer3] stage0_seek_to_earliest_on_start=true -> seeking to OFFSET_BEGINNING for all assigned partitions")
        except Exception as e:
            logger_agg.error(f"[kafka_consumer3] seek-to-beginning failed: {e}")
    else:
        # Explicit assign to make assignment behavior deterministic across platforms.
        try:
            consumer.assign(partitions)
        except Exception:
            pass
    try:
        committed = consumer.committed(partitions, timeout=10)
        logger_agg.info(f"[kafka_consumer3] COMMITTED={_fmt_partitions(committed)}")
    except Exception as e:
        logger_agg.warning(f"[kafka_consumer3] committed() failed: {e}")

def _on_revoke(consumer, partitions):
    logger_agg.warning(f"[kafka_consumer3] REVOKED partitions={_fmt_partitions(partitions)}")


def kafka_consumer3(group_id_override=None):
    try:
        logger_agg.info("STEP 1 - Kafka consumer-kafka_consumer phase 3 initialized.")

        props = consumer_props.copy()
        props["enable.auto.commit"] = False

        if group_id_override:
            props["group.id"] = group_id_override

        consumer = Consumer(props)
        logger_agg.info("STEP 2 - Kafka consumer-kafka_consumer phase 3 initialized.")

        consumer.subscribe([topic], on_assign=_on_assign, on_revoke=_on_revoke)
        logger_agg.info("STEP 3 - Kafka consumer-kafka_consumer phase 3 subscribed to topic: " + str(cfg["consume_topic_phase3"]))
        
        return consumer

    except Exception as e:
        logger_agg.error("An error occurred during Kafka consumer-kafka_consumer initialization or subscription: " + str(e))
        logger_agg.debug("Error details:", exc_info=True)
        
        return None