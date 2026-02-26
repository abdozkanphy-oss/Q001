from concurrent.futures import ThreadPoolExecutor
from utils.config_reader import ConfigReader

from runtime.kafka_fanout import execute_kafka_fanout

from thread.phase_2_multivariate_lstm_pipeline.phase2_runtime import execute_phase_two_worker
# Phase3 v2 redesign entrypoint (Patch0)
from thread.phase_3_v2.phase3_v2_runtime import execute_phase_three_worker

from thread.phase_trigger_bus_worker import execute_phase_trigger_bus_worker
from thread.training_orchestrator_worker import execute_training_orchestrator_worker
from utils.keypoint_recorder import execute_keypoint_reporter

cfg = ConfigReader()
threads = []

def _get_int(key: str, default: int = 0) -> int:
    try:
        return int(cfg[key])
    except Exception:
        return default

# ensure max_workers is large enough for enabled threads
kafka_n = _get_int("kafka_fanout_thread", 1)
phase2_n = _get_int("execute_phase_two_thread", 1)
phase3_n = _get_int("execute_phase_three_thread", 1)
trig_n = _get_int("phase_trigger_bus_thread", 1)
orch_n = _get_int("training_orchestrator_thread", 1)
kp_n = _get_int("keypoint_reporter_thread", 0)

required_workers = sum(max(0, x) for x in [kafka_n, phase2_n, phase3_n, trig_n, orch_n, kp_n])
max_workers = max(_get_int("thread_num", 6), required_workers)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 1) Kafka fanout thread (single consumer)
    for _ in range(max(0, kafka_n)):
        threads.append(executor.submit(execute_kafka_fanout))
    # 2) Phase2 worker(s)
    for _ in range(max(0, phase2_n)):
        threads.append(executor.submit(execute_phase_two_worker))
    # 3) Phase3 worker(s) (compute threads, not Kafka consumers)
    for _ in range(max(0, phase3_n)):
        threads.append(executor.submit(execute_phase_three_worker))
    # 4) Trigger Bus maintenance
    for _ in range(max(0, trig_n)):
        threads.append(executor.submit(execute_phase_trigger_bus_worker))
    # 5) Training Orchestrator
    for _ in range(max(0, orch_n)):
        threads.append(executor.submit(execute_training_orchestrator_worker))

    # 6) Keypoint reporter
    for _ in range(max(0, kp_n)):
        threads.append(executor.submit(execute_keypoint_reporter))