import json
import numpy as np
import os
import pytz
import pickle
import threading
from collections import deque
from datetime import datetime
##
from cassandra_utils.models.dw_raw_data import dw_raw_data
from cassandra_utils.models.dw_single_data import dw_tbl_raw_data  # Need to use raw data table here instead!
from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import generate_testing_dataframe, \
    type_casting_df
from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import check_model_exists
from thread.phase_2_multivariate_lstm_pipeline._2_5_driver_training import execute_training
from thread.phase_2_multivariate_lstm_pipeline._2_6_driver_inference import exe_live
##
from modules.kafka_modules import kafka_consumer3
from utils.logger_2 import setup_logger

from pathlib import Path

# Bu dosyanın bulunduğu yerden base_dir hesaplayalım
# thread/phase_2_multivariate_lstm_pipeline/_2_3_processing_layer.py
# buradan 2 seviye yukarı çıkınca pm-phase2 kökü:
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "phase2models"
THRESHOLD_DIR = MODEL_DIR / "threshold"
SCALER_DIR = MODEL_DIR / "scalers"
BUFFER_DIR = MODEL_DIR / "historical-buffer"

# Logger for debugging
p2_1_inf_log = setup_logger(
    "p2_1_ingestion_layer_inference_logger", "logs/p2_1_ingestion_layer_inference.log"
)

p2_1_agg_log = setup_logger(
    "p2_1_aggregation_logger", "logs/p2_1_aggregation.log"
)

# Global buffer for past 20 dataframes (N sequences for LSTM)
buffer_per_key = {}  # key -> deque
BUFFER_DIR = "models/phase2models/historical-buffer/"

# Lock mechanism to ensure one message is completely processed at a time
processing_lock = threading.Lock()

# PROCESS_CURRENT_DATE_ONLY = True
consumer3 = kafka_consumer3()


def update_buffer(key, new_data):
    if key not in buffer_per_key:
        buffer_per_key[key] = deque(maxlen=19)
    buffer_per_key[key].append(new_data)


def save_buffer_to_file(key):
    """Save the buffer for a specific key to the disk."""

    p2_1_inf_log.debug(f"Attempting to save buffer for key: {key}")

    if key in buffer_per_key:
        p2_1_inf_log.debug(f"Buffer for key {key} exists, proceeding to save...")
        buffer_path = os.path.join(BUFFER_DIR, f"{key}_buffer.pkl")

        # Ensure the directory exists
        os.makedirs(BUFFER_DIR, exist_ok=True)  # Create the directory if it does not exist

        p2_1_inf_log.debug(f"Saving buffer to file path: {buffer_path}")

        try:
            with open(buffer_path, 'wb') as f:
                p2_1_inf_log.debug(f"Serializing buffer for key {key}...")
                pickle.dump(buffer_per_key[key], f)
            p2_1_inf_log.debug(f"Buffer for key {key} saved successfully to {buffer_path}")

        except Exception as e:
            p2_1_inf_log.debug(f"Error while saving buffer for key {key}: {e}")

    else:
        p2_1_inf_log.debug(f"Error: Buffer for key {key} not found in buffer_per_key.")


def load_buffer_from_file(key):
    """Load the buffer for a specific key from the disk."""

    p2_1_inf_log.debug(f"Attempting to load buffer for key: {key}")

    if not os.path.exists(BUFFER_DIR):
        p2_1_inf_log.warning(f"Warning: The directory {BUFFER_DIR} does not exist. No buffers can be loaded.")
        return

    buffer_path = os.path.join(BUFFER_DIR, f"{key}_buffer.pkl")
    p2_1_inf_log.debug(f"Looking for buffer file at: {buffer_path}")

    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, 'rb') as f:
                p2_1_inf_log.debug(f"Deserializing buffer for key {key}...")
                buffer_per_key[key] = pickle.load(f)
            p2_1_inf_log.debug(f"Buffer for key {key} loaded successfully from {buffer_path}")
        except Exception as e:
            p2_1_inf_log.debug(f"Error while loading buffer for key {key}: {e}")
    else:
        p2_1_inf_log.debug(f"No saved buffer found for key {key}.")


def _is_valid_message_date(message, today):
    """
    Checks if all 'measDt' values in 'outVals' exist and match today's date.
    Returns True if valid, False otherwise.
    """
    for out_val in message.get('outVals', []):
        meas_dt_ms = out_val.get('measDt')
        if meas_dt_ms is not None:
            meas_date = datetime.fromtimestamp(meas_dt_ms / 1000, tz=pytz.UTC).date()
            if meas_date != today:
                p2_1_inf_log.warning("Date does not match today")
                return False  # Date does not match today, invalid message
        else:
            return False  # measDt missing, invalid message
    return True


def _is_message_valid(message):
    """
    Validates required fields in message.
    Returns True if valid, False otherwise.
    """

    # 1) outVals → cust alanı boş mu?
    if message.get("outVals") and message["outVals"][0].get("cust") is None:
        p2_1_inf_log.warning("[execute_phase_three] Skipping message: 'cust' field missing in outVals")
        return False

    # 2) plId → varsa None olamaz
    if "plId" in message and message["plId"] is None:
        p2_1_inf_log.info("[execute_phase_three] Skipping message: plId is None")
        return False

    # 3) wsId → varsa None olamaz
    if "wsId" in message and message["wsId"] is None:
        p2_1_inf_log.info("[execute_phase_three] Skipping message: wsId is None")
        return False

    # 4) joOpId → varsa None olamaz
    if "joOpId" in message and message["joOpId"] is None:
        p2_1_inf_log.info("[execute_phase_three] Skipping message: joOpId is None")
        return False

    return True


# Main Logic of the system
def consumer_preprocess_2():
    p2_1_inf_log.debug(" in consuming ")
    try:
        while True:

            # Step 1 - Fetching Message from Topic
            p2_1_inf_log.debug("2.1 STEP 1- Fetching Message from Topic")
            msg = consumer3.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                p2_1_inf_log.error(f"phase2_consumer - Consumer error: {msg.error()}")
                continue

            try:
                # Step 2 Processing Message - Unloading in json format
                p2_1_inf_log.debug("2.1 STEP 2 - Unloading in json format")
                raw_value = msg.value().decode('utf-8')
                message = json.loads(raw_value)

                """message["wsId"] = 441165
                message["wsNo"] = "MARIFARM"
                message["wsNm"] = "MARIFARM"

                message["wcNm"] = "WORKCENTER"
                message["wcNo"] = "WC1"
                message["wcId"] = 951"""

                # Belirli müşteri skip
                if _is_message_valid(message=message) is False:
                    p2_1_inf_log.info("Invalid message encountered, skipping to next message")
                    continue  # Skip to the next message

                """if message.get("outVals") and message["outVals"][0].get("cust") == "teknia_group":
                    p2_1_inf_log.info("[execute_phase_three] Skipping teknia_group customer")
                    continue"""

                if message.get("plId") and message["plId"] == 20:
                    p2_1_inf_log.info("[execute_phase_three] Skipping Savola customer")
                    continue

                """if message.get("plId") and message["plId"] != 155:
                    p2_1_inf_log.info("[execute_phase_three] Skipping Savola customer")
                    continue"""

                # Defensive: ensure inputVariableList key is present and is a list
                if 'inputVariableList' not in message or message['inputVariableList'] is None:
                    message['inputVariableList'] = []

                p2_1_inf_log.debug(f"phase2_consumer - Consumed message: {message}")

                # Step 3 Filter messages by date -  # START WHEN TOPIC IS INITIALIZED
                p2_1_inf_log.debug("2.1 STEP 3 - Date filter XX measurement date wise")
                p2_1_inf_log.debug(datetime.fromtimestamp(int(message['crDt']) / 1000, tz=pytz.UTC).date())
                # today = datetime.now(pytz.UTC).date()
                # if not _is_valid_message_date(message,today):
                #     p2_1_inf_log.info("Old message encountered, stopping consumer loop")
                #     break  # Stop processing further messages

                if isinstance(message, list):
                    p2_1_inf_log.error("phase2_consumer - Message is a list, skipping")
                    continue

                # Step 4    Saving message to Cassandra
                p2_1_inf_log.debug("2.1 STEP 4 - Saving message to Cassandra")
                data = dw_raw_data.saveData(message)
                p2_1_inf_log.debug(f"phase2_consumer - Data saved in Cassandra: {data}")

                print(message)

                # Step 5 : Extract key from Message based on combination of WorkstationID and StockID
                p2_1_inf_log.debug("2.1 STEP 5 - Extract key")
                wsid = message['wsId']
                stid = message['prodList'][0]['stId'] if message['prodList'] else None
                key = f"{wsid}_{stid}"

                p2_1_inf_log.debug(f"The key for current execution :{key}")

                # Step 6: Check if Pretrained model for key exists otherwise train it
                p2_1_inf_log.debug(f"2.1 STEP 6 - Check if Pretrained model for key : {key}")
                if not check_model_exists(key):
                    p2_1_inf_log.debug(
                        f"No previous Model found for key, Training a fresh LSTM model for key : {key}"
                    )
                    last_n_seqs = execute_training(key)

                    if last_n_seqs is None:
                        p2_1_inf_log.warning(
                            f"Not enough historical data to train model for key {key}. "
                            f"Skipping training/inference for now."
                        )
                        consumer3.commit()
                        continue

                    p2_1_inf_log.debug(f"Completed Training Phase for key '{key}'")
                    for seq in last_n_seqs:
                        update_buffer(key, seq)

                    save_buffer_to_file(key)
                    p2_1_inf_log.debug("Buffer for last n sequences saved to directory")

                else:
                    p2_1_inf_log.debug(f"Model already exists for key {key}")
                    # Step 6b : If pretrained model exists continue with inference
                    p2_1_inf_log.debug("2.1 STEP 6B - Model Exists")
                    p2_1_inf_log.debug(f"2.1 STEP 6C - output value list -> {data.outputvaluelist}")
                    current_df = generate_testing_dataframe(data)
                    p2_1_inf_log.debug(f"2.1 STEP 6D - created dataframe current_df -> {current_df.head()}")

                    current_df = type_casting_df(current_df)
                    # p2_1_inf_log.debug(f"inputs -> {current_df}")
                    current_row = current_df.values[0]

                    # ---- SADECE NUMERIK KOLONLARDA NaN / Inf KONTROLÜ ----
                    numeric_df = current_df.select_dtypes(include=[np.number])

                    if numeric_df.empty:
                        p2_1_inf_log.warning(
                            f"No numeric columns found in current_df for key={key}. "
                            f"Skipping inference. dtypes=\n{current_df.dtypes}"
                        )
                        consumer3.commit()
                        continue

                    has_nan = numeric_df.isna().values.any()
                    has_inf = not np.isfinite(numeric_df.to_numpy()).all()

                    if not has_nan and not has_inf:
                        p2_1_inf_log.debug(
                            f"Input data is clean: no NaNs or Infs. "
                            f"shape={numeric_df.shape}, cols={list(numeric_df.columns)}"
                        )

                        # Step 7 : Fetch Last N Sequences and send it with current to Model
                        p2_1_inf_log.debug("2.1 STEP 7 - Fetching N seqs")
                        if key in buffer_per_key:
                            p2_1_inf_log.debug("2.1 STEP 7A- found in buffer")
                            p2_1_inf_log.debug(
                                f"{key} key found in buffer for last n sequences executing prediction"
                            )
                            exe_live(current_df, buffer_per_key[key], data, key, message)
                            p2_1_inf_log.debug("--- EXE_LIVE CALL FINISHED ---")
                            update_buffer(
                                key,
                                current_df.drop(columns=["timestamp"], errors="ignore").values[0]
                            )
                        else:
                            p2_1_inf_log.debug("2.1 STEP 7B - loading for dir")
                            p2_1_inf_log.error(f"Key not found in buffer for key:'{key}'")
                            load_buffer_from_file(key)
                            if key in buffer_per_key:
                                p2_1_inf_log.debug("key found in buffer (loading from dir) executing prediction")
                                exe_live(current_df, buffer_per_key[key], data, key, message)
                                p2_1_inf_log.debug("--- EXE_LIVE CALL FINISHED ---")

                                update_buffer(
                                    key,
                                    current_df.drop(columns=["timestamp"], errors="ignore").values[0]
                                )
                            else:
                                p2_1_inf_log.error(
                                    "Key not found in buffer(even after loading from directory)"
                                )
                    else:
                        p2_1_inf_log.warning(
                            f"Input data has NaNs or Infs. Skipping inference. "
                            f"has_nan={has_nan}, has_inf={has_inf}, key={key}"
                        )

                consumer3.commit()
                p2_1_inf_log.debug(
                    f"phase2_consumer - Offset committed for message offset: {msg.offset()}"
                )

            except Exception as e:
                p2_1_inf_log.error(f"phase2_consumer - Error processing message: {e}")
                continue

    except Exception as e:
        p2_1_inf_log.error(f"phase2_consumer - Fatal error in consumer loop: {e}")
        print(e)

    finally:
        consumer3.close()
        p2_1_inf_log.info("phase2_consumer - Consumer closed and resources cleaned up")


import json
import time

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import BatchStatement, SimpleStatement

from utils.config_reader import ConfigReader
from modules.kafka_modules import kafka_aggregate_consumer

cfg = ConfigReader()
cassandra_props = cfg["cassandra_props"]
_cluster = None
_session = None

consumer_agg = kafka_aggregate_consumer()


def get_cassandra_session():
    """
    Singleton gibi davran: tek bir Cluster/Session yarat, herkes onu kullansın.
    """
    global _cluster, _session

    if _session is not None:
        return _session

    auth_provider = PlainTextAuthProvider(
        username=cassandra_props["username"],
        password=cassandra_props["password"]
    )
    _cluster = Cluster(
        contact_points=[cassandra_props["cassandra_host"]],
        auth_provider=auth_provider
    )
    # İstersen keyspace'i burada da verebilirsin
    keyspace = cassandra_props.get("keyspace")
    if keyspace:
        _session = _cluster.connect(keyspace)
    else:
        _session = _cluster.connect()
    p2_1_agg_log.info("Cassandra session initialized via get_cassandra_session()")
    return _session


def shutdown_cassandra():
    """
    Uygulama kapanırken çağırmak için (isteğe bağlı).
    """
    global _cluster, _session
    if _session is not None:
        _session.shutdown()
        _session = None
    if _cluster is not None:
        _cluster.shutdown()
        _cluster = None
    p2_1_agg_log.info("Cassandra cluster/session shut down.")


def cassandra_aggregation_query(message_list):
    """
    Daha önce her seferinde yeni Cluster/Session açıyordu.
    Artık tekil session üzerinden batch insert yapıyor.
    message_list: List[(row_tuple, insert_query)]
    """
    if not message_list:
        return

    try:
        session = get_cassandra_session()

        batch = BatchStatement()
        for params, query in message_list:
            # query her bir kayıt için aynı da olabilir, farklı tablo da olabilir.
            stmt = SimpleStatement(query)
            batch.add(stmt, params)

        session.execute(batch)
    except Exception as e:
        p2_1_agg_log.error(f"Cassandra batch insert failed: {e}", exc_info=True)


def aggregate_consumer():
    message_list = []
    while True:
        try:
            p2_1_agg_log.info("STEP 0.0-aggregate_consumer - while True")
            msg_agg = consumer_agg.poll(1.0)
            if msg_agg is None:
                continue
            if msg_agg.error():
                p2_1_agg_log.error(f"STEP 0.3-aggregate_consumer - Consumer error: {msg_agg.error()}")
                continue

            p2_1_agg_log.info("STEP 0.1-aggregate_consumer - Message received. consumer_agg.poll(1.0)")
            # --- parse message & key safely ---
            value, key = get_json(msg_agg)
            topic = msg_agg.topic()
            partition = msg_agg.partition()
            offset = msg_agg.offset()
            p2_1_agg_log.info(f"STEP 0.8-aggregate_consumer Topic={topic} Part={partition} Off={offset} Key={key}")

            if not value:
                p2_1_agg_log.error("STEP 1-aggregate_consumer - Empty/invalid message value; skipping")
                continue

            # ------------ choose table by topic -------------
            if topic == "SENSOR_MULTI_ANOMALY_ONE_MIN_TABLE":
                table = cassandra_props["one_min_table"]
            elif topic == "SENSOR_MULTI_ANOMALY_TEN_MIN_TABLE":
                table = cassandra_props["ten_min_table"]
            elif topic == "SENSOR_MULTI_ANOMALY_ONE_HOUR_TABLE":
                table = cassandra_props["one_hour_table"]
            elif topic == "SENSOR_MULTI_ANOMALY_ONE_DAY_TABLE":
                table = cassandra_props["one_day_table"]
            elif topic == "SENSOR_MULTI_ANOMALY_ONE_WEEK_TABLE":
                table = cassandra_props["one_week_table"]
            # elif topic == "SENSOR_MULTI_ANOMALY_SHIFTLY_TABLE":
            # table = cassandra_props["shift_table"]
            else:
                p2_1_agg_log.error(f"STEP 2-aggregate_consumer - Unknown topic: {topic}")
                continue

            # ------------- build INSERT (explicit ordered columns) -------------
            columns = [
                "message_key",
                "start_time", "finish_time",

                "measurement_count",
                "anomaly_detected_count",
                "anomaly_not_detected_count",

                "anomaly_importance_avg", "anomaly_importance_max", "anomaly_importance_min",
                "anomaly_score_avg", "anomaly_score_max", "anomaly_score_min",

                # latest-by-offset fields
                "partition_date",
                "measurement_date",
                "unique_code",

                "active",
                "algorithm",

                "customer",
                "good",
                "heapmap_threshold",

                "job_order_operation_id",
                "job_order_operation_ref_id",

                "machine_state",

                "plant_id", "plant_name", "plant_no",

                "produced_stock_id", "produced_stock_name", "produced_stock_no",

                "production_order_ref_id",
                "quantity_changed",

                "workcenter_id", "workcenter_name", "workcenter_no",

                "workstation_id", "workstation_name", "workstation_no", "workstation_state",

                "current_quantity",
                "employee_id",

                "shift_finish_text", "shift_finish_time",
                "shift_start_text", "shift_start_time",

                # multi-sensor snapshots (maps)
                "heatmap",
                "sensor_values",
                "input_variables",

                "record_year",
            ]

            row = [
                key,  # message_key (from record key)

                dict_get_ci(value, "start_time"),
                dict_get_ci(value, "finish_time"),

                dict_get_ci(value, "measurement_count", default=0),
                dict_get_ci(value, "anomaly_detected_count", default=0),
                dict_get_ci(value, "anomaly_not_detected_count", default=0),

                dict_get_ci(value, "anomaly_importance_avg"),
                dict_get_ci(value, "anomaly_importance_max"),
                dict_get_ci(value, "anomaly_importance_min"),

                dict_get_ci(value, "anomaly_score_avg"),
                dict_get_ci(value, "anomaly_score_max"),
                dict_get_ci(value, "anomaly_score_min"),

                dict_get_ci(value, "partition_date"),
                dict_get_ci(value, "measurement_date"),
                dict_get_ci(value, "unique_code"),

                dict_get_ci(value, "active"),
                dict_get_ci(value, "algorithm"),

                dict_get_ci(value, "customer"),
                dict_get_ci(value, "good"),
                dict_get_ci(value, "heapmap_threshold"),

                dict_get_ci(value, "job_order_operation_id"),
                dict_get_ci(value, "job_order_operation_ref_id"),

                dict_get_ci(value, "machine_state"),

                dict_get_ci(value, "plant_id"),
                dict_get_ci(value, "plant_name"),
                dict_get_ci(value, "plant_no"),

                dict_get_ci(value, "produced_stock_id"),
                dict_get_ci(value, "produced_stock_name"),
                dict_get_ci(value, "produced_stock_no"),

                dict_get_ci(value, "production_order_ref_id"),
                dict_get_ci(value, "quantity_changed"),

                dict_get_ci(value, "workcenter_id"),
                dict_get_ci(value, "workcenter_name"),
                dict_get_ci(value, "workcenter_no"),

                dict_get_ci(value, "workstation_id"),
                dict_get_ci(value, "workstation_name"),
                dict_get_ci(value, "workstation_no"),
                dict_get_ci(value, "workstation_state"),

                dict_get_ci(value, "current_quantity"),
                dict_get_ci(value, "employee_id"),

                dict_get_ci(value, "shift_finish_text"),
                dict_get_ci(value, "shift_finish_time"),
                dict_get_ci(value, "shift_start_text"),
                dict_get_ci(value, "shift_start_time"),

                # If Cassandra columns are TEXT -> store JSON string.
                # If Cassandra columns are nested MAP types and your driver supports dict binding,
                # replace json.dumps(...) with direct dict_get_ci(...)
                (json.dumps(dict_get_ci(value, "heatmap"), ensure_ascii=False)
                 if dict_get_ci(value, "heatmap") is not None else None),

                (json.dumps(dict_get_ci(value, "sensor_values"), ensure_ascii=False)
                 if dict_get_ci(value, "sensor_values") is not None else None),

                (json.dumps(dict_get_ci(value, "input_variables"), ensure_ascii=False)
                 if dict_get_ci(value, "input_variables") is not None else None),

                dict_get_ci(value, "record_year"),
            ]

            placeholders = ", ".join(["%s"] * len(columns))
            col_list = ", ".join(columns)
            insert_query = f"INSERT INTO {cassandra_props['keyspace']}.{table} ({col_list}) VALUES ({placeholders})"

            message_list.append((tuple(row), insert_query))

            # ------------- batch flush -------------
            p2_1_agg_log.info(f"STEP 3-aggregate_consumer - message_list size: {len(message_list)}")
            if len(message_list) >= cfg["cassandra_batch_size"]:
                t0 = time.time()
                cassandra_aggregation_query(message_list)
                message_list.clear()
                t1 = time.time()
                p2_1_agg_log.info(f"STEP 3-aggregate_consumer - Batch write took {t1 - t0:.3f}s")

        except Exception as e:
            p2_1_agg_log.error(f"aggregate_consumer loop error: {e}", exc_info=True)


# ---------- helpers ----------
def get_json(msg):
    """Return (value_dict, key_str). Key may be None."""
    try:
        value_bytes = msg.value()
        val = json.loads(value_bytes.decode("utf-8")) if value_bytes else {}
    except Exception as e:
        p2_1_agg_log.error(f"JSON decode failed: {e!r}; raw={msg.value()!r}")
        val = {}
    key_bytes = msg.key()
    key = None
    if "message_key" in val:
        key = val["message_key"]
    elif "MESSAGE_KEY" in val:
        key = val["MESSAGE_KEY"]
    elif key_bytes is not None:
        try:
            key = key_bytes.decode("utf-8")
        except Exception:
            try:
                key = json.loads(key_bytes.decode("utf-8"))
            except Exception:
                key = str(key_bytes)
    return val, key


def dict_get_ci(d, *names, default=None):
    """Case-insensitive getter: tries provided names (any case), then lowercased variants."""
    for n in names:
        if n in d:
            return d[n]
        ln = n.lower()
        un = n.upper()
        if ln in d:
            return d[ln]
        if un in d:
            return d[un]
    return default