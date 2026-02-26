import os
import numpy as np
import pandas as pd

from cassandra_utils.models.dw_tbl_multiple_anomalies3 import DwTblMultipleAnomalies
from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import (
    generate_sequences_only,
    scale_live_data_from_dir
)
from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import (
    predict_with_model,
    MODEL_DIR,
)
from thread.phase_2_multivariate_lstm_pipeline._2_4_post_processing_layer import (
    compute_reconstruction_error,
    live_anomaly_detection,
    compute_anomaly_importance,
    sensorwise_error_score,
)
from modules.kafka_modules import kafka_producer
from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger
from pathlib import Path

# thread/phase_2_multivariate_lstm_pipeline/_2_3_processing_layer.py
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "phase2models"
THRESHOLD_DIR = MODEL_DIR / "threshold"
SCALER_DIR = MODEL_DIR / "scalers"
BUFFER_DIR = MODEL_DIR / "historical-buffer"

p2_5_driver_inf_log = setup_logger(
    "p2_5_driver_inf_log", "logs/p2_5_driver_inf.log"
)

cfg = ConfigReader()
producer = kafka_producer()


def clean_history(X_history, X_in):
    ts = X_in.index[0] if hasattr(X_in.index, '__getitem__') else None
    new_row = X_in.drop(columns=["timestamp"], errors="ignore").to_numpy().flatten()

    # --- Determine expected length dynamically ---
    lengths = [len(np.array(r).flatten()) for r in X_history] + [len(new_row)]
    expected_len = max(set(lengths), key=lengths.count)

    cleaned_history = []
    timestamps = []

    for idx, row in enumerate(X_history):
        arr = np.array(row).flatten()

        ts_candidate = None
        if arr.dtype == object:
            ts_candidate = next(
                (x for x in arr if isinstance(x, (pd.Timestamp, np.datetime64))),
                None
            )
            arr = np.array([x for x in arr
                            if not isinstance(x, (pd.Timestamp, np.datetime64))])

        if len(arr) != expected_len:
            if len(arr) > expected_len:
                arr = arr[-expected_len:]
            else:
                arr = np.pad(arr, (expected_len - len(arr), 0), constant_values=np.nan)

        cleaned_history.append(arr.astype(float))
        timestamps.append(ts_candidate)

    if len(new_row) != expected_len:
        if len(new_row) > expected_len:
            new_row = new_row[-expected_len:]
        else:
            new_row = np.pad(new_row, (expected_len - len(new_row), 0),
                             constant_values=np.nan)

    cleaned_history.append(new_row.astype(float))
    timestamps.append(ts)

    X = pd.DataFrame(np.vstack(cleaned_history))
    p2_5_driver_inf_log.debug(f"CLEAN HISTORY X.columns : \n{X.columns} ")
    p2_5_driver_inf_log.debug(f"CLEAN HISTORY X_in.columns : \n{X_in.columns} ")

    X_in = X_in.drop(columns=["timestamp"], errors="ignore")
    X.columns = X_in.columns
    return cleaned_history, timestamps, X


def exe_live(X_in, X_history, main_data, key, message):
    try:
        p2_5_driver_inf_log.debug("2.6 STEP 1")
        p2_5_driver_inf_log.debug(f"Initializing prediction for {main_data} & key {key}")
        p2_5_driver_inf_log.debug(f"Inputs received : \n{X_in} ")

        try:
            p2_5_driver_inf_log.debug(f"DEBUG LINE | X_in -->> : \n{X_in.head()} ")
            X_history, ts_history, X = clean_history(X_history, X_in)
            p2_5_driver_inf_log.debug(f"DEBUG LINE | X -->> : \n{X.head()} ")

            p2_5_driver_inf_log.debug("df generated")
        except Exception as e:
            p2_5_driver_inf_log.debug("df missed")
            p2_5_driver_inf_log.error(
                f"Error in merging live data with historical N sequences: {e}",
                exc_info=True
            )
            return

        # Step 2: Scaling
        p2_5_driver_inf_log.debug("2.6 STEP 2 - Scaling")
        X_scaled = scale_live_data_from_dir(X, key)
        if X_scaled is None:
            p2_5_driver_inf_log.debug(f"X_scaled is null")
            return

        # Step 3: Sequence Generation/Validation
        p2_5_driver_inf_log.debug("2.6 STEP 3 - Seq Gen & Validation")
        X_sequences = generate_sequences_only(X_scaled, timesteps=20)
        p2_5_driver_inf_log.debug(
            f"Inputs scaled and sequences generated from X_history. "
            f"X_sequences shape={X_sequences.shape}"
        )

        if len(X_sequences) == 0:
            p2_5_driver_inf_log.warning(
                f"No sequences generated for key={key}, skipping live inference."
            )
            return

        # Step 4: Initializing Model from directory
        p2_5_driver_inf_log.debug("2.6 STEP 4 - Initializing Model from directory")
        p2_5_driver_inf_log.debug("Initiazling Model from directory ...")
        model_dir = MODEL_DIR
        model_files = sorted(
            f for f in os.listdir(model_dir)
            if f.startswith(str(key)) and f.endswith(".keras")
        )

        if not model_files:
            p2_5_driver_inf_log.error(
                f"No model file found for key={key} in {model_dir}"
            )
            return

        latest_model = model_files[-1]
        model_path = os.path.join(model_dir, latest_model)
        p2_5_driver_inf_log.debug(
            f"Using model for live inference: {model_path}"
        )

        # Step 5: Live prediction
        p2_5_driver_inf_log.debug("2.6 STEP 5 - Initializing Live Prediction")
        p2_5_driver_inf_log.debug("Initialzing Live Prediction")
        prediction_live = predict_with_model(model_path, X_sequences)

        # Step 6: Reconstruction error & anomaly
        p2_5_driver_inf_log.debug("Computing Reconstruction Error")
        recon_error_live = compute_reconstruction_error(X_sequences, prediction_live)

        last_recon_error = float(recon_error_live[-1])
        p2_5_driver_inf_log.debug(
            f"Reconstruction Error (last 5) = {recon_error_live[-5:]}, "
            f"last={last_recon_error}"
        )

        anomaly_flags = live_anomaly_detection(recon_error_live, key)
        anomaly_live = bool(anomaly_flags[-1])

        p2_5_driver_inf_log.debug(
            f"Anomaly flags (last 5) = {anomaly_flags[-5:]}, "
            f"final flag={anomaly_live}"
        )

        anomaly_imp_array, threshold = compute_anomaly_importance(
            recon_error_live, key
        )
        anomaly_imp = float(anomaly_imp_array[-1])

        p2_5_driver_inf_log.debug(
            f"Anomaly importance (last 5) = {anomaly_imp_array[-5:]}, "
            f"final importance={anomaly_imp}, threshold={threshold}"
        )

        # Step 7: Heatmap
        p2_5_driver_inf_log.debug("2.6 STEP 7 - Heatmap")
        heatmap_values = sensorwise_error_score(X_sequences, prediction_live)

        sensor_values = {}
        # main_data -> ORM obj, o yüzden attribute gibi değil, list gibi kullanıyorsun,
        # burada main_data'nin raw dict hali geldiğini varsayıyorum:
        for item in main_data["outputvaluelist"]:
            eq_no = item.get("eqNo")
            sensor_values[str(eq_no)] = {k: str(v) for k, v in item.items()}

        heatmap_sensordata = {
            eq_no: {**sensor_data, "cntRead": str(heatmap_values[i])}
            for i, (eq_no, sensor_data) in enumerate(sensor_values.items())
        }

        # Step 9: Save to Cassandra
        p2_5_driver_inf_log.debug(
            f"Saving live inference to Cassandra. "
            f"score={last_recon_error}, detected={anomaly_live}, "
            f"importance={anomaly_imp}"
        )

        try:
            DwTblMultipleAnomalies.saveData(
                topic_data=message,
                main_data=main_data,
                anomaly_score_=last_recon_error,
                anomaly_detected_=anomaly_live,
                anomaly_importance_=anomaly_imp,
                sensor_values_=sensor_values,
                heatmap_threshold_=threshold,
                heatmap_=heatmap_sensordata,
                key=key,
                producer=producer,
                produce_topic=cfg["produce_topic"]
            )
            p2_5_driver_inf_log.debug("Prediction - Step 10F")
            p2_5_driver_inf_log.debug("Finished Inference (Cassandra save OK)")
        except Exception as e:
            p2_5_driver_inf_log.error(
                f"Error while saving DwTblMultipleAnomalies for key={key}: {e}",
                exc_info=True
            )

    except Exception as e:
        p2_5_driver_inf_log.error(f"Error occured during exe_live {e}")
        return