# import os
# import numpy as np
# import pandas as pd
#
# from cassandra_utils.models.dw_raw_data import dw_raw_data
# from cassandra_utils.models.dw_single_data import dw_tbl_raw_data  # şu an kullanmıyoruz ama dursun
# from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import (
#     generate_training_dataframe,
#     generate_sequences_with_timestamps,
#     has_nan_cntread,
#     scale_and_save_training_data,
# )
# from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import (
#     initial_train_and_save,
#     check_model_exists,
#     predict_with_model,
#     MODEL_DIR,
# )
# from thread.phase_2_multivariate_lstm_pipeline._2_4_post_processing_layer import (
#     compute_reconstruction_error,
#     compute_and_save_threshold,
#     detect_anomalies,
# )
# from utils.logger_2 import setup_logger
#
# from pathlib import Path
#
# # Bu dosyanın bulunduğu yerden base_dir hesaplayalım
# BASE_DIR = Path(__file__).resolve().parents[2]
# MODEL_DIR = BASE_DIR / "models" / "phase2models"
# THRESHOLD_DIR = MODEL_DIR / "threshold"
# SCALER_DIR = MODEL_DIR / "scalers"
# BUFFER_DIR = MODEL_DIR / "historical-buffer"
#
# p2_5_driver_tr_log = setup_logger(
#     "p2_5_driver_tr_log", "logs/p2_5_driver_tr.log"
# )
#
# # -------------------------- FETCH BASE DATA -------------------------- #
#
# def preload_training_data(values_to_fetch=200):
#     """
#     Son X kaydı Cassandra'dan getirir (tüm ws + stok kombinasyonları için).
#     """
#     try:
#         returnList, inputList, outputList, batchList = dw_raw_data.fetchData(values_to_fetch)
#         if not returnList:
#             p2_5_driver_tr_log.warning("2_1_1 - No data found in fetch call")
#             p2_5_driver_tr_log.error("2_1_1 - No data received for training")
#             raise Exception("No training data fetched")
#         else:
#             p2_5_driver_tr_log.debug("2_1_1 - Sucessfully preloading data")
#             return returnList, inputList, outputList, batchList
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"2_1_1 - Error preloading training data: {e}")
#
# # -------------------------- BUILD TRAINING DATAFRAME -------------------------- #
#
# def build_training_dataframe_from_raw(return_list, output_list):
#     """
#     Build DataFrame with sensor values AND quality labels (good, productionstate)
#     """
#     print("Starting dataframe generation...")
#     p2_5_driver_tr_log.debug(
#         f"build_training_dataframe_from_raw: got {len(return_list)} rows"
#     )
#
#     rows = []
#
#     for idx, (row_obj, outvals) in enumerate(zip(return_list, output_list)):
#         if not outvals:
#             continue
#
#         # Timestamp
#         ts = getattr(row_obj, "measurement_date", None)
#         if ts is None:
#             try:
#                 ts = pd.to_datetime(outvals[0].get("measDt"), unit="ms", utc=True)
#             except Exception:
#                 continue
#
#         # ========== EXTRACT QUALITY LABELS ==========
#         # Extract 'good' flag (boolean: TRUE/NULL in your table)
#         good_value = getattr(row_obj, "good", None)
#
#         # Extract 'productionstate' (string: "PRODUCTION", "STOPPED", etc.)
#         production_state = getattr(row_obj, "productionstate", None)
#
#         base = {
#             "timestamp": ts,
#             "plantid": getattr(row_obj, "plantid", None),
#             "workstationid": getattr(row_obj, "workstationid", None),
#             "outputstockid": getattr(row_obj, "outputstockid", None),
#             "good": good_value,                    # Boolean column
#             "productionstate": production_state,   # String column
#         }
#
#         # Sensor values
#         sensor_values = {}
#         for ov in outvals:
#             if not isinstance(ov, dict):
#                 continue
#
#             eq_id = ov.get("eqId")
#             cnt = ov.get("cntRead")
#
#             if eq_id is None or cnt is None:
#                 continue
#
#             col_name = f"sensor_value_{eq_id}"
#             sensor_values[col_name] = cnt
#
#         if not sensor_values:
#             continue
#
#         merged = {**base, **sensor_values}
#         rows.append(merged)
#
#     if not rows:
#         raise ValueError("No usable rows in build_training_dataframe_from_raw")
#
#     X = pd.DataFrame(rows)
#
#     # Sort by timestamp
#     if "timestamp" in X.columns:
#         X.sort_values("timestamp", inplace=True)
#         X.reset_index(drop=True, inplace=True)
#
#     p2_5_driver_tr_log.debug(
#         f"build_training_dataframe_from_raw: shape={X.shape}, "
#         f"columns={list(X.columns)}"
#     )
#
#     # Log label distribution
#     if 'good' in X.columns and 'productionstate' in X.columns:
#         good_counts = X['good'].value_counts(dropna=False)
#         state_counts = X['productionstate'].value_counts(dropna=False)
#         p2_5_driver_tr_log.info(f"[build_df] Good distribution:\n{good_counts}")
#         p2_5_driver_tr_log.info(f"[build_df] Production state distribution:\n{state_counts}")
#
#     return X
# # -------------------------- MAIN TRAINING DRIVER -------------------------- #
#
# TIMESTEPS = 20
#
# def execute_training(key: str):
#     p2_5_driver_tr_log.debug(f"[{key}] ---- execute_training START ----")
#
#     # ---------------- Step 1: base fetch ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 1 - Historical Data fetch (preload_training_data)")
#     try:
#         returnList, inputList, outputList, batchList = preload_training_data(20000)
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 1 FAILED in preload_training_data: {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 1 DONE - total rows fetched: {len(returnList)}")
#
#     # ---------------- Step 2: filter by key (wsId_stId) ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 2 - filtering by key")
#     matching_indices = []
#
#     for idx, item in enumerate(returnList):
#         try:
#             wsid_ = item.workstationid
#             stid_ = item.producelist[0]['stId'] if item.producelist else None
#         except Exception as e:
#             p2_5_driver_tr_log.warning(f"[{key}] STEP 2 - error while reading wsid/stid at idx={idx}: {repr(e)}")
#             continue
#
#         if f"{wsid_}_{stid_}" == key:
#             matching_indices.append(idx)
#
#     filtered_returnList = [returnList[i] for i in matching_indices]
#     filtered_outputList = [outputList[i] for i in matching_indices]
#
#     # p2_5_driver_tr_log.debug(
#     #     f"[{key}] STEP 2 - matching_indices: {matching_indices} "
#     #     f"(filtered_outputList: {filtered_outputList[0]})"
#     # )
#
#
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 2 DONE - recent window matched rows: {len(filtered_outputList)} "
#         f"(total fetched: {len(returnList)})"
#     )
#
#     # ---------------- FALLBACK window ----------------
#     if len(filtered_outputList) == 0:
#         try:
#             wsid_str, stid_str = key.split("_")
#             wsid = int(wsid_str)
#             stid = int(stid_str)
#
#             p2_5_driver_tr_log.warning(
#                 f"[{key}] STEP 2 - No rows in recent window. FALLBACK: fetch full history by key"
#             )
#
#             returnList_fb, _, outputList_fb, _ = dw_raw_data.fetchData_by_key(
#                 wsid, stid, limit=100000
#             )
#
#             filtered_returnList = returnList_fb
#             filtered_outputList = outputList_fb
#
#             p2_5_driver_tr_log.debug(
#                 f"[{key}] STEP 2 FALLBACK DONE - full history rows: {len(filtered_outputList)}"
#             )
#
#         except Exception as e:
#             p2_5_driver_tr_log.error(f"[{key}] STEP 2 FALLBACK FAILED: {repr(e)}")
#             return None
#
#     # ---------------- Step 3: NaN check ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 3 - Checking for NaN with has_nan_cntread")
#     for inst in filtered_returnList:
#         try:
#             if has_nan_cntread(inst):
#                 p2_5_driver_tr_log.warning(f"[{key}] STEP 3 - NaN found in row={inst}")
#         except Exception as e:
#             p2_5_driver_tr_log.warning(f"[{key}] STEP 3 - has_nan_cntread error: {repr(e)}")
#             continue
#
#     # ---------------- Step 4: enough samples ----------------
#     N = len(filtered_outputList)
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 4 - Validate Enough Training Data: N={N}")
#
#     if N < TIMESTEPS:
#         p2_5_driver_tr_log.debug(
#             f"[{key}] STEP 4 - Skipping training: need >= {TIMESTEPS}, got {N}"
#         )
#         return None
#
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 4 DONE - N={N} >= {TIMESTEPS}, continue training")
#
#     # ---------------- Step 5: Build DF ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 5 - Build Dataframe/extract features")
#     try:
#         X = build_training_dataframe_from_raw(filtered_returnList, filtered_outputList)
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 5 FAILED (feature extraction): {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 5 DONE - X shape={X.shape}, cols={list(X.columns)}")
#
#     # ---------------- Step 5b: type conversions ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 5b - Type conversions (sensor columns only)")
#     try:
#         # Identify sensor columns
#         sensor_cols = [c for c in X.columns if c.startswith("sensor_value_")]
#
#         p2_5_driver_tr_log.debug(f"[{key}] STEP 5b - Found {len(sensor_cols)} sensor columns")
#
#         # Convert ONLY sensor columns to numeric
#         for col in sensor_cols:
#             X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
#
#         # Log info about label columns (don't convert them!)
#         label_info = {
#             'good': X['good'].dtype if 'good' in X.columns else 'missing',
#             'prSt': X['prSt'].dtype if 'prSt' in X.columns else 'missing'
#         }
#         p2_5_driver_tr_log.debug(f"[{key}] STEP 5b - Label column types: {label_info}")
#         p2_5_driver_tr_log.debug(f"[{key}] STEP 5b - SENSOR LABELS: {sensor_cols}")
#
#         # Drop rows where ANY sensor column has NaN (but keep label columns intact)
#         before_drop = len(X)
#         p2_5_driver_tr_log.debug(f"[{key}] STEP 5b, X -  {X.head(n=5)}")
#         p2_5_driver_tr_log.debug(f"[{key}] STEP 5b, filtered_returnList -  {filtered_returnList[0]}")
#         p2_5_driver_tr_log.debug(f"[{key}] STEP 5b, filtered_outputList -  {filtered_outputList[0]}")
#
#
#         X.dropna(subset=sensor_cols, how='all', inplace=True)
#         after_drop = len(X)
#
#         p2_5_driver_tr_log.debug(
#             f"[{key}] STEP 5b DONE - dropna on sensors only: {before_drop} -> {after_drop} rows"
#         )
#
#         # Safety check
#         if len(X) < TIMESTEPS:
#             p2_5_driver_tr_log.warning(
#                 f"[{key}] STEP 5b - After dropna insufficient rows: {len(X)} < {TIMESTEPS}"
#             )
#             return None
#
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 5b FAILED (type conversion / dropna): {repr(e)}")
#         import traceback
#         p2_5_driver_tr_log.error(traceback.format_exc())
#         return None
#
#     # ========== NEW: STEP 5c - DERIVE AND FILTER BY QUALITY LABEL ==========
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 5c - Derive quality labels and filter")
#
#     try:
#         # Check if we have the required columns
#         if 'good' not in X.columns or 'productionstate' not in X.columns:
#             p2_5_driver_tr_log.warning(
#                 f"[{key}] STEP 5c - Missing 'good' or 'productionstate' columns. "
#                 f"Available columns: {list(X.columns)}"
#             )
#             p2_5_driver_tr_log.error(
#                 f"[{key}] STEP 5c - Cannot train without quality labels. Aborting."
#             )
#             return None
#
#         # Import the derive_goodcnt function from preprocessing layer
#         from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import derive_goodcnt
#
#         # Derive the quality label
#         # Note: parameter order is (production_state, good)
#         X['quality_label'] = derive_goodcnt(X['productionstate'].values, X['good'].values)
#
#         # Log label distribution BEFORE filtering
#         good_count = np.sum(X['quality_label'] == True)
#         bad_count = np.sum(X['quality_label'] == False)
#         unlabeled_count = np.sum(pd.isna(X['quality_label']) | (X['quality_label'] == None))
#
#         p2_5_driver_tr_log.info(
#             f"[{key}] STEP 5c - Label distribution BEFORE filter:\n"
#             f"  Good (True): {good_count}\n"
#             f"  Bad (False): {bad_count}\n"
#             f"  Unlabeled (None): {unlabeled_count}\n"
#             f"  Total rows: {len(X)}"
#         )
#
#         # ===== CRITICAL: Filter to ONLY good quality data for training =====
#         X_original_size = len(X)
#         X_train_only = X[X['quality_label'] == True].copy()
#
#         if len(X_train_only) == 0:
#             p2_5_driver_tr_log.warning(
#                 f"[{key}] STEP 5c - NO good quality data found! "
#                 f"All {X_original_size} rows are either bad or unlabeled. Cannot train."
#             )
#             return None
#
#         p2_5_driver_tr_log.info(
#             f"[{key}] STEP 5c - Filtered: {X_original_size} → {len(X_train_only)} rows "
#             f"({len(X_train_only)/X_original_size*100:.1f}% are good quality)"
#         )
#
#         # Check if we have enough good data
#         if len(X_train_only) < TIMESTEPS:
#             p2_5_driver_tr_log.warning(
#                 f"[{key}] STEP 5c - Insufficient good quality data: "
#                 f"{len(X_train_only)} < {TIMESTEPS}. Cannot train."
#             )
#             return None
#
#         # Keep: timestamp + sensor_value_* columns only
#         sensor_cols = [c for c in X_train_only.columns if c.startswith('sensor_value_')]
#         columns_to_keep = ['timestamp'] + sensor_cols
#
#         X_for_scaling = X_train_only[columns_to_keep].copy()
#
#         p2_5_driver_tr_log.debug(
#             f"[{key}] STEP 5c DONE - Columns for scaling: {list(X_for_scaling.columns)}"
#         )
#
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 5c FAILED (quality filtering): {repr(e)}")
#         import traceback
#         p2_5_driver_tr_log.error(traceback.format_exc())
#         return None
#
#     # ========== STEP 5d: Type conversions (now on filtered data) ==========
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 5d - Type conversions on filtered data")
#     try:
#         sensor_cols = [c for c in X_for_scaling.columns if c.startswith("sensor_value_")]
#
#         for col in sensor_cols:
#             X_for_scaling[col] = pd.to_numeric(X_for_scaling[col], errors="coerce").astype(float)
#
#         before_drop = len(X_for_scaling)
#         X_for_scaling.dropna(how='all', inplace=True)
#         after_drop = len(X_for_scaling)
#
#         p2_5_driver_tr_log.debug(
#             f"[{key}] STEP 5d DONE - dropna: {before_drop} → {after_drop} rows"
#         )
#
#         # Safety check after dropna
#         if len(X_for_scaling) < TIMESTEPS:
#             p2_5_driver_tr_log.warning(
#                 f"[{key}] STEP 5d - After dropna insufficient rows: "
#                 f"{len(X_for_scaling)} < {TIMESTEPS}"
#             )
#             return None
#
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 5d FAILED (type conversion): {repr(e)}")
#         return None
#
#     # ========== Continue with STEP 6: SCALING ==========
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 6 - Scaling with scale_and_save_training_data")
#     try:
#         X_scaled, scaler_path, min_max_path = scale_and_save_training_data(X_for_scaling, key)
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 6 FAILED in scale_and_save_training_data: {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 6 DONE - X_scaled shape={X_scaled.shape}"
#     )
#
#     # --- CRITICAL SAFETY CHECK ---
#     if len(X_scaled) < TIMESTEPS:
#         p2_5_driver_tr_log.debug(
#             f"[{key}] STEP 6 - After scaling insufficient rows: {len(X_scaled)} < {TIMESTEPS}"
#         )
#         return None
#
#     # ---------------- Step 7: Generate Sequences ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 7 - Sequence Generation timesteps={TIMESTEPS}")
#     try:
#         X_sequences, X_seq_timestamps = generate_sequences_with_timestamps(
#             X_scaled, timesteps=TIMESTEPS
#         )
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 7 FAILED (SeqGen): {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 7 DONE - X_sequences shape={X_sequences.shape}, "
#         f"train/test split incoming..."
#     )
#
#     # ---------------- Step 8: Train/test split ----------------
#     train_size = int(len(X_sequences) * 0.8)
#     X_sequences_train = X_sequences[:train_size]
#     X_sequences_test = X_sequences[train_size:]
#
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 8 - Train/test split: train={X_sequences_train.shape}, "
#         f"test={X_sequences_test.shape}"
#     )
#
#     # ---------------- Step 9: Training ----------------
#     model_dir = MODEL_DIR
#     existing_models = sorted(
#         f for f in os.listdir(model_dir)
#         if f.startswith(str(key)) and f.endswith(".keras")
#     )
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 9 - Existing models for key in {model_dir}: {existing_models}"
#     )
#
#     try:
#         if not existing_models:
#             p2_5_driver_tr_log.debug(f"[{key}] STEP 9 - No existing model, calling initial_train_and_save")
#             model_path = initial_train_and_save(
#                 X_sequences_train,
#                 X_sequences_train.shape[1],
#                 X_sequences_train.shape[2],
#                 key,
#             )
#         else:
#             model_path = os.path.join(model_dir, existing_models[-1])
#             p2_5_driver_tr_log.debug(f"[{key}] STEP 9 - Using existing model_path={model_path}")
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 9 FAILED (initial_train_and_save): {repr(e)}")
#         return None
#
#     # ---------------- Step 10: Predictions ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 10 - Predict with model_path={model_path}")
#     try:
#         predictions_test = predict_with_model(model_path, X_sequences_test)
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 10 FAILED (predict_with_model): {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 10 DONE - Predictions_test shape={predictions_test.shape}")
#
#     # ---------------- Step 11: Thresholds ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 11 - compute_reconstruction_error / threshold / anomaly")
#     try:
#         recon_error_test = compute_reconstruction_error(
#             X_sequences_test, predictions_test
#         )
#         threshold, threshold_path = compute_and_save_threshold(recon_error_test, key)
#         anomaly = detect_anomalies(recon_error_test, threshold)
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 11 FAILED (threshold/anomaly): {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 11 DONE - recon_error len={len(recon_error_test)}, "
#         f"threshold={threshold}, anomalies_sum={int(anomaly.sum())}"
#     )
#
#     # ---------------- Step 12: Last 20 sequences ----------------
#     p2_5_driver_tr_log.debug(f"[{key}] STEP 12 - extract last {TIMESTEPS} sequences for buffer")
#     try:
#         # Use X_for_scaling (the filtered good quality data) instead of X
#         last_20_seqs = X_for_scaling.tail(TIMESTEPS).drop(columns=['timestamp'], errors='ignore').to_numpy()
#     except Exception as e:
#         p2_5_driver_tr_log.error(f"[{key}] STEP 12 FAILED (last_20_seqs): {repr(e)}")
#         return None
#
#     p2_5_driver_tr_log.debug(
#         f"[{key}] STEP 12 DONE - last_20_seqs shape={last_20_seqs.shape}"
#     )
#     p2_5_driver_tr_log.debug(f"[{key}] ---- execute_training END (SUCCESS) ----")
#
#     return last_20_seqs


import os
import numpy as np
import pandas as pd

from cassandra_utils.models.dw_raw_data import dw_raw_data
from cassandra_utils.models.dw_single_data import dw_tbl_raw_data  # şu an kullanmıyoruz ama dursun
from thread.phase_2_multivariate_lstm_pipeline._2_2_pre_processing_layer import (
    generate_training_dataframe,
    generate_sequences_with_timestamps,
    has_nan_cntread,
    scale_and_save_training_data,
)
from thread.phase_2_multivariate_lstm_pipeline._2_3_processing_layer import (
    initial_train_and_save,
    check_model_exists,
    predict_with_model,
    MODEL_DIR,
)
from thread.phase_2_multivariate_lstm_pipeline._2_4_post_processing_layer import (
    compute_reconstruction_error,
    compute_and_save_threshold,
    detect_anomalies,
)
from utils.logger_2 import setup_logger

from pathlib import Path

# Bu dosyanın bulunduğu yerden base_dir hesaplayalım
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "phase2models"
THRESHOLD_DIR = MODEL_DIR / "threshold"
SCALER_DIR = MODEL_DIR / "scalers"
BUFFER_DIR = MODEL_DIR / "historical-buffer"

p2_5_driver_tr_log = setup_logger(
    "p2_5_driver_tr_log", "logs/p2_5_driver_tr.log"
)

# -------------------------- FETCH BASE DATA -------------------------- #

def preload_training_data(values_to_fetch=200):
    """
    Son X kaydı Cassandra'dan getirir (tüm ws + stok kombinasyonları için).
    """
    try:
        returnList, inputList, outputList, batchList = dw_raw_data.fetchData(values_to_fetch)
        if not returnList:
            p2_5_driver_tr_log.warning("2_1_1 - No data found in fetch call")
            p2_5_driver_tr_log.error("2_1_1 - No data received for training")
            raise Exception("No training data fetched")
        else:
            p2_5_driver_tr_log.debug("2_1_1 - Sucessfully preloading data")
            return returnList, inputList, outputList, batchList
    except Exception as e:
        p2_5_driver_tr_log.error(f"2_1_1 - Error preloading training data: {e}")

# -------------------------- BUILD TRAINING DATAFRAME -------------------------- #

def build_training_dataframe_from_raw(return_list, output_list):
    """
    dw_raw_data satırlarından ve outputvaluelist (outVals) listesinden
    LSTM'e uygun, wide-format bir DataFrame üretir.

    - Her satır: tek bir timestamp için tüm sensörler
    - Kolonlar: timestamp, plantid, workstationid, outputstockid, sensor_value_<eqId> ...
    """
    print("Starting dataframe generation...")
    p2_5_driver_tr_log.debug(
        f"build_training_dataframe_from_raw: got {len(return_list)} rows"
    )

    rows = []

    for idx, (row_obj, outvals) in enumerate(zip(return_list, output_list)):
        # Güvenlik: outvals boşsa atla
        if not outvals:
            continue

        # Timestamp:
        ts = getattr(row_obj, "measurement_date", None)
        if ts is None:
            # Yedek: outVals[0]["measDt"] ms → datetime
            try:
                ts = pd.to_datetime(outvals[0].get("measDt"), unit="ms", utc=True)
            except Exception:
                # timestamp bile çekemiyorsak bu kaydı at
                continue

        base = {
            "timestamp": ts,
            "plantid": getattr(row_obj, "plantid", None),
            "workstationid": getattr(row_obj, "workstationid", None),
            "outputstockid": getattr(row_obj, "outputstockid", None),
        }

        # Tek tek sensörleri sensor_value_<eqId> kolonlarına pivot et
        sensor_values = {}
        p2_5_driver_tr_log.debug(f" outvals count ********************** {len(outvals)}")
        p2_5_driver_tr_log.debug(f" outvals ********************** {outvals}")

        for ov in outvals:
            # ov dict değilse atla
            if not isinstance(ov, dict):
                continue

            eq_id = ov.get("eqId")
            cnt = ov.get("cntRead")

            if eq_id is None or cnt is None:
                continue

            col_name = f"sensor_value_{eq_id}"
            sensor_values[col_name] = cnt

        # Bu satırda hiç sensör yoksa DataFrame'e ekleme
        if not sensor_values:
            continue

        merged = {**base, **sensor_values}
        rows.append(merged)

    if not rows:
        raise ValueError(
            "No usable rows in build_training_dataframe_from_raw (rows list empty)."
        )

    X = pd.DataFrame(rows)

    p2_5_driver_tr_log.debug(f"X dataset: initial shape={X.shape}")
    p2_5_driver_tr_log.debug(f"X dataset columns: {list(X.columns)}")
    p2_5_driver_tr_log.debug(f"X dataset samples: {X.head(3)}")

    # timestamp'e göre sırala
    if "timestamp" in X.columns:
        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)

    print(f"Dataframe generated with shape {X.shape}")
    p2_5_driver_tr_log.debug(
        f"build_training_dataframe_from_raw: final shape={X.shape}, cols={list(X.columns)}"
    )
    return X

# -------------------------- MAIN TRAINING DRIVER -------------------------- #

TIMESTEPS = 20

def execute_training(key: str):
    p2_5_driver_tr_log.debug(f"[{key}] ---- execute_training START ----")

    # ---------------- Step 1: base fetch ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 1 - Historical Data fetch (preload_training_data)")
    try:
        returnList, inputList, outputList, batchList = preload_training_data(20000)
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 1 FAILED in preload_training_data: {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(f"[{key}] STEP 1 DONE - total rows fetched: {len(returnList)}")

    # ---------------- Step 2: filter by key (wsId_stId) ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 2 - filtering by key")
    matching_indices = []

    for idx, item in enumerate(returnList):
        try:
            wsid_ = item.workstationid
            stid_ = item.producelist[0]['stId'] if item.producelist else None
        except Exception as e:
            p2_5_driver_tr_log.warning(f"[{key}] STEP 2 - error while reading wsid/stid at idx={idx}: {repr(e)}")
            continue

        if f"{wsid_}_{stid_}" == key:
            matching_indices.append(idx)

    filtered_returnList = [returnList[i] for i in matching_indices]
    filtered_outputList = [outputList[i] for i in matching_indices]

    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 2 DONE - recent window matched rows: {len(filtered_outputList)} "
        f"(total fetched: {len(returnList)})"
    )

    # ---------------- FALLBACK window ----------------
    if len(filtered_outputList) == 0:
        try:
            wsid_str, stid_str = key.split("_")
            wsid = int(wsid_str)
            stid = int(stid_str)

            p2_5_driver_tr_log.warning(
                f"[{key}] STEP 2 - No rows in recent window. FALLBACK: fetch full history by key"
            )

            returnList_fb, _, outputList_fb, _ = dw_raw_data.fetchData_by_key(
                wsid, stid, limit=100000
            )

            filtered_returnList = returnList_fb
            filtered_outputList = outputList_fb

            p2_5_driver_tr_log.debug(
                f"[{key}] STEP 2 FALLBACK DONE - full history rows: {len(filtered_outputList)}"
            )

        except Exception as e:
            p2_5_driver_tr_log.error(f"[{key}] STEP 2 FALLBACK FAILED: {repr(e)}")
            return None

    # ---------------- Step 3: NaN check ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 3 - Checking for NaN with has_nan_cntread")
    for inst in filtered_returnList:
        try:
            if has_nan_cntread(inst):
                p2_5_driver_tr_log.warning(f"[{key}] STEP 3 - NaN found in row={inst}")
        except Exception as e:
            p2_5_driver_tr_log.warning(f"[{key}] STEP 3 - has_nan_cntread error: {repr(e)}")
            continue

    # ---------------- Step 4: enough samples ----------------
    N = len(filtered_outputList)
    p2_5_driver_tr_log.debug(f"[{key}] STEP 4 - Validate Enough Training Data: N={N}")

    if N < TIMESTEPS:
        p2_5_driver_tr_log.debug(
            f"[{key}] STEP 4 - Skipping training: need >= {TIMESTEPS}, got {N}"
        )
        return None

    p2_5_driver_tr_log.debug(f"[{key}] STEP 4 DONE - N={N} >= {TIMESTEPS}, continue training")

    # ---------------- Step 5: Build DF ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 5 - Build Dataframe/extract features")
    try:
        X = build_training_dataframe_from_raw(filtered_returnList, filtered_outputList)
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 5 FAILED (feature extraction): {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(f"[{key}] STEP 5 DONE - X shape={X.shape}, cols={list(X.columns)}")

    # ---------------- Step 5b: type conversions ----------------
    try:
        sensor_cols = [c for c in X.columns if c.startswith("sensor_value_")]
        p2_5_driver_tr_log.debug(f"[{key}] STEP 5b - sensor_cols={sensor_cols}")

        for col in sensor_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        other_cols = [c for c in X.columns if c not in sensor_cols and c != "timestamp"]
        p2_5_driver_tr_log.debug(f"[{key}] STEP 5b - other_cols={other_cols}")

        for col in other_cols:
            X[col] = X[col].astype(float)

        before_drop = len(X)
        X.dropna(inplace=True)
        after_drop = len(X)

        p2_5_driver_tr_log.debug(
            f"[{key}] STEP 5b DONE - dropna: {before_drop} -> {after_drop} rows"
        )
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 5b FAILED (type conversion / dropna): {repr(e)}")
        return None

    # ---------------- Step 6: SCALING ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 6 - Scaling with scale_and_save_training_data")
    try:
        X_scaled, scaler_path, min_max_path = scale_and_save_training_data(X, key)
        p2_5_driver_tr_log.debug(f"SCALED X -> {X_scaled.head}")
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 6 FAILED in scale_and_save_training_data: {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 6 DONE - X_scaled shape={X_scaled.shape}, "
        f"scaler_path={scaler_path}, min_max_path={min_max_path}"
    )

    # --- CRITICAL SAFETY CHECK ---
    if len(X_scaled) < TIMESTEPS:
        p2_5_driver_tr_log.debug(
            f"[{key}] STEP 6 - After scaling insufficient rows: {len(X_scaled)} < {TIMESTEPS}"
        )
        return None

    # ---------------- Step 7: Generate Sequences ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 7 - Sequence Generation timesteps={TIMESTEPS}")
    try:
        X_sequences, X_seq_timestamps = generate_sequences_with_timestamps(
            X_scaled, timesteps=TIMESTEPS
        )
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 7 FAILED (SeqGen): {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 7 DONE - X_sequences shape={X_sequences.shape}, "
        f"train/test split incoming..."
    )

    # ---------------- Step 8: Train/test split ----------------
    train_size = int(len(X_sequences) * 0.8)
    X_sequences_train = X_sequences[:train_size]
    X_sequences_test = X_sequences[train_size:]

    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 8 - Train/test split: train={X_sequences_train.shape}, "
        f"test={X_sequences_test.shape}"
    )

    # ---------------- Step 9: Training ----------------
    model_dir = MODEL_DIR
    existing_models = sorted(
        f for f in os.listdir(model_dir)
        if f.startswith(str(key)) and f.endswith(".keras")
    )
    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 9 - Existing models for key in {model_dir}: {existing_models}"
    )

    try:
        if not existing_models:
            p2_5_driver_tr_log.debug(f"[{key}] STEP 9 - No existing model, calling initial_train_and_save")
            model_path = initial_train_and_save(
                X_sequences_train,
                X_sequences_train.shape[1],
                X_sequences_train.shape[2],
                key,
            )
        else:
            model_path = os.path.join(model_dir, existing_models[-1])
            p2_5_driver_tr_log.debug(f"[{key}] STEP 9 - Using existing model_path={model_path}")
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 9 FAILED (initial_train_and_save): {repr(e)}")
        return None

    # ---------------- Step 10: Predictions ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 10 - Predict with model_path={model_path}")
    try:
        predictions_test = predict_with_model(model_path, X_sequences_test)
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 10 FAILED (predict_with_model): {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(f"[{key}] STEP 10 DONE - Predictions_test shape={predictions_test.shape}")

    # ---------------- Step 11: Thresholds ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 11 - compute_reconstruction_error / threshold / anomaly")
    try:
        recon_error_test = compute_reconstruction_error(
            X_sequences_test, predictions_test
        )
        threshold, threshold_path = compute_and_save_threshold(recon_error_test, key)
        anomaly = detect_anomalies(recon_error_test, threshold)
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 11 FAILED (threshold/anomaly): {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 11 DONE - recon_error len={len(recon_error_test)}, "
        f"threshold={threshold}, anomalies_sum={int(anomaly.sum())}"
    )

    # ---------------- Step 12: Last 20 sequences ----------------
    p2_5_driver_tr_log.debug(f"[{key}] STEP 12 - extract last {TIMESTEPS} sequences for buffer")
    try:
        last_20_seqs = X.tail(TIMESTEPS).iloc[:, 1:].to_numpy()
    except Exception as e:
        p2_5_driver_tr_log.error(f"[{key}] STEP 12 FAILED (last_20_seqs): {repr(e)}")
        return None

    p2_5_driver_tr_log.debug(
        f"[{key}] STEP 12 DONE - last_20_seqs shape={last_20_seqs.shape}"
    )
    p2_5_driver_tr_log.debug(f"[{key}] ---- execute_training END (SUCCESS) ----")

    return last_20_seqs