# Data transformation prior to feeding to model
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler
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

#Loggers
p2_2_inf_log = setup_logger(
    "p2_2_preprocessing_layer_inference_logger", "logs/p2_2_preprocessing_layer_inference.log"
)

p2_2_train_log = setup_logger(
    "p2_2_preprocessing_layer_training_logger", "logs/p2_2_preprocessing_layer_training.log"
)

state_mapping = {
    "Closed": 0,
    "StandBy": 1,
    "Running": 2,
    "Setup": 3,
    "Stopped": 5,
    "Manipulation": 6,
    "Malfunction": 7,
    "Loading": 8
}

def get_state_code(state_str):
    return state_mapping.get(state_str, -1)

def derive_goodcnt(production_state, good):
    """
    Derive quality label from production state and good flag.
    
    Logic:
    - If productionstate == 'PRODUCTION' AND good == True  → True (good quality)
    - If productionstate == 'PRODUCTION' AND good == False → False (bad quality)
    - If productionstate != 'PRODUCTION' OR good is NULL   → None (unlabeled)
    
    Args:
        production_state: Series or array of production state values ("PRODUCTION", "STOPPED", etc.)
        good: Series or array of good flag values (True, False, None/NULL)
        
    Returns:
        np.ndarray: Quality labels (True=good, False=bad, None=unlabeled)
    """
    p = pd.Series(production_state, dtype="object")
    g = pd.Series(good, dtype="object")

    # Production mask - check if state is "PRODUCTION"
    is_prod = (p.astype(str).str.upper().eq("PRODUCTION") & p.notna())
    is_prod = is_prod.fillna(False).to_numpy(dtype=bool)

    # Good/bad masks
    # In your table: TRUE means good, NULL means unknown
    # We need to handle: True (explicit good), False (explicit bad), None/NULL (unknown)
    g_is_true = (g == True).fillna(False).to_numpy(dtype=bool)
    g_is_false = (g == False).fillna(False).to_numpy(dtype=bool)
    # Note: g == None or g.isna() will be neither true nor false

    # Allocate output
    y = np.empty(len(p), dtype=object)
    y[:] = None
    
    # Only label data that is in PRODUCTION state
    y[is_prod & g_is_true] = True   # PRODUCTION + good=True → Good quality
    y[is_prod & g_is_false] = False # PRODUCTION + good=False → Bad quality
    # Everything else stays None (unlabeled)
    
    # Log statistics
    total = len(y)
    good_count = np.sum(y == True)
    bad_count = np.sum(y == False)
    unlabeled_count = np.sum(pd.isna(y) | (y == None))
    
    p2_2_train_log.info(
        f"derive_goodcnt: Total={total}, "
        f"Good={good_count} ({good_count/total*100:.1f}%), "
        f"Bad={bad_count} ({bad_count/total*100:.1f}%), "
        f"Unlabeled={unlabeled_count} ({unlabeled_count/total*100:.1f}%)"
    )
    
    return y

def generate_training_dataframe(batchList, returnList, outputList):
    print("Starting dataframe generation...")
    all_rows = []
    all_columns = []

    for i in range(len(batchList)):

        wsid = returnList[i].workstationid
        mcst = get_state_code(returnList[i].machinestate)
        gd = 1 if returnList[i].good else 0
        cng = returnList[i].quantitychanged

        eq_id = []
        cnt_read = []
        timestamp = outputList[i][0].get("measDt", "0")

        for sensor in outputList[i]:
            eq_id.append(sensor.get('eqId'))
            cnt_read.append(sensor.get('cntRead'))


        stid = batchList[i][0]['stId'] if batchList[i] and any(batchList[i]) else 0

        row_data = [timestamp, wsid, mcst, gd, cng, stid] + eq_id + cnt_read

        if not all_columns:
            columns = (
                ['timestamp', 'workstationid', 'machinestate', 'good', 'quantitychanged', 'stId'] +
                [f'eqId_{j+1}' for j in range(len(eq_id))] +
                [f'sensor_value_{j+1}' for j in range(len(cnt_read))]
            )
            all_columns = columns

        all_rows.append(row_data)

    df = pd.DataFrame(all_rows, columns=all_columns)
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms', errors='coerce')

    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    p2_2_train_log.debug(f"Final DataFrame ready for training \n {df.head()}")
    return df

def generate_sequences_with_timestamps(scaled_df, timesteps=20):

    if 'timestamp' not in scaled_df.columns:
        p2_2_train_log.error("'timestamp' column not found in input DataFrame.")        
        raise ValueError("'timestamp' column not found in input DataFrame.")

    if len(scaled_df) < timesteps:
        p2_2_train_log.error(f" Not enough rows to generate sequences. Need at least {timesteps}, got {len(scaled_df)}.")
        raise ValueError(f" Not enough rows to generate sequences. Need at least {timesteps}, got {len(scaled_df)}.")

    sensor_cols = scaled_df.columns.difference(['timestamp'])
    values = scaled_df[sensor_cols].values
    timestamps = scaled_df['timestamp'].reset_index(drop=True)

    sequences = np.array([
        values[i:i+timesteps] for i in range(len(values) - timesteps + 1)
    ])
    
    # Align timestamps to the last row in each sequence
    seq_timestamps = timestamps[timesteps - 1:].reset_index(drop=True)

    return sequences, seq_timestamps

def has_nan_cntread(instance):
    output_list = getattr(instance, "outputValueList", [])
    for item in output_list:
        try:
            val = float(item.get("cntRead", 0))
            if math.isnan(val):
                return True
        except (ValueError, TypeError):
            return True  # In case cntRead is non-convertible
    return False

def scale_and_save_training_data(df, key, save_dir='models/phase2models/scalers'):
    """
    Scale ONLY sensor columns (timestamp + sensor_value_* columns expected)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Remove any previous scaler/min-max files
    for f in os.listdir(save_dir):
        if f.startswith(key) and (f.endswith('.pkl') or f.endswith('.npz')):
            os.remove(os.path.join(save_dir, f))

    # Step 2: Convert timestamp column to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    if df.empty:
        p2_2_train_log.error("DataFrame is empty after timestamp parsing!")
        raise ValueError("DataFrame is empty after timestamp parsing!")

    # Step 3: Extract and scale sensor data
    # Remove timestamp + any label columns that might have leaked through
    label_cols = ['quality_label', 'good', 'prSt', 'plantid', 'workstationid', 'outputstockid']
    sensor_cols = [c for c in df.columns if c not in label_cols + ['timestamp']]
    sensor_count = len(sensor_cols)


    p2_2_inf_log.debug(f"SENSOR COLS FOR SCALING -> {sensor_cols}")

    sensor_df = df[sensor_cols]

    if sensor_df.empty or len(sensor_df) < 1:
        p2_2_train_log.error("No valid sensor columns found for scaling!")
        raise ValueError("No valid sensor columns found for scaling!")

    p2_2_train_log.debug(f"Scaling {len(sensor_cols)} sensor columns: {sensor_cols}")

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(sensor_df)

    # Step 4: Construct final DataFrame with timestamp as first column
    scaled_df = pd.DataFrame(scaled_array, columns=sensor_cols)
    scaled_df.insert(0, 'timestamp', df['timestamp'].values)

    # Step 5: Save scaler and min/max
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scaler_path = os.path.join(save_dir, f'{key}_{timestamp}_standard_scaler.pkl')
    joblib.dump(scaler, scaler_path)

    min_max_path = os.path.join(save_dir, f'{key}_{timestamp}_min_max.npz')
    np.savez(min_max_path, train_min=sensor_df.min().to_numpy(), train_max=sensor_df.max().to_numpy())

    p2_2_train_log.debug(f"Scaler saved to: {scaler_path}")
    p2_2_train_log.debug(f"Min/Max saved to: {min_max_path}")

    return scaled_df, scaler_path, min_max_path

#Inference Modules

def generate_testing_dataframe(instance):
    """
    Tek bir dw_raw_data satırını (dw_raw_data.saveData(...)'in döndürdüğü instance),
    trainingte kullandığımız şemayla AYNI forma çevirir.

    Training X kolonları:
      ['timestamp',
       'plantid',
       'workstationid',
       'outputstockid',
       'sensor_value_6944',
       'sensor_value_6945',
       'sensor_value_6946',
       'sensor_value_6947']
    """

    print("Starting testing dataframe generation...")

    # 1) outVals: dw_raw_data modelindeki field adı -> outputvaluelist
    outvals = getattr(instance, "outputvaluelist", None) or []
    if not outvals:
        raise ValueError("generate_testing_dataframe: outputvaluelist is empty")

    # 2) timestamp:
    #    Önce measurement_date alanını dene (train tarafında da bu kullanılıyor)
    ts = getattr(instance, "measurement_date", None)

    #    Eğer measurement_date yoksa/falsy ise ilk outVal içindeki measDt'den üret
    if ts is None:
        try:
            meas_ms = outvals[0].get("measDt")
            ts = pd.to_datetime(meas_ms, unit="ms", utc=True)
        except Exception:
            ts = pd.NaT

    base = {
        "timestamp": ts,
        "plantid": getattr(instance, "plantid", None),
        "workstationid": getattr(instance, "workstationid", None),
        "outputstockid": getattr(instance, "outputstockid", None),
    }

    # 3) sensör değerleri → sensor_value_<eqId>
    sensor_values = {}
    for ov in outvals:
        # Cassandra list<map<text,text>> yapısı dict gibi geliyor olmalı
        if not isinstance(ov, dict):
            continue

        eq_id = ov.get("eqId")
        cnt = ov.get("cntRead")
        if eq_id is None or cnt is None:
            continue

        col_name = f"sensor_value_{eq_id}"
        sensor_values[col_name] = cnt

    if not sensor_values:
        raise ValueError("generate_testing_dataframe: no sensor values built from outputvaluelist")

    # 4) tek satırlık DataFrame oluştur
    row = {**base, **sensor_values}
    p2_2_inf_log.debug(f"final X row -> {row}")
    df = pd.DataFrame([row])

    # 5) timestamp'i datetime'a çevir, NaT olan satırı at
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # 6) Kolon sırasını TRAIN ile aynı hale getir:
    #    train tarafında scale_and_save_training_data içinde:
    #    sensor_cols = df.columns.difference(['timestamp'])
    #    ile alfabetik sıralı kullanıyorsun.
    other_cols = sorted([c for c in df.columns if c != "timestamp"])
    df = df[["timestamp"] + other_cols]

    p2_2_inf_log.debug(
        f"[generate_testing_dataframe] cols={df.columns.tolist()}, "
        f"row={df.iloc[0].to_dict()}"
    )

    return df

def generate_testing_dataframe2(data):
    print("Starting testing dataframe generation...")
    all_rows = []
    all_columns = []


    wsid = data['workstationid']
    mcst = get_state_code(data['machinestate'])
    gd = 1 if data['good'] else 0
    cng = data['quantitychanged']

    eq_id = []
    cnt_read = []
    output_values = data['outputvaluelist']
    timestamp = output_values[0]['measDt'] if output_values else 0

    for sensor in output_values:
        eq_id.append(sensor['eqId'])
        cnt_read.append(sensor['cntRead'])

    produce_list = data['producelist']
    stid = produce_list[0]['stId'] if produce_list else 0

    row_data = [timestamp, wsid, mcst, gd, cng, stid] + eq_id + cnt_read

    if not all_columns:
        columns = (
                ['timestamp', 'workstationid', 'machinestate', 'good', 'quantitychanged', 'stId'] +
                [f'eqId_{j+1}' for j in range(len(eq_id))] +
                [f'sensor_value_{j+1}' for j in range(len(cnt_read))]
            )
        all_columns = columns

        # Ensure row matches column length
    if len(row_data) < len(all_columns):
        row_data += [''] * (len(all_columns) - len(row_data))
    elif len(row_data) > len(all_columns):
        row_data = row_data[:len(all_columns)]

    all_rows.append(row_data)

    df = pd.DataFrame(all_rows, columns=all_columns)

    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms', errors='coerce')

    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    p2_2_train_log.debug(f"Final DataFrame ready for training \n {df.head()}")
    return df

def type_casting_df(X):
    sensor_cols = [col for col in X.columns if col.startswith('sensor_value_')]

    for col in sensor_cols:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].astype(float)  # force type
        except Exception as e:
            print(f"[ERROR] Failed to convert {col} to float: {e}")


        # Convert all other columns (except timestamp and sensor values) to int
    cols_to_int = [col for col in X.columns if col not in sensor_cols and col != 'timestamp']
    for col in cols_to_int:
        try:
            X[col] = X[col].astype(int)
        except Exception as e:
            print(f"[ERROR] Failed to convert {col} to int: {e}")
    return X

def scale_live_data_from_dir(X, key, scaler_dir='models/phase2models/scalers'):
    p2_2_inf_log.debug(f" Looking for scaler and min/max in directory: {scaler_dir}")



    # Step 1: Find files
    files = os.listdir(scaler_dir)
    sensor_cols = [c for c in X.columns if c.startswith("sensor_value_")]


    scaler_file = [f for f in files if f.startswith(str(key)) and f.endswith('_standard_scaler.pkl')]
    minmax_file = [f for f in files if f.startswith(str(key)) and f.endswith('_min_max.npz')]

    if not scaler_file or not minmax_file:
        p2_2_inf_log.error("Scaler or min/max file not found in directory.")
        raise FileNotFoundError("Scaler or min/max file not found in directory.")

    scaler_path = os.path.join(scaler_dir, scaler_file[0])
    minmax_path = os.path.join(scaler_dir, minmax_file[0])

    p2_2_inf_log.debug(f" Loading scaler from: {scaler_path}")
    p2_2_inf_log.debug(f" Loading min/max from: {minmax_path}")

    # Step 2: Load
    scaler = joblib.load(scaler_path)
    minmax = np.load(minmax_path)
    train_min = minmax['train_min']
    train_max = minmax['train_max']

    if hasattr(scaler, 'feature_names_in_'):
        p2_2_inf_log.debug(f"Feature names: {scaler.feature_names_in_}")

    # Step 3: Prepare input
    # timestamp kolonunu at, kalan kolon sayısı train_min/train_max ile aynı olmalı
    #X_numeric = X.drop(columns=['timestamp', ], errors='ignore')
    columns_to_drop = [col for col in X.columns if 'sensor_value_' not in col]
    X_numeric = X.drop(columns=columns_to_drop)

    X_numeric = X_numeric.fillna(0)

    p2_2_inf_log.debug(f"X_numeric -> {X_numeric.head()}")
    p2_2_inf_log.debug(f"train_min -> {train_min}")

    if X_numeric.shape[1] != len(train_min):
        p2_2_inf_log.error(
            f"Feature count mismatch: X has {X_numeric.shape[1]} features, "
            f"but scaler expects {len(train_min)}."
        )
        return None
        #raise ValueError("Feature count mismatch between live data and scaler.")

    p2_2_inf_log.debug(f" Input shape: {X_numeric.shape} | Expected shape: (?, {len(train_min)})")

    try:
        X_clipped = np.clip(X_numeric.values, train_min, train_max)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            X_scaled = scaler.transform(X_clipped)

        return X_scaled
    except Exception as e:
        p2_2_inf_log.error(f" Error during scaling: {e}")
        return None

def scale_live_data_from_dir_old(X, key ,scaler_dir='models/phase2models/scalers'):
    p2_2_inf_log.debug(f" Looking for scaler and min/max in directory: {scaler_dir}")

    # Step 1: Find files
    files = os.listdir(scaler_dir)
    scaler_file = [f for f in files if f.startswith(key) and f.endswith('_standard_scaler.pkl')]
    minmax_file = [f for f in files if f.startswith(key) and f.endswith('_min_max.npz')]

    if not scaler_file or not minmax_file:
        p2_2_inf_log.error("Scaler or min/max file not found in directory.")
        raise FileNotFoundError("Scaler or min/max file not found in directory.")

    scaler_path = os.path.join(scaler_dir, scaler_file[0])
    minmax_path = os.path.join(scaler_dir, minmax_file[0])

    p2_2_inf_log.debug(f" Loading scaler from: {scaler_path}")
    p2_2_inf_log.debug(f" Loading min/max from: {minmax_path}")

    # Step 2: Load
    scaler = joblib.load(scaler_path)
    minmax = np.load(minmax_path)
    train_min = minmax['train_min']
    train_max = minmax['train_max']

    # Step 3: Prepare input
#    X = X.iloc[:, 1:1+len(train_min)]   # Skip timestamp, match feature count
    X = X.fillna(0)

    p2_2_inf_log.debug(f" Input shape: {X.shape} | Expected shape: (?, {len(train_min)})")

    try:
        X_clipped = np.clip(X.values, train_min, train_max)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            X_scaled = scaler.transform(X_clipped)

        return X_scaled
    except Exception as e:
        p2_2_inf_log.error(f" Error during scaling: {e}")
        return None

def generate_sequences_only(X_scaled, timesteps=20):
    """
    Generate overlapping sequences from scaled sensor data.
    
    Args:
        X_scaled (np.ndarray): 2D array of shape (n_samples, n_features)
        timesteps (int): Number of timesteps per sequence

    Returns:
        np.ndarray: 3D array of shape (n_sequences, timesteps, n_features)
    """
    if X_scaled.shape[0] < timesteps:
        p2_2_inf_log.error(f" Not enough rows to generate sequences. Need at least {timesteps}, got {X_scaled.shape[0]}.")
        raise ValueError(f" Not enough rows to generate sequences. Need at least {timesteps}, got {X_scaled.shape[0]}.")

    sequences = np.array([
        X_scaled[i:i + timesteps]
        for i in range(X_scaled.shape[0] - timesteps + 1)
    ])
    return sequences
