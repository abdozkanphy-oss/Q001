import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils.logger_2 import setup_logger

# ================== PATH SETUP ==================
# Bu dosyanın bulunduğu yerden base_dir hesaplayalım
# thread/phase_2_multivariate_lstm_pipeline/_2_4_post_processing_layer.py
# buradan 2 seviye yukarı çıkınca pm-phase2 kökü:
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "phase2models"
THRESHOLD_DIR = MODEL_DIR / "threshold"
SCALER_DIR = MODEL_DIR / "scalers"
BUFFER_DIR = MODEL_DIR / "historical-buffer"

# ================== LOGGERS ==================
p2_4_inf_log = setup_logger(
    "p2_4_postprocessing_layer_inference_logger",
    "logs/p2_4_postprocessing_layer_inference.log"
)

p2_4_train_log = setup_logger(
    "p2_4_postprocessing_layer_training_logger",
    "logs/p2_4_postprocessing_layer_training.log"
)

# ================== THRESHOLD MULTIPLIERS ==================
# base_threshold: eğitim setinden gelen percentile
# soft_threshold = base * SOFT_MULT
# hard_threshold = base * HARD_MULT  → gerçek anomaly kararı buradan
SOFT_MULT = 1.5
HARD_MULT = 2.2   # aralığı geniş tutmak için önceki projeye göre biraz yukarı


# ===========================================================
#                       COMMON UTILS
# ===========================================================
def _ensure_dir(path: Path):
    """Path nesnesini stringe çevirerek dizini garanti et."""
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path


def _load_latest_threshold_npz(key: str, threshold_dir: Path = THRESHOLD_DIR):
    """
    Verilen key için threshold npz dosyasını (en son kaydedileni) yükler.
    Dönen: (threshold: float, npz_full_path: str, npz_loaded_data)
    """
    threshold_dir = Path(threshold_dir)
    _ensure_dir(threshold_dir)

    files = sorted(
        [f for f in os.listdir(threshold_dir) if f.startswith(key) and f.endswith(".npz")],
        reverse=True,
    )
    if not files:
        msg = f"No threshold file found for key={key} in {threshold_dir}"
        p2_4_inf_log.error(msg)
        raise FileNotFoundError(msg)

    latest = files[0]
    full_path = threshold_dir / latest
    data = np.load(full_path)

    if "threshold" not in data:
        msg = f"threshold key missing in npz file: {full_path}"
        p2_4_inf_log.error(msg)
        raise KeyError(msg)

    base_threshold = float(data["threshold"])
    return base_threshold, str(full_path), data


# ===========================================================
#                      RECON ERROR
# ===========================================================
def compute_reconstruction_error(y_true, y_pred):
    """
    Compute mean absolute reconstruction error across timesteps and features.

    y_true, y_pred: shape (n_seq, timesteps, n_features)
    Returns:
      error: shape (n_seq,)  → her sequence için tek bir skor
    """
    error = np.mean(np.abs(y_true - y_pred), axis=(1, 2))
    return error


# ===========================================================
#                 TRAINING-SIDE THRESHOLD LOGIC
# ===========================================================
def compute_and_save_threshold(
    recon_error,
    key: str,
    percentile: int = 95,
    save_dir: Path = THRESHOLD_DIR,
):
    """
    Eğitim setindeki reconstruction error'lardan threshold üretir
    ve aynı key'e ait eski threshold dosyalarını silip yenisini kaydeder.
    """
    p2_4_train_log.debug("=== Starting compute_and_save_threshold ===")

    # 1) Numpy array'e çevir
    p2_4_train_log.debug(f"Input type: {type(recon_error)}")
    recon_error = np.asarray(recon_error)
    p2_4_train_log.debug(f"Array shape: {recon_error.shape}")

    # 2) Şekil ve doluluk kontrolü
    if recon_error.ndim != 1:
        p2_4_train_log.error(f"Expected 1D array, got shape {recon_error.shape}")
        raise ValueError(f"Expected 1D array for recon_error, got shape: {recon_error.shape}")

    if recon_error.size == 0:
        p2_4_train_log.error("recon_error array is empty.")
        raise ValueError("recon_error is empty.")

    if np.isnan(recon_error).any():
        p2_4_train_log.error("recon_error contains NaN values.")
        raise ValueError("recon_error contains NaN values.")

    if np.isinf(recon_error).any():
        p2_4_train_log.error("recon_error contains infinite values.")
        raise ValueError("recon_error contains infinite values.")

    p2_4_train_log.debug(f"Input reconstruction error array size: {len(recon_error)}")
    p2_4_train_log.debug(f"Using percentile: {percentile}")

    # 3) Kayıt dizinini hazırla
    save_dir = _ensure_dir(save_dir)
    p2_4_train_log.debug(f"Ensuring save directory exists: {save_dir}")

    # 4) Aynı key için eski threshold dosyalarını sil
    try:
        p2_4_train_log.debug("Clearing old threshold files in the directory.")
        for f in os.listdir(save_dir):
            if f.startswith(key) and f.endswith(".npz"):
                old_path = save_dir / f
                p2_4_train_log.debug(f"Deleting old file: {old_path}")
                os.remove(old_path)
        p2_4_train_log.debug("Old threshold files cleared successfully.")
    except Exception as e:
        p2_4_train_log.error(f"Error clearing old threshold files: {e}")
        raise

    # 5) Yeni threshold hesapla
    try:
        p2_4_train_log.debug("Computing threshold using np.percentile.")
        threshold = float(np.percentile(recon_error, percentile))
        p2_4_train_log.debug(f"Threshold computed successfully: {threshold:.6f}")
    except Exception as e:
        p2_4_train_log.error(f"Failed to compute threshold: {e}")
        raise

    # 6) Kaydet
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        threshold_path = save_dir / f"{key}_{timestamp}.npz"
        p2_4_train_log.debug(f"Saving threshold and recon_error to: {threshold_path}")
        np.savez(threshold_path, threshold=threshold, recon_error=recon_error)
        p2_4_train_log.debug("Threshold file saved successfully.")
    except Exception as e:
        p2_4_train_log.error(f"Error saving threshold file: {e}")
        raise

    p2_4_train_log.debug(f"Percentile used: {percentile}")
    p2_4_train_log.debug(f"Final threshold value: {threshold:.6f}")
    p2_4_train_log.debug(f"Threshold file path: {threshold_path}")
    p2_4_train_log.debug("=== compute_and_save_threshold complete ===")

    return threshold, str(threshold_path)


def detect_anomalies(recon_error, threshold):
    """
    Eğitim tarafı için klasik anomaly tespiti:
    recon_error > threshold
    """
    recon_error = np.asarray(recon_error, dtype=float)
    anomalies = recon_error > float(threshold)
    p2_4_train_log.debug(
        f"Detected {np.sum(anomalies)} anomalies out of {len(recon_error)} samples."
    )
    return anomalies


# ===========================================================
#                  INFERENCE-SIDE ANOMALY LOGIC
# ===========================================================
def live_anomaly_detection(recon_error, key: str, threshold_dir: Path = THRESHOLD_DIR):
    """
    Canlı inference için anomaly kararı.

    - threshold npz içinden base_threshold okunur
    - soft_threshold = base * SOFT_MULT
    - hard_threshold = base * HARD_MULT
    - yalnızca recon_error >= hard_threshold ise True (anomaly)
    """
    base_threshold, path, _ = _load_latest_threshold_npz(key, threshold_dir)
    soft_thr = base_threshold * SOFT_MULT
    hard_thr = base_threshold * HARD_MULT

    err = np.asarray(recon_error, dtype=float)

    flags = err >= hard_thr

    p2_4_inf_log.debug(
        f"[live_anomaly_detection] key={key}, path={path}, "
        f"base={base_threshold:.6f}, soft={soft_thr:.6f}, hard={hard_thr:.6f}"
    )
    p2_4_inf_log.debug(f"Recon errors (last 5): {err[-5:]}")
    p2_4_inf_log.debug(f"Anomaly flags (last 5): {flags[-5:]}")

    return flags


def compute_anomaly_importance(
    recon_error,
    key: str,
    threshold_dir: Path = THRESHOLD_DIR,
):
    """
    Anomaly importance skoru hesapla.

    Mantık:
      - base_threshold eğitimden
      - soft_thr = base * SOFT_MULT
      - hard_thr = base * HARD_MULT
      - importance ~ 0  → soft_thr civarı (normal / borderline)
      - importance ~ 1  → hard_thr civarı (kesin anomaly eşiği)
      - importance > 1  → çok daha kritik anomaly
    """
    base_threshold, path, _ = _load_latest_threshold_npz(key, threshold_dir)

    err = np.asarray(recon_error, dtype=float)

    soft_thr = base_threshold * SOFT_MULT
    hard_thr = base_threshold * HARD_MULT

    # denom: soft ile hard arası mesafe
    denom = hard_thr - soft_thr
    if denom <= 0:
        # Güvenlik amaçlı, çok uç bir durumda fallback
        denom = max(base_threshold, 1e-6)

    # normalize: soft_thr → 0, hard_thr → 1
    raw_imp = (err - soft_thr) / denom
    importance = np.clip(raw_imp, 0.0, None)

    p2_4_inf_log.debug(
        f"[compute_anomaly_importance] key={key}, path={path}, "
        f"base={base_threshold:.6f}, soft={soft_thr:.6f}, hard={hard_thr:.6f}"
    )
    p2_4_inf_log.debug(f"Recon error (last 5): {err[-5:]}")
    p2_4_inf_log.debug(f"Importance  (last 5): {importance[-5:]}")

    return importance, base_threshold


# ===========================================================
#                SENSORWISE ERROR (HEATMAP SCORE)
# ===========================================================
def sensorwise_error_score(X_actual, X_pred):
    """
    Sensor bazlı hata hesabı (heatmap için).

    Parametreler:
      X_actual : np.ndarray, shape (n_seq, timesteps, n_sensors)
                  → buraya genelde X_sequences (scaled) geliyor
      X_pred   : np.ndarray, shape (n_seq, timesteps, n_sensors)
                  → model çıktısı

    Dönüş:
      error : 1D array, shape (n_sensors,)
              → son sequence'in son timestep'i için her sensorun mutlak hatası
    """
    X_actual = np.asarray(X_actual, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)

    actual = X_actual[-1, -1, :]
    predicted = X_pred[-1, -1, :]

    error = np.abs(actual - predicted)
    return error
