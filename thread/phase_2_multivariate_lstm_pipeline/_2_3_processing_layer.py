# Main Processing Unit : Model training and subsequent processing
import os
import numpy as np
from datetime import datetime
#from keras.models import load_model
from utils.logger_2 import setup_logger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from pathlib import Path

# Bu dosyanın bulunduğu yerden base_dir hesaplayalım
# thread/phase_2_multivariate_lstm_pipeline/_2_3_processing_layer.py
# buradan 2 seviye yukarı çıkınca pm-phase2 kökü:
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "phase2models"
THRESHOLD_DIR = MODEL_DIR / "threshold"
SCALER_DIR = MODEL_DIR / "scalers"
BUFFER_DIR = MODEL_DIR / "historical-buffer"


p2_3_train_log = setup_logger(
    "p2_3_processing_layer_training_logger", "logs/p2_3_processing_layer_training.log"
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models", "phase2models")

#LSTM Configuration
hyperparams = {
    'lstm_units': 64,
    'activation': 'relu',
    'kernel_regularizer': 0.001,
    'dropout_rate': 0.3,
    'optimizer': 'adam',
    'loss': 'mse',
    'batch_size': 64,
    'epochs': 50,
    'validation_split': 0.1,
    'early_stopping_patience': 5,
    'shuffle': True,
}

def check_model_exists(key, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    p2_3_train_log.debug(f"[check_model_exists] key={key}, dir={model_dir}")

    if not model_dir.exists():
        p2_3_train_log.debug(f"[check_model_exists] model_dir does not exist: {model_dir}")
        return False

    keras_files = [
        f.name
        for f in model_dir.iterdir()
        if f.is_file() and f.name.startswith(str(key)) and f.name.endswith(".keras")
    ]
    p2_3_train_log.debug(f"[check_model_exists] found files={keras_files}")

    return len(keras_files) > 0


def check_model_exists_old(key,model_dir='models/phase2models'):
    if not os.path.exists(model_dir):
        return False
    keras_files = [f for f in os.listdir(model_dir) if f.startswith(key) and f.endswith('.keras')]
    
    if not keras_files:
        return False
    else:
        return True

# MODEL CREATION / TRAINING PIPELINE ONLY
def build_lstm_autoencoder(timesteps, n_features):
    p2_3_train_log.debug(f"Building LSTM autoencoder with {hyperparams['lstm_units']} units and dropout {hyperparams['dropout_rate']}")
    inputs = Input(shape=(timesteps, n_features))
    x = LSTM(hyperparams['lstm_units'], activation=hyperparams['activation'], return_sequences=False,
             kernel_regularizer=l2(hyperparams['kernel_regularizer']))(inputs)
    x = Dropout(hyperparams['dropout_rate'])(x)
    x = RepeatVector(timesteps)(x)
    x = LSTM(hyperparams['lstm_units'], activation=hyperparams['activation'], return_sequences=True,
             kernel_regularizer=l2(hyperparams['kernel_regularizer']))(x)
    x = Dropout(hyperparams['dropout_rate'])(x)
    outputs = TimeDistributed(Dense(n_features))(x)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=hyperparams['optimizer'], loss=hyperparams['loss'])

    p2_3_train_log.debug("Model compiled.")
    
    return autoencoder

# TRAINING & SAVING / TRAINING PIPELINE ONLY
def initial_train_and_save(X_train, timesteps, n_features, key, save_dir=MODEL_DIR):
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model = build_lstm_autoencoder(timesteps, n_features)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=hyperparams['early_stopping_patience'],
        restore_best_weights=True
    )

    p2_3_train_log.debug(f"Starting Training for key: {key}")
    model.fit(
        X_train, X_train,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        validation_split=hyperparams['validation_split'],
        shuffle=hyperparams['shuffle'],
        callbacks=[early_stop]
    )
    p2_3_train_log.debug(f"Completed Training for key: {key}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_dir / f"{key}_lstm_autoencoder_{timestamp}.keras"

    try:
        model.save(model_path)
        p2_3_train_log.debug(f"Initial model trained and saved to: {model_path}")
    except Exception as e:
        p2_3_train_log.error(f"Failed to save model to {model_path}: {e}")
        raise

    return str(model_path)


def initial_train_and_save_old(X_train, timesteps, n_features,key, save_dir='models/phase2models'):
    os.makedirs(save_dir, exist_ok=True)
    model = build_lstm_autoencoder(timesteps, n_features)

    early_stop = EarlyStopping(monitor='val_loss', patience=hyperparams['early_stopping_patience'], restore_best_weights=True)

    p2_3_train_log.debug(f"Starting Training for key: {key}")
    model.fit(X_train, X_train,
              epochs=hyperparams['epochs'],
              batch_size=hyperparams['batch_size'],
              validation_split=hyperparams['validation_split'],
              shuffle=hyperparams['shuffle'],
              callbacks=[early_stop])
    p2_3_train_log.debug(f"Completed Training for key: {key}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_path = os.path.join(save_dir, f'{key}_lstm_autoencoder_{timestamp}.keras')
    p2_3_train_log.debug(f"[initial_train_and_save] saving model to: {model_path}")
    model.save(model_path)
    p2_3_train_log.debug(f"[initial_train_and_save] file exists after save: {os.path.exists(model_path)}")
    return model_path

# PREDICTION / TRAINING + INFERENCE PIPELINES
def predict_with_model(model_path, X_input):
    model = load_model(model_path)
    predictions = model.predict(X_input, verbose = 0)
    return predictions

# Thresholding / TRAINING PIPELINE ONLY
def set_threshold(model_path, x_test, percentile=95, save_dir='models/phase2models'):
    """
    Loads a trained LSTM autoencoder, computes reconstruction error on test data,
    sets an anomaly threshold at a given percentile, and saves it.

    Parameters:
    - model_path (str): Path to the saved `.keras` model.
    - x_test (ndarray): Test input data of shape (samples, timesteps, features).
    - percentile (float): Percentile to compute threshold (default: 95).
    - save_dir (str): Directory to save threshold file.

    Returns:
    - threshold (float): The computed anomaly threshold.
    - recon_error (ndarray): Reconstruction errors per test sample.
    - threshold_path (str): Path where threshold was saved.
    """
    # Load model
    model = load_model(model_path)

    # Predict and compute reconstruction error
    x_pred = model.predict(x_test)
    recon_error = np.mean(np.abs(x_test - x_pred), axis=(1, 2))

    # Compute threshold
    threshold = np.percentile(recon_error, percentile)

    # Prepare save path
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold_path = os.path.join(save_dir, f"threshold_{timestamp}.npz")

    # Save threshold and optionally the full error array
    np.savez(threshold_path, threshold=threshold, recon_error=recon_error)

    p2_3_train_log.debug(f"[INFO] Threshold saved to: {threshold_path}")
  #  print(f"[INFO] Threshold saved to: {threshold_path}")

    return threshold, recon_error, threshold_path
