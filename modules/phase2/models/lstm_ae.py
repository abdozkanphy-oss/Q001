from __future__ import annotations

import numpy as np

from modules.phase2.model_base import Phase2Model, Phase2Score


class LSTMAEModel(Phase2Model):
    def __init__(self, *, keras_model, threshold: float) -> None:
        self._model = keras_model
        self._threshold = float(threshold)

    def score_sequence(self, X_seq) -> Phase2Score:
        # X_seq: np.ndarray shape (1, T, F)
        X = np.asarray(X_seq, dtype=float)
        if X.ndim != 3 or X.shape[0] != 1:
            raise ValueError(f"X_seq must be (1,T,F); got {X.shape}")

        pred = self._model.predict(X, verbose=0)
        # per-feature mean squared reconstruction error (mean over time)
        per_feat = np.mean(np.square(pred - X), axis=(0, 1))
        # global score: mean over features
        err = float(np.mean(per_feat))
        is_anom = bool(err >= self._threshold)
        return Phase2Score(
            score=err,
            is_anomaly=is_anom,
            threshold=float(self._threshold),
            detail={"score_type": "recon_mse", "per_feature_mse": per_feat.astype(float).tolist()},
        )
