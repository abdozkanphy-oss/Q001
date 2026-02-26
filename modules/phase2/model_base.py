from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class Phase2Score:
    score: float
    is_anomaly: bool
    threshold: float
    detail: Dict[str, Any]


class Phase2Model:
    """Interface for Phase2 models.

    Models score a *sequence* of vectors (timesteps x n_features).
    Runtime is responsible for producing the sequence and applying the same preprocessing
    contract as offline training.
    """

    def score_sequence(self, X_seq) -> Phase2Score:
        raise NotImplementedError
