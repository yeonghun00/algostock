"""Base ranker class — shared interface for all ML models."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """Abstract base for stock ranking models."""

    BEST_PARAMS: Dict = {}

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "target_rank_21d",
        time_decay: float = 0.4,
        patience: int = 80,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.time_decay = time_decay
        self.patience = patience
        self.model = None
        self.metadata: Dict = {}  # arbitrary backtest config (market_cap, horizon, etc.)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _calculate_time_weights(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.time_decay <= 0:
            return None
        dates = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        if dates.isna().all():
            return None
        min_date = dates.min()
        max_date = dates.max()
        span = (max_date - min_date).days
        if span <= 0:
            return None
        age = (dates - min_date).dt.days / span
        raw = np.exp(self.time_decay * 2.0 * age)
        return (raw / raw.mean()).to_numpy()

    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        params: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseRanker":
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def rank_stocks(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        ranked = df.copy()
        ranked["ml_score"] = self.predict(ranked)
        ranked["ml_rank"] = ranked["ml_score"].rank(ascending=False, method="first")
        cols = ["stock_code", "name", "closing_price", "market_cap", "ml_score", "ml_rank"]
        cols.extend(self.feature_cols[:min(6, len(self.feature_cols))])
        cols = [c for c in cols if c in ranked.columns]
        return ranked.nsmallest(top_n, "ml_rank")[cols]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["feature", "importance"])

    def save(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "model_class": self.__class__.__name__,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "time_decay": self.time_decay,
            "saved_at": datetime.now().isoformat(),
            "version": "unified_v2",
            "metadata": self.metadata,
        }
        with out_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BaseRanker":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)

        # Determine correct subclass
        from ml.models import get_model_class
        model_class_name = payload.get("model_class", "LGBMRanker")
        class_map = {"LGBMRanker": "lgbm", "XGBRanker": "xgboost", "CatBoostRanker": "catboost", "MLRanker": "lgbm"}
        model_key = class_map.get(model_class_name, "lgbm")
        target_cls = get_model_class(model_key)

        ranker = target_cls(
            feature_cols=payload["feature_cols"],
            target_col=payload.get("target_col", "target_rank_21d"),
            time_decay=payload.get("time_decay", 0.0),
        )
        ranker.model = payload["model"]
        ranker.metadata = payload.get("metadata", {})
        return ranker
