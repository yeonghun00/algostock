"""LightGBM ranker — the default model."""

from __future__ import annotations

from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseRanker


class LGBMRanker(BaseRanker):
    """LightGBM ranking model using LambdaRank for cross-sectional stock ranking."""

    BEST_PARAMS = {
        "objective": "huber",
        "metric": "huber",
        "alpha": 0.9,
        "boosting_type": "gbdt",
        "num_leaves": 7,
        "max_depth": 3,
        "learning_rate": 0.05,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 750,
        "verbose": -1,
        "n_estimators": 1000,
        "n_jobs": -1,
        "seed": 42,
    }

    @staticmethod
    def _compute_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Sort by date and compute group sizes for ranking objective."""
        sorted_df = df.sort_values("date").reset_index(drop=True)
        groups = sorted_df.groupby("date", sort=True).size().values
        return sorted_df, groups

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        params: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LGBMRanker":
        params = params or self.BEST_PARAMS.copy()
        is_ranking = params.get("objective") in ("lambdarank", "rank_xendcg")

        # Sort by date and compute groups for ranking objectives
        if is_ranking:
            train_df, train_groups = self._compute_groups(train_df)
        else:
            train_groups = None

        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.target_col].to_numpy()
        time_weight = self._calculate_time_weights(train_df)
        if sample_weight is not None and time_weight is not None:
            weight = sample_weight * time_weight
            weight = weight / weight.mean()
        elif sample_weight is not None:
            weight = sample_weight
        else:
            weight = time_weight

        train_data = lgb.Dataset(
            X_train, label=y_train, weight=weight,
            feature_name=self.feature_cols, group=train_groups,
        )
        callbacks = [lgb.log_evaluation(period=100)]

        if val_df is not None and len(val_df) > 0:
            if is_ranking:
                val_df, val_groups = self._compute_groups(val_df)
            else:
                val_groups = None
            X_val = val_df[self.feature_cols].to_numpy()
            y_val = val_df[self.target_col].to_numpy()
            val_data = lgb.Dataset(
                X_val, label=y_val, reference=train_data, group=val_groups,
            )
            callbacks.append(lgb.early_stopping(self.patience))
            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get("n_estimators", 3000),
                valid_sets=[val_data],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get("n_estimators", 3000),
                callbacks=callbacks,
            )

        self.logger.info("Trained LGBMRanker with %s samples", len(train_df))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(df[self.feature_cols].to_numpy())

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "importance": self.model.feature_importance(importance_type="gain"),
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
