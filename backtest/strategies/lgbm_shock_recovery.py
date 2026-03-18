"""LightGBM-powered Shock Recovery Strategy.

Replaces the hand-crafted scoring formula in ShockRecoveryStrategy with a
LightGBM model trained on historical trade outcomes.  Uses walk-forward
cross-validation to avoid look-ahead bias, and Optuna to tune LightGBM
hyperparameters by maximising out-of-sample Sharpe ratio.

Key differences from rule-based ShockRecoveryStrategy:
  - No hard thresholds for shock_threshold / volume_spike — LightGBM learns
    the optimal decision surface from all features simultaneously.
  - Feature importances reveal which signals actually drive recovery.
  - Walk-forward ensures the model is always trained on the past, tested on
    the future (same regime as live trading).

Usage:
    from backtest.strategies.lgbm_shock_recovery import LGBMShockRecoveryStrategy

    strategy = LGBMShockRecoveryStrategy(top_n=20)
    strategy.train(
        feature_matrix=fm,      # DataFrame from feature_engineer.build_feature_matrix()
        target_col="target_win_50bps",
        n_optuna_trials=50,
    )
    # Then pass to BacktestEngine as normal strategy
    result = engine.run(strategy, start_date, end_date)
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from backtest.strategies.base import BaseStrategy
from backtest.feature_engineer import (
    build_candidate_features,
    FEATURE_COLS,
    CATEGORICAL_COLS,
)

logger = logging.getLogger(__name__)

# Optuna + LightGBM imported lazily to keep the base module lightweight
try:
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


class LGBMShockRecoveryStrategy(BaseStrategy):
    """
    Walk-forward LightGBM strategy for shock recovery.

    Training protocol:
        For each prediction window of PRED_WINDOW_DAYS:
          - Train on the TRAIN_WINDOW_DAYS immediately before the window
          - Generate signals for each day in the window
        This mimics live trading: always train on the past, predict the future.

    Hyperparameter tuning:
        Optuna minimises -Sharpe on a held-out validation fold inside the
        training window.  Best params are re-fitted on the full training set
        before each prediction window.
    """

    TRAIN_WINDOW_DAYS = 180   # ~9 months of history to train on
    PRED_WINDOW_DAYS  = 60    # ~3 months per walk-forward fold
    MIN_TRAIN_SAMPLES = 200   # Minimum trades needed to fit a model

    def __init__(
        self,
        top_n: int = 20,
        hold_days: int = 3,
        target_pct: float = 0.03,
        stop_pct: float = 0.02,
        target_col: str = "target_win_50bps",
        n_optuna_trials: int = 50,
        lgbm_seed: int = 42,
    ):
        """
        Args:
            top_n:            Stocks to select per day (equal-weighted)
            hold_days:        Max hold duration (adaptive exit in engine)
            target_pct:       Profit target for adaptive exit
            stop_pct:         Stop-loss for adaptive exit
            target_col:       Label column — 'target_win', 'target_win_50bps',
                              or 'target_return' (regression)
            n_optuna_trials:  Optuna trials per walk-forward fold
            lgbm_seed:        Random seed for reproducibility
        """
        if not _LGBM_AVAILABLE:
            raise ImportError(
                "lightgbm and optuna are required: pip install lightgbm optuna"
            )
        super().__init__(
            top_n=top_n,
            hold_days=hold_days,
            target_pct=target_pct,
            stop_pct=stop_pct,
        )
        self.target_col       = target_col
        self.n_optuna_trials  = n_optuna_trials
        self.lgbm_seed        = lgbm_seed

        self._models: dict[date, lgb.Booster] = {}  # prediction_start → model
        self._feature_matrix: Optional[pd.DataFrame] = None
        self._best_params_log: list[dict] = []

        # Runtime data (set before calling generate_signals)
        self._runtime_data: dict = {}

    # ─────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────

    def train(
        self,
        feature_matrix: pd.DataFrame,
        target_col: Optional[str] = None,
        n_optuna_trials: Optional[int] = None,
    ) -> "LGBMShockRecoveryStrategy":
        """
        Walk-forward train LightGBM models across the feature_matrix date range.

        Args:
            feature_matrix:  Output of feature_engineer.build_feature_matrix()
            target_col:      Override default target column
            n_optuna_trials: Override default Optuna trials

        Returns:
            self (for chaining)
        """
        if target_col:
            self.target_col = target_col
        if n_optuna_trials is not None:
            self.n_optuna_trials = n_optuna_trials

        self._feature_matrix = feature_matrix.copy()
        fm = self._feature_matrix

        # Need _signal_date as a date object for windowing
        fm["_signal_date"] = pd.to_datetime(fm["_signal_date"]).dt.date
        all_dates = sorted(fm["_signal_date"].unique())

        if len(all_dates) < 2:
            logger.warning("Feature matrix has < 2 dates; cannot walk-forward train.")
            return self

        start = all_dates[0] + timedelta(days=self.TRAIN_WINDOW_DAYS)
        end   = all_dates[-1]

        # Walk forward: step through prediction windows
        pred_start = start
        n_folds = 0
        while pred_start <= end:
            pred_end_dt = min(
                pred_start + timedelta(days=self.PRED_WINDOW_DAYS - 1), end
            )
            train_start_dt = pred_start - timedelta(days=self.TRAIN_WINDOW_DAYS)

            train_mask = (fm["_signal_date"] >= train_start_dt) & \
                         (fm["_signal_date"] <  pred_start)
            train_df = fm[train_mask]

            if len(train_df) < self.MIN_TRAIN_SAMPLES:
                logger.warning(
                    "Fold %s: only %d training samples (< %d) — skipping",
                    pred_start, len(train_df), self.MIN_TRAIN_SAMPLES,
                )
                pred_start += timedelta(days=self.PRED_WINDOW_DAYS)
                continue

            logger.info(
                "Walk-forward fold: train %s→%s (%d samples), predict %s→%s",
                train_start_dt, pred_start - timedelta(days=1), len(train_df),
                pred_start, pred_end_dt,
            )

            model = self._fit_fold(train_df)
            if model is not None:
                self._models[pred_start] = model
                n_folds += 1

            pred_start += timedelta(days=self.PRED_WINDOW_DAYS)

        logger.info("Walk-forward training complete: %d folds fitted.", n_folds)
        return self

    def feature_importance(self) -> pd.DataFrame:
        """
        Average feature importance across all walk-forward models.

        Returns DataFrame with columns: feature, avg_importance, std_importance
        """
        if not self._models:
            return pd.DataFrame()

        imp_dfs = []
        for model in self._models.values():
            fi = pd.DataFrame({
                "feature":    model.feature_name(),
                "importance": model.feature_importance(importance_type="gain"),
            })
            imp_dfs.append(fi)

        all_imp = pd.concat(imp_dfs)
        summary = (
            all_imp.groupby("feature")["importance"]
            .agg(avg_importance="mean", std_importance="std")
            .sort_values("avg_importance", ascending=False)
            .reset_index()
        )
        return summary

    def set_runtime_data(self, data: dict):
        """Inject live data dict (same format as BacktestEngine data_loader output)."""
        self._runtime_data = data

    # ─────────────────────────────────────────────────────────────────────
    #  BaseStrategy interface
    # ─────────────────────────────────────────────────────────────────────

    def generate_signals(self, target_date: date, data: dict) -> pd.DataFrame:
        """
        Score all candidate stocks using the appropriate walk-forward model.

        Falls back to rule-based shock_score if no model available for date.
        """
        signal_date = target_date - timedelta(days=1)
        model = self._get_model_for_date(signal_date)

        quotes     = data.get("quotes", pd.DataFrame())
        us_data    = data.get("us_overnight", {})
        tdnet_df   = data.get("tdnet",   pd.DataFrame())
        edinet_df  = data.get("edinet",  pd.DataFrame())
        short_df   = data.get("short",   pd.DataFrame())
        margin_df  = data.get("margin",  pd.DataFrame())
        universe_df = data.get("universe", pd.DataFrame())

        if quotes.empty:
            return pd.DataFrame()

        # Get all candidate codes (stocks with price data for signal_date)
        quotes = quotes.copy()
        quotes["code"]      = quotes["code"].astype(str).str[:4]
        quotes["date_only"] = pd.to_datetime(quotes["date"]).dt.date
        today_quotes = quotes[quotes["date_only"] == signal_date]

        if today_quotes.empty:
            return pd.DataFrame()

        # Build features for all candidates
        feature_rows = []
        for _, row in today_quotes.iterrows():
            code = str(row["code"])[:4]
            feats = build_candidate_features(
                code=code,
                signal_date=signal_date,
                quotes=quotes,
                topix_quotes=data.get("topix", pd.DataFrame()),
                us_data=us_data,
                tdnet_df=tdnet_df,
                edinet_df=edinet_df,
                short_df=short_df,
                margin_df=margin_df,
                universe_df=universe_df,
            )
            if feats is not None:
                feature_rows.append(feats)

        if not feature_rows:
            return pd.DataFrame()

        feat_df = pd.DataFrame(feature_rows)

        # Exclude stocks with negative disclosures (hard filter — model cannot override)
        feat_df = feat_df[feat_df["disc_has_negative"] == 0]

        # Must show some drop (loose filter — LightGBM handles fine-tuning)
        feat_df = feat_df[feat_df["price_total_ret"] < -0.01]

        if feat_df.empty:
            return pd.DataFrame()

        if model is not None:
            scores = self._predict(model, feat_df)
        else:
            # Fallback: rule-based shock score (normalised 0–1)
            logger.debug("No model for %s — using rule-based fallback", signal_date)
            scores = self._rule_based_score(feat_df)

        feat_df["score"] = scores
        result = (
            feat_df[["_code", "score", "price_total_ret", "vol_ratio_20d",
                     "supply_short_ratio", "macro_sp500", "disc_has_positive"]]
            .rename(columns={"_code": "code"})
            .sort_values("score", ascending=False)
            .head(self.top_n)
            .reset_index(drop=True)
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _fit_fold(self, train_df: pd.DataFrame) -> Optional["lgb.Booster"]:
        """Tune hyperparameters with Optuna then refit on full training fold."""
        X = train_df[FEATURE_COLS].copy()
        y = train_df[self.target_col]

        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")

        is_classification = self.target_col in ("target_win", "target_win_50bps")

        # ── Optuna study ──────────────────────────────────────────────────
        best_params = self._optuna_tune(X, y, is_classification)

        # ── Refit on full fold ────────────────────────────────────────────
        try:
            dtrain = lgb.Dataset(
                X, label=y,
                categorical_feature=CATEGORICAL_COLS,
                free_raw_data=False,
            )
            params = {
                **best_params,
                "objective":  "binary" if is_classification else "regression",
                "metric":     "binary_logloss" if is_classification else "rmse",
                "verbosity":  -1,
                "seed":       self.lgbm_seed,
            }
            model = lgb.train(
                params, dtrain,
                num_boost_round=best_params.pop("n_estimators", 200),
            )
            self._best_params_log.append(best_params)
            return model
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)
            return None

    def _optuna_tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool,
    ) -> dict:
        """
        Use Optuna to maximise out-of-sample Sharpe on a time-series split.

        The last 30% of training data is used as the validation fold
        (preserving temporal order — no shuffling).
        """
        n = len(X)
        split = int(n * 0.70)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        if len(X_tr) < 50 or len(X_val) < 20:
            # Not enough data; return safe defaults
            return _default_lgbm_params(is_classification)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "objective":        "binary" if is_classification else "regression",
                "metric":           "binary_logloss" if is_classification else "rmse",
                "verbosity":        -1,
                "seed":             self.lgbm_seed,
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 16, 128),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "min_child_samples":trial.suggest_int("min_child_samples", 10, 60),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            }
            n_est = params.pop("n_estimators")
            try:
                dtrain = lgb.Dataset(
                    X_tr, label=y_tr,
                    categorical_feature=CATEGORICAL_COLS,
                    free_raw_data=False,
                )
                model = lgb.train(params, dtrain, num_boost_round=n_est)

                if is_classification:
                    preds = model.predict(X_val)
                    # Rank validation stocks by predicted win probability
                    # and compute "Sharpe-like" quality of that ranking
                    score = _ranking_sharpe(preds, y_val.values)
                else:
                    preds = model.predict(X_val)
                    score = _ranking_sharpe(preds, y_val.values)

                return -score  # Optuna minimises
            except Exception:
                return 0.0

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=self.lgbm_seed))
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)

        best = study.best_params
        best["n_estimators"] = best.pop("n_estimators", 200)
        return best

    def _predict(self, model: "lgb.Booster", feat_df: pd.DataFrame) -> np.ndarray:
        X = feat_df[FEATURE_COLS].copy()
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return model.predict(X)

    def _rule_based_score(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Fallback: simple linear combination mirroring the original formula."""
        df = feat_df
        score = (
            df["price_total_ret"].abs() * 10           # drop magnitude
            + (df["vol_ratio_20d"] - 1).clip(0) * 8   # volume spike
            + df["supply_short_ratio"].fillna(0) * 1.5  # short fuel
            + df["macro_sector_etf"].clip(0) * 10      # sector recovery
            + df["shock_is_macro"] * 10                # macro-confirmed
            + df["shock_excess_drop"].abs() * 8        # relative weakness
        )
        return score.values

    def _get_model_for_date(self, signal_date: date) -> Optional["lgb.Booster"]:
        """Return the walk-forward model whose prediction window covers signal_date."""
        if not self._models:
            return None
        # Find the most recent prediction-start that is <= signal_date
        valid = [d for d in self._models if d <= signal_date]
        if not valid:
            return None
        return self._models[max(valid)]


# ─────────────────────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _ranking_sharpe(scores: np.ndarray, actual_returns: np.ndarray) -> float:
    """
    Evaluate ranking quality as a Sharpe proxy:
      Take the top-quartile stocks by score → compute mean return of that bucket.
      Divide by std to get a Sharpe-like ratio.
    This aligns the Optuna objective with the actual trading objective
    (pick the top-N stocks with highest expected return).
    """
    if len(scores) < 4:
        return 0.0
    cutoff = np.percentile(scores, 75)
    selected = actual_returns[scores >= cutoff]
    if len(selected) == 0:
        return 0.0
    mean_ret = selected.mean()
    std_ret  = selected.std()
    return mean_ret / std_ret if std_ret > 0 else mean_ret * 10


def _default_lgbm_params(is_classification: bool) -> dict:
    return {
        "learning_rate":     0.05,
        "num_leaves":        31,
        "max_depth":         5,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "n_estimators":      200,
    }
