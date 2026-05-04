"""
demand_forecast.py
==================
Part A — Localized Demand Prediction

What this file does:
  1. Trains a Gradient Boosting model (XGBoost-equivalent) to forecast
     short-term electricity demand at zone/feeder level.
  2. Implements a seasonal decomposition model (Prophet-equivalent) that
     separates trend, weekly seasonality, and daily seasonality.
  3. Combines both models into an ensemble forecast.
  4. Classifies each zone into risk levels: LOW / MEDIUM / HIGH / CRITICAL
     based on predicted demand vs historical capacity thresholds.
  5. Evaluates against a naive baseline (yesterday's values).

NOTE ON LIBRARIES:
  We use sklearn's GradientBoostingRegressor here (same algorithm as XGBoost).
  To switch to XGBoost on your own machine:
    pip install xgboost
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
  Everything else stays identical.

  To switch to Prophet for the seasonal model:
    pip install prophet
    from prophet import Prophet
  See SeasonalModel.fit_prophet() method at the bottom of this file.

Input:  features_zone_clean.csv   (from feature_engineering.py)
Output: forecasts.csv, zone_risk.csv, forecast_model.pkl
"""

import numpy as np
import pandas as pd
import joblib                          # Save/load trained models to disk
import os
import warnings
warnings.filterwarnings("ignore")      # Suppress sklearn convergence warnings in demo

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Configuration
# ══════════════════════════════════════════════════════════════════════════════

FORECAST_CONFIG = {
    # Features the Gradient Boosting model will use as inputs
    # These come directly from features_zone_clean.csv
    "feature_cols": [
        "hour_sin", "hour_cos",          # Time of day (cyclical)
        "dow_sin", "dow_cos",            # Day of week (cyclical)
        "is_weekend",                    # Binary weekend flag
        "month",                         # Month (1-12)
        "zone_roll_mean_24h",            # Rolling 24h zone average
        "zone_lag_24h",                  # Same time yesterday
        "zone_lag_7d",                   # Same time last week
        "mean_consumption",              # Average per-meter consumption
        "std_consumption",               # Spread within zone
    ],

    # Target variable: what we're predicting
    "target_col": "total_consumption",

    # Forecast horizon: how many steps ahead to predict
    # 4 steps × 15 min = 1 hour ahead
    # 96 steps × 15 min = 24 hours ahead
    "horizons": {
        "1h":  4,
        "6h":  24,
        "24h": 96,
    },

    # Risk thresholds — what fraction of a zone's historical peak
    # triggers each risk level
    "risk_thresholds": {
        "LOW":      0.60,    # < 60% of historical peak → safe
        "MEDIUM":   0.75,    # 60–75% → moderate load
        "HIGH":     0.90,    # 75–90% → elevated risk, prepare
        "CRITICAL": 1.00,    # > 90% → immediate action needed
    },

    # Train/test split: train on first 75 days, test on last 15 days
    "train_days": 75,

    # GradientBoostingRegressor hyperparameters
    # (mirror these to XGBRegressor for XGBoost)
    "gb_params": {
        "n_estimators":  200,      # Number of boosting rounds (trees)
        "max_depth":     5,        # Max depth of each tree
        "learning_rate": 0.05,     # How much each tree contributes (shrinkage)
        "subsample":     0.8,      # Fraction of rows used per tree (prevents overfit)
        "min_samples_leaf": 10,    # Min samples in a leaf (regularisation)
        "random_state":  42,       # Reproducibility
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Data Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_zone_data(path: str = "features_zone_clean.csv") -> pd.DataFrame:
    """
    Loads the zone-level feature file produced by feature_engineering.py.
    Validates required columns are present.
    """

    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = (
        FORECAST_CONFIG["feature_cols"] +
        [FORECAST_CONFIG["target_col"], "zone_id", "timestamp"]
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in zone feature file: {missing}")

    # Sort by zone and time — critical for lag-based models
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    print(f"Loaded zone data: {len(df):,} rows, {df['zone_id'].nunique()} zones")
    print(f"Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Train/Test Split (Time-Aware)
# ══════════════════════════════════════════════════════════════════════════════

def time_split(df: pd.DataFrame, train_days: int) -> tuple:
    """
    Splits data into train and test sets by time — NOT randomly.

    WHY NOT RANDOM SPLIT?
    In time-series forecasting, random splitting causes "data leakage":
    the model sees future data during training and learns to "cheat".
    We always split at a point in time — train on past, test on future.

    train_days: number of days to use for training
    Returns: (train_df, test_df)
    """

    cutoff_date = df["timestamp"].min() + pd.Timedelta(days=train_days)

    train = df[df["timestamp"] < cutoff_date].copy()
    test  = df[df["timestamp"] >= cutoff_date].copy()

    print(f"\nTrain period: {train['timestamp'].min().date()} → {train['timestamp'].max().date()}")
    print(f"Test period:  {test['timestamp'].min().date()}  → {test['timestamp'].max().date()}")
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")

    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Gradient Boosting Forecaster (XGBoost-equivalent)
# ══════════════════════════════════════════════════════════════════════════════

class GBForecaster:
    """
    Gradient Boosting demand forecaster — one model trained per zone.

    Why one model per zone?
      Each zone has different base load, peak times, and patterns.
      A single global model would average out these differences.
      Zone-specific models give much better accuracy.

    Why Gradient Boosting?
      - Handles non-linear relationships between features and demand
      - Built-in feature importance (tells us which features matter most)
      - Robust to outliers (resistant to anomalous meter readings in train set)
      - Fast inference (sub-millisecond per prediction)
      - No black box — fully explainable with SHAP

    TO SWITCH TO XGBoost (on your own machine):
      from xgboost import XGBRegressor
      self.model = XGBRegressor(**FORECAST_CONFIG["gb_params"])
      Everything else stays the same.
    """

    def __init__(self):
        self.models   = {}    # zone_id → trained GBR model
        self.scalers  = {}    # zone_id → fitted StandardScaler
        self.feature_importances = {}
        self.imputers = {}   # zone_id → feature importance dict

    def fit(self, train_df: pd.DataFrame):
        """
        Trains one Gradient Boosting model per zone.

        StandardScaler is applied to features before training.
        GBR can work without scaling, but it speeds up convergence.
        """

        feature_cols = FORECAST_CONFIG["feature_cols"]
        target_col   = FORECAST_CONFIG["target_col"]

        for zone_id in train_df["zone_id"].unique():
            zone_train = train_df[train_df["zone_id"] == zone_id].copy()

            X = zone_train[feature_cols].values  # Feature matrix: shape (n_rows, n_features)
            y = zone_train[target_col].values    # Target vector: shape (n_rows,)

            # Scale features to zero mean, unit variance
            # GBR is less sensitive to this than linear models, but it helps
            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(X)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            # fit_transform: computes mean/std from X, then scales X
            # On test data we only call transform() — no refitting

            # Train the model
            model = GradientBoostingRegressor(**FORECAST_CONFIG["gb_params"])
            model.fit(X_scaled, y)
            # model.fit learns a sequence of decision trees where each tree
            # corrects the errors of the previous ones (boosting)

            # Store model and scaler
            self.models[zone_id]  = model
            self.scalers[zone_id] = scaler
            self.imputers[zone_id] = imputer

            # Extract feature importance
            # model.feature_importances_ is an array of shape (n_features,)
            # Each value = how much that feature reduced prediction error
            self.feature_importances[zone_id] = dict(
                zip(feature_cols, model.feature_importances_)
            )

        print(f"  Trained {len(self.models)} zone-specific GB models")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for a DataFrame that may contain multiple zones.
        Returns an array of predictions aligned with df's row order.
        """

        feature_cols = FORECAST_CONFIG["feature_cols"]
        predictions  = np.zeros(len(df))   # Pre-allocate output array

        for zone_id in df["zone_id"].unique():
            # Get the indices of rows belonging to this zone
            zone_mask = df["zone_id"] == zone_id
            zone_data = df[zone_mask][feature_cols].values

            # Scale using the same scaler fitted on training data
            # NEVER refit the scaler on test data — that would cause leakage
            X_imputed = self.imputers[zone_id].transform(zone_data)
            X_scaled = self.scalers[zone_id].transform(X_imputed)

            # Predict and place results back at the correct indices
            predictions[zone_mask] = self.models[zone_id].predict(X_scaled)

        return predictions

    def get_top_features(self, zone_id: str, n: int = 5) -> dict:
        """Returns the top N most important features for a given zone."""
        importances = self.feature_importances.get(zone_id, {})
        sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:n])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Seasonal Decomposition Model (Prophet-equivalent)
# ══════════════════════════════════════════════════════════════════════════════

class SeasonalModel:
    """
    Decomposes zone demand into three additive components:
      total = trend + weekly_seasonality + daily_seasonality + residual

    This replicates Prophet's core decomposition without requiring installation.

    Prophet uses Fourier series to model seasonality. We use the same idea:
    represent each seasonal pattern as a sum of sine and cosine waves.
    The number of Fourier terms controls how flexible the pattern is.

    TO USE ACTUAL PROPHET (on your own machine):
      pip install prophet
      See fit_prophet() method below — just uncomment it.
    """

    def __init__(self, fourier_order: int = 5):
        # fourier_order: number of sin/cos pairs to use
        # Higher = more complex seasonal pattern can be captured
        # Too high → overfitting; typical range 3–10
        self.fourier_order = fourier_order
        self.zone_profiles = {}   # zone_id → fitted seasonal profile

    def _fourier_features(self, timestamps: pd.Series, period: float) -> np.ndarray:
        """
        Creates Fourier series features for a given period.

        For each order k from 1 to fourier_order:
          feature_2k-1 = sin(2π × k × t / period)
          feature_2k   = cos(2π × k × t / period)

        t = time in hours from the start of the dataset
        period = length of the cycle in hours (24 for daily, 168 for weekly)

        Returns array of shape (n_rows, 2 × fourier_order)
        """

        # Convert timestamp to hours elapsed since dataset start
        t = (timestamps - timestamps.min()).dt.total_seconds() / 3600.0

        features = []
        for k in range(1, self.fourier_order + 1):
            features.append(np.sin(2 * np.pi * k * t / period))
            features.append(np.cos(2 * np.pi * k * t / period))

        # Stack into 2D array: shape (n_rows, 2 × fourier_order)
        return np.column_stack(features)

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the seasonal model per zone using Ridge regression.

        Ridge (L2 regularised linear regression) finds the Fourier coefficients
        that best explain zone demand while preventing overfitting.
        """
        from sklearn.linear_model import Ridge

        for zone_id in train_df["zone_id"].unique():
            zone_data = train_df[train_df["zone_id"] == zone_id].copy()

            # Build Fourier feature matrix: daily (period=24h) + weekly (period=168h)
            daily_feats  = self._fourier_features(zone_data["timestamp"], period=24)
            weekly_feats = self._fourier_features(zone_data["timestamp"], period=168)

            # Trend feature: linear time component (handles gradual growth/decline)
            t_normalized = (
                (zone_data["timestamp"] - zone_data["timestamp"].min())
                .dt.total_seconds() / 3600.0
            ).values.reshape(-1, 1)  # shape (n_rows, 1)

            # Concatenate all features: [trend, daily_fourier, weekly_fourier]
            X = np.hstack([t_normalized, daily_feats, weekly_feats])

            y = zone_data[FORECAST_CONFIG["target_col"]].values

            # Ridge regression: minimises (MSE + α × sum(weights²))
            # α=1.0 is a mild regularisation — prevents Fourier coefficients from blowing up
            model = Ridge(alpha=1.0)
            model.fit(X, y)

            self.zone_profiles[zone_id] = {
                "model":      model,
                "start_time": zone_data["timestamp"].min(),
            }

        print(f"  Fitted seasonal profiles for {len(self.zone_profiles)} zones")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generates seasonal baseline predictions for each zone."""

        predictions = np.zeros(len(df))

        for zone_id in df["zone_id"].unique():
            zone_mask = df["zone_id"] == zone_id
            zone_data = df[zone_mask].copy()

            profile    = self.zone_profiles[zone_id]
            start_time = profile["start_time"]

            # Build same feature matrix as in fit() but for test timestamps
            ts = zone_data["timestamp"]
            daily_feats  = self._fourier_features(ts, period=24)
            weekly_feats = self._fourier_features(ts, period=168)
            t_normalized = (
                (ts - start_time).dt.total_seconds() / 3600.0
            ).values.reshape(-1, 1)

            X = np.hstack([t_normalized, daily_feats, weekly_feats])

            predictions[zone_mask] = profile["model"].predict(X)

        return predictions

    # ── Prophet version (uncomment on your own machine) ──────────────────────
    # def fit_prophet(self, train_df: pd.DataFrame):
    #     from prophet import Prophet
    #     for zone_id in train_df["zone_id"].unique():
    #         zone_data = train_df[train_df["zone_id"] == zone_id].copy()
    #         prophet_df = zone_data.rename(columns={
    #             "timestamp": "ds",
    #             "total_consumption": "y"
    #         })[["ds", "y"]]
    #         m = Prophet(
    #             daily_seasonality=True,
    #             weekly_seasonality=True,
    #             yearly_seasonality=False,
    #         )
    #         m.fit(prophet_df)
    #         self.zone_profiles[zone_id] = m
    #
    # def predict_prophet(self, df: pd.DataFrame) -> np.ndarray:
    #     predictions = np.zeros(len(df))
    #     for zone_id in df["zone_id"].unique():
    #         zone_mask = df["zone_id"] == zone_id
    #         zone_data = df[zone_mask].rename(columns={"timestamp": "ds"})
    #         forecast = self.zone_profiles[zone_id].predict(zone_data[["ds"]])
    #         predictions[zone_mask] = forecast["yhat"].values
    #     return predictions


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Ensemble Combiner
# Combines GB and Seasonal predictions with a learned weight.
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleForecaster:
    """
    Combines the Gradient Boosting and Seasonal models:
      final_prediction = α × GB_prediction + (1 - α) × Seasonal_prediction

    The weight α is learned from validation data to minimise MAE.

    Why ensemble?
      - GB is good at capturing non-linear patterns and exogenous features
      - Seasonal model is good at capturing regular repeating patterns
      - Combining them gets the best of both worlds
      - Ensembles almost always outperform individual models
    """

    def __init__(self, gb_model: GBForecaster, seasonal_model: SeasonalModel):
        self.gb       = gb_model
        self.seasonal = seasonal_model
        self.alpha    = 0.7   # Default: favour GB slightly over seasonal

    def fit_weight(self, val_df: pd.DataFrame):
        """
        Finds the optimal α by grid search on validation data.
        Tries α values from 0.0 to 1.0 in steps of 0.05.
        Picks the α that minimises Mean Absolute Error.
        """

        gb_preds  = self.gb.predict(val_df)
        sea_preds = self.seasonal.predict(val_df)
        actuals   = val_df[FORECAST_CONFIG["target_col"]].values

        best_alpha = 0.7
        best_mae   = float("inf")

        for alpha in np.arange(0.0, 1.05, 0.05):
            combined = alpha * gb_preds + (1 - alpha) * sea_preds
            mae      = mean_absolute_error(actuals, combined)
            if mae < best_mae:
                best_mae   = mae
                best_alpha = alpha

        self.alpha = round(best_alpha, 2)
        print(f"  Optimal ensemble weight: α={self.alpha:.2f} "
              f"(GB={self.alpha:.0%}, Seasonal={1-self.alpha:.0%}), "
              f"Validation MAE={best_mae:.4f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generates ensemble predictions."""

        gb_preds  = self.gb.predict(df)
        sea_preds = self.seasonal.predict(df)

        # Clip to 0 — demand can't be negative
        return np.clip(
            self.alpha * gb_preds + (1 - self.alpha) * sea_preds,
            0, None
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Naive Baseline
# All ML models should be compared to a simple baseline.
# If your model barely beats "just use yesterday's value", it's not useful.
# ══════════════════════════════════════════════════════════════════════════════

def naive_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Naive baseline: predict that demand equals yesterday's value at the same time.
    This is the "same time yesterday" model — the simplest meaningful benchmark.

    If our ensemble can't beat this, we haven't done anything useful.
    We expect our model to beat it significantly (especially on weekends/holidays).
    """

    # zone_lag_24h is already computed in feature_engineering.py
    # It's the same-time-yesterday value per zone
    return df["zone_lag_24h"].values


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(actuals: np.ndarray, predictions: np.ndarray, name: str) -> dict:
    """
    Computes standard forecast evaluation metrics.

    MAE  (Mean Absolute Error)      — average prediction error in kWh
    RMSE (Root Mean Squared Error)  — penalises large errors more than MAE
    MAPE (Mean Absolute % Error)    — percentage error (scale-independent)
    R²   (R-squared)                — % of variance explained (1.0 = perfect)
    """

    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100
    r2   = r2_score(actuals, predictions)

    results = {
        "model": name,
        "MAE":   round(mae, 4),
        "RMSE":  round(rmse, 4),
        "MAPE":  round(mape, 2),
        "R2":    round(r2, 4),
    }

    print(f"\n  [{name}]")
    print(f"    MAE  = {mae:.4f} kWh     (average error per 15-min interval)")
    print(f"    RMSE = {rmse:.4f} kWh")
    print(f"    MAPE = {mape:.2f}%       (percentage error)")
    print(f"    R²   = {r2:.4f}          (1.0 = perfect, 0 = useless)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Zone Risk Classifier
# ══════════════════════════════════════════════════════════════════════════════

def classify_zone_risk(forecast_df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each zone at each timestamp into a risk level based on
    predicted demand relative to its historical peak.

    Risk levels:
      LOW      → predicted < 60% of peak   → normal operation
      MEDIUM   → 60–75% of peak            → monitor closely
      HIGH     → 75–90% of peak            → prepare load shedding
      CRITICAL → > 90% of peak             → immediate action required

    This directly answers the problem statement requirement:
    "identify high-risk zones for peak load or grid stress"
    """

    # Compute historical peak per zone (from training data)
    zone_peaks = (
        zone_df.groupby("zone_id")["total_consumption"]
        .quantile(0.95)   # 95th percentile = peak (not absolute max, avoids outliers)
        .rename("zone_peak")
        .reset_index()
    )

    # Join peaks onto forecast DataFrame
    result = forecast_df.merge(zone_peaks, on="zone_id", how="left")

    # Compute load factor: predicted / peak
    result["load_factor"] = result["predicted_demand"] / result["zone_peak"]

    # Assign risk level using thresholds
    thresholds = FORECAST_CONFIG["risk_thresholds"]

    def assign_risk(load_factor: float) -> str:
        if load_factor >= thresholds["HIGH"]:
            return "CRITICAL"
        elif load_factor >= thresholds["MEDIUM"]:
            return "HIGH"
        elif load_factor >= thresholds["LOW"]:
            return "MEDIUM"
        else:
            return "LOW"

    result["risk_level"] = result["load_factor"].apply(assign_risk)

    # Risk score: 0–100 scale (useful for dashboard colour gradients)
    result["risk_score"] = (result["load_factor"] * 100).clip(0, 100).round(1)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast_pipeline(input_path: str = "features_zone_clean.csv") -> pd.DataFrame:
    """
    Full demand forecasting pipeline:
      Load → Split → Train GB → Train Seasonal → Ensemble → Evaluate → Risk classify
    """

    print("BESCOM Demand Forecast Pipeline — Part A")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/7] Loading zone features...")
    df = load_zone_data(input_path)

    # ── Train/test split ──────────────────────────────────────────────────────
    print("\n[2/7] Splitting into train/validation/test sets...")
    train_df, test_df = time_split(df, FORECAST_CONFIG["train_days"])

    # Further split train into train + validation (last 5 days of train)
    # Validation is used to tune ensemble weight without touching test set
    val_cutoff = train_df["timestamp"].max() - pd.Timedelta(days=5)
    val_df     = train_df[train_df["timestamp"] >= val_cutoff].copy()
    train_df   = train_df[train_df["timestamp"] <  val_cutoff].copy()

    print(f"  Final train: {len(train_df):,} rows | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # ── Train Gradient Boosting model ─────────────────────────────────────────
    print("\n[3/7] Training Gradient Boosting forecaster (per zone)...")
    gb = GBForecaster()
    gb.fit(train_df)

    # Print top features for first zone
    first_zone = train_df["zone_id"].iloc[0]
    top_feats  = gb.get_top_features(first_zone, n=5)
    print(f"  Top features for {first_zone}:")
    for feat, imp in top_feats.items():
        bar = "█" * int(imp * 100)
        print(f"    {feat:30s} {bar} {imp:.3f}")

    # ── Train Seasonal model ──────────────────────────────────────────────────
    print("\n[4/7] Fitting seasonal decomposition model (per zone)...")
    seasonal = SeasonalModel(fourier_order=5)
    seasonal.fit(train_df)

    # ── Build ensemble ────────────────────────────────────────────────────────
    print("\n[5/7] Building ensemble and tuning weights on validation set...")
    ensemble = EnsembleForecaster(gb, seasonal)
    ensemble.fit_weight(val_df)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[6/7] Evaluating on held-out test set...")
    actuals       = test_df[FORECAST_CONFIG["target_col"]].values
    gb_preds      = gb.predict(test_df)
    seasonal_preds= seasonal.predict(test_df)
    ensemble_preds= ensemble.predict(test_df)
    baseline_preds= naive_baseline_predict(test_df)

    # Evaluate all four
    results = []
    results.append(evaluate(actuals, baseline_preds,  "Naive baseline (yesterday)"))
    results.append(evaluate(actuals, seasonal_preds,  "Seasonal model"))
    results.append(evaluate(actuals, gb_preds,        "Gradient Boosting"))
    results.append(evaluate(actuals, ensemble_preds,  "Ensemble (GB + Seasonal)"))

    # Compute improvement over baseline
    baseline_mae = results[0]["MAE"]
    ensemble_mae = results[3]["MAE"]
    improvement  = (baseline_mae - ensemble_mae) / baseline_mae * 100
    print(f"\n  Ensemble improves over naive baseline by {improvement:.1f}%")

    # ── Build forecast output DataFrame ───────────────────────────────────────
    print("\n[7/7] Building forecast DataFrame and classifying zone risk...")

    # Generate forecasts for the FULL dataset (train + test) for dashboard
    all_preds = ensemble.predict(df)

    forecast_df = df[["timestamp", "zone_id", FORECAST_CONFIG["target_col"]]].copy()
    forecast_df = forecast_df.rename(columns={FORECAST_CONFIG["target_col"]: "actual_demand"})
    forecast_df["predicted_demand"] = np.round(all_preds, 4)
    forecast_df["prediction_error"] = np.round(
        forecast_df["actual_demand"] - forecast_df["predicted_demand"], 4
    )
    forecast_df["is_test_period"] = (
        forecast_df["timestamp"] >= test_df["timestamp"].min()
    ).astype(int)

    # Classify risk levels
    forecast_df = classify_zone_risk(forecast_df, df)

    # Risk summary
    print("\n  Zone risk summary (test period):")
    test_risk = forecast_df[forecast_df["is_test_period"] == 1]
    risk_counts = test_risk.groupby(["zone_id", "risk_level"]).size().unstack(fill_value=0)
    print(risk_counts.to_string())

    return forecast_df, ensemble, results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    forecast_df, ensemble_model, eval_results = run_forecast_pipeline("features_zone.csv")

    # Save forecast output
    out_path = "forecasts.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({os.path.getsize(out_path)/1024**2:.1f} MB)")

    # Save zone risk summary (lightweight — used by dashboard)
    risk_summary = (
        forecast_df[forecast_df["is_test_period"] == 1]
        .groupby("zone_id")
        .agg(
            avg_predicted   = ("predicted_demand", "mean"),
            peak_predicted  = ("predicted_demand", "max"),
            pct_high_risk   = ("risk_level",
                               lambda x: (x.isin(["HIGH","CRITICAL"])).mean() * 100),
            dominant_risk   = ("risk_level",
                               lambda x: x.value_counts().index[0]),
        )
        .reset_index()
        .round(3)
    )
    risk_summary.to_csv("zone_risk.csv", index=False)
    print(f"Saved: zone_risk.csv")

    # Save trained model
    joblib.dump(ensemble_model, "forecast_model.pkl")
    print(f"Saved: forecast_model.pkl")

    # Print final eval table
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    eval_df = pd.DataFrame(eval_results)
    print(eval_df.to_string(index=False))

    print("\nDone! Next step: python anomaly_detector.py")
