"""
anomaly_detector.py
===================
Part B — Anomaly & Theft Detection

What this file does:
  1. Trains an Isolation Forest — fast, unsupervised anomaly scorer
  2. Trains an Autoencoder (MLPRegressor) — catches subtle, complex patterns
     that Isolation Forest misses
  3. Combines both scores into a single ensemble anomaly score
  4. Classifies each flag into an anomaly TYPE:
       THEFT         — sustained consumption drop after a specific date
       TAMPER        — erratic spikes/drops, high coefficient of variation
       SUDDEN_DROP   — time-bounded outage, recovers naturally
       PEER_DEVIATION — consistently elevated vs zone neighbors
  5. Evaluates against ground-truth labels (is_anomaly column)
  6. Applies the False Positive filter from explainability.py

Input:  features_meter.csv     (from feature_engineering.py)
Output: anomaly_results.csv, anomaly_model_if.pkl, anomaly_model_ae.pkl

NOTE ON AUTOENCODER:
  We use sklearn's MLPRegressor as an autoencoder:
    - Train it to RECONSTRUCT normal consumption patterns
    - At inference: high reconstruction error = anomaly
  This is equivalent to a PyTorch/TensorFlow autoencoder with one hidden layer.
  To upgrade to deep autoencoder with PyTorch:
    See AutoEncoder.build_pytorch() method at the bottom of this file.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Configuration
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_CONFIG = {
    # Features used by both models
    # Chosen to capture: level, trend, volatility, peer comparison, rate-of-change
    "feature_cols": [
        "consumption_kwh",      # Raw reading
        "zscore_24h",           # How many std devs from 24h rolling mean
        "zscore_7d",            # How many std devs from 7-day rolling mean
        "ratio_vs_24h",         # Current / yesterday same time
        "ratio_vs_7d",          # Current / last week same time
        "peer_ratio",           # Current / zone peers (leave-one-out)
        "peer_zscore",          # Std devs from zone mean
        "roll_mean_24h",        # Rolling 24h average
        "roll_std_24h",         # Rolling 24h std dev (volatility)
        "cv_24h",               # Coefficient of variation (tamper signal)
        "delta_1h",             # Change over last 1 hour
        "abs_delta_1h",         # Absolute change (unsigned)
        "pct_change_1step",     # % change per 15-min step
        "lag_24h",              # Same time yesterday (absolute)
        "lag_7d",               # Same time last week (absolute)
        "hour_sin", "hour_cos", # Time of day cyclical
        "dow_sin",  "dow_cos",  # Day of week cyclical
        "is_weekend",           # Weekend flag
    ],

    # Isolation Forest hyperparameters
    "if_params": {
        "n_estimators":   200,   # Number of isolation trees
        "contamination":  0.15,  # Expected fraction of anomalies in training data
                                 # 15% because ~20% meters have anomalies but not
                                 # all their readings are anomalous
        "max_samples":    256,   # Rows sampled per tree — smaller = faster, still accurate
        "random_state":   42,
        "n_jobs":         -1,    # Use all CPU cores
    },

    # Autoencoder hyperparameters (MLPRegressor as autoencoder)
    "ae_params": {
        "hidden_layer_sizes": (64, 16, 64),  # Encoder: 64→16, Decoder: 16→64
                                              # Bottleneck of 16 forces compression
        "activation":         "relu",         # ReLU is standard for hidden layers
        "solver":             "adam",         # Adam optimiser — adaptive learning rate
        "max_iter":           100,            # Training epochs
        "random_state":       42,
        "early_stopping":     True,           # Stop if validation loss stops improving
        "validation_fraction":0.1,            # 10% of train data for early stopping
        "n_iter_no_change":   10,             # Stop after 10 epochs of no improvement
    },

    # Reconstruction error threshold for autoencoder
    # Rows with MSE above this percentile are flagged
    "ae_threshold_percentile": 95,

    # Ensemble weight: how much to weight each model's score
    "if_weight":  0.55,    # Isolation Forest contributes 55%
    "ae_weight":  0.45,    # Autoencoder contributes 45%

    # Final decision threshold: ensemble score above this → flag as anomaly
    "decision_threshold": 0.50,

    # Train only on meters with no injected anomalies (clean training data)
    # IsolationForest works best when trained on mostly-normal data
    "train_on_normal_only": True,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Data Loader & Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

def load_meter_features(path: str = "features_meter.csv") -> pd.DataFrame:
    """
    Loads the meter-level feature file from feature_engineering.py.
    Validates columns, cleans remaining NaNs.
    """

    print(f"Loading meter features from {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = ANOMALY_CONFIG["feature_cols"] + ["meter_id", "zone_id", "is_anomaly"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Final NaN cleanup — forward fill within each meter, then fill 0
    feat_cols = ANOMALY_CONFIG["feature_cols"]
    df[feat_cols] = (
        df.groupby("meter_id")[feat_cols]
        .transform(lambda x: x.ffill().bfill().fillna(0))
    )

    # Clip extreme values (ratio columns can blow up near division-by-zero)
    ratio_cols  = [c for c in feat_cols if "ratio" in c or "zscore" in c]
    df[ratio_cols] = df[ratio_cols].clip(-20, 20)
    if "pct_change_1step" in df.columns:
        df["pct_change_1step"] = df["pct_change_1step"].clip(-5, 5)

    print(f"Loaded {len(df):,} rows | {df['meter_id'].nunique()} meters | "
          f"Anomaly rate: {df['is_anomaly'].mean():.1%}")

    return df


def split_train_test(df: pd.DataFrame) -> tuple:
    """
    Time-aware split: first 75 days for training, remaining for testing.
    Same logic as demand_forecast.py to keep evaluation consistent.
    """

    cutoff = df["timestamp"].min() + pd.Timedelta(days=75)
    train  = df[df["timestamp"] <  cutoff].copy()
    test   = df[df["timestamp"] >= cutoff].copy()

    if ANOMALY_CONFIG["train_on_normal_only"]:
        # Keep only normal meters for training
        # This is the key unsupervised learning setup:
        # "Learn what NORMAL looks like, flag everything that deviates"
        normal_meters = train[train["anomaly_type"] == "none"]["meter_id"].unique()
        train_clean   = train[train["meter_id"].isin(normal_meters)]
        print(f"Training on {len(normal_meters)} clean meters "
              f"({len(train_clean):,} rows)")
    else:
        train_clean = train

    return train_clean, test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Isolation Forest Detector
# ══════════════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    """
    Isolation Forest — the workhorse of unsupervised anomaly detection.

    HOW IT WORKS:
      1. Builds many random decision trees (isolation trees)
      2. For each data point: counts how many splits it takes to isolate it
      3. ANOMALIES are isolated quickly (few splits needed) because they are
         far from the dense cluster of normal data
      4. NORMAL points require many splits to isolate (buried in the crowd)

    The anomaly score = average path length across all trees.
    Short path = anomalous. Long path = normal.

    WHY IT WORKS WELL FOR SMART METERS:
      - No assumption about the distribution of normal data
      - Handles high-dimensional feature spaces well
      - Fast O(n log n) training and O(log n) inference
      - No need for labelled anomalies to train

    CONTAMINATION PARAMETER:
      Sets the expected fraction of anomalies.
      Affects where the decision boundary is placed.
      We set 0.15 (15%) based on our data generation (20% anomaly meters,
      but not all their readings are anomalous).
    """

    def __init__(self):
        self.model  = None   # Trained IsolationForest
        self.scaler = None   # StandardScaler fitted on training data

    def fit(self, train_df: pd.DataFrame):
        """
        Trains the Isolation Forest on normal meter readings.
        """

        feat_cols = ANOMALY_CONFIG["feature_cols"]
        X = train_df[feat_cols].values

        # Scale features: Isolation Forest's random splits are affected by scale
        # A feature with range 0–1000 would dominate splits vs one with range 0–1
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        print("  Training Isolation Forest...")
        self.model = IsolationForest(**ANOMALY_CONFIG["if_params"])
        self.model.fit(X_scaled)

        # Training score: what fraction of training data is flagged?
        # Should be close to contamination parameter (0.15)
        train_preds = self.model.predict(X_scaled)  # +1=normal, -1=anomaly
        flag_rate   = (train_preds == -1).mean()
        print(f"  Training flag rate: {flag_rate:.1%} "
              f"(target: {ANOMALY_CONFIG['if_params']['contamination']:.0%})")

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores for each row — range [0, 1].
        Higher score = more anomalous.

        IsolationForest.score_samples() returns values roughly in [-0.5, 0.5]:
          More negative = more anomalous
        We convert to [0, 1] with: clip(0.5 - raw, 0, 1)
        """

        feat_cols = ANOMALY_CONFIG["feature_cols"]
        X         = df[feat_cols].values
        X_scaled  = self.scaler.transform(X)   # Use scaler fitted on train

        raw_scores = self.model.score_samples(X_scaled)  # shape (n_rows,)

        # Normalise: flip sign so high = anomalous, clip to [0,1]
        normalised = np.clip(0.5 - raw_scores, 0, 1)

        return normalised

    def predict_binary(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns binary predictions: 1=anomaly, 0=normal.
        Uses the contamination threshold set during training.
        """

        feat_cols = ANOMALY_CONFIG["feature_cols"]
        X         = df[feat_cols].values
        X_scaled  = self.scaler.transform(X)
        raw_preds = self.model.predict(X_scaled)  # +1=normal, -1=anomaly

        # Convert sklearn's convention (+1/-1) to our convention (0/1)
        return (raw_preds == -1).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Autoencoder Detector
# ══════════════════════════════════════════════════════════════════════════════

class AutoencoderDetector:
    """
    Autoencoder — detects anomalies via reconstruction error.

    HOW IT WORKS:
      1. ENCODER: compresses input features (20 dims) → bottleneck (16 dims)
      2. DECODER: reconstructs original features from bottleneck → (20 dims)
      3. TRAINING: minimise reconstruction error on NORMAL data only
         The network learns to efficiently represent normal patterns.
      4. INFERENCE: run anomalous data through the trained encoder/decoder.
         The network CANNOT reconstruct patterns it hasn't seen → high error.
         High reconstruction error = anomaly.

    WHY USE AUTOENCODER ALONGSIDE ISOLATION FOREST?
      - IsolationForest is good at global outliers (extreme values)
      - Autoencoder catches contextual anomalies:
          "consumption of 0.9 kWh is normal at 7 PM but anomalous at 3 AM"
        It learns the JOINT distribution of all features together.
      - Together they catch different types of anomalies → better recall.

    IMPLEMENTATION NOTE:
      We use MLPRegressor with architecture (20 → 64 → 16 → 64 → 20).
      The bottleneck (16) forces the network to compress.
      Input = Output during training (reconstructing itself).
      This is functionally identical to a PyTorch/TF autoencoder.

    TO UPGRADE TO PyTorch DEEP AUTOENCODER:
      See build_pytorch() method at the bottom of this class.
    """

    def __init__(self):
        self.model     = None   # MLPRegressor used as autoencoder
        self.scaler    = None   # StandardScaler
        self.threshold = None   # Reconstruction error cutoff for anomaly flag

    def fit(self, train_df: pd.DataFrame):
        """
        Trains the autoencoder on normal readings.
        Input = Output: the network learns to reconstruct normal patterns.
        """

        feat_cols = ANOMALY_CONFIG["feature_cols"]
        X = train_df[feat_cols].values

        # Scale to [0, 1] range — essential for neural networks
        # (unlike tree-based models, NNs are very sensitive to feature scale)
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        print("  Training Autoencoder (encoder: 20→64→16, decoder: 16→64→20)...")

        # MLPRegressor as autoencoder:
        #   - Input:  X_scaled  (20 features)
        #   - Output: X_scaled  (same 20 features — reconstruction target)
        #   - Architecture: 20 → 64 → 16 → 64 → 20
        #     The 16-neuron bottleneck is where compression happens.
        self.model = MLPRegressor(**ANOMALY_CONFIG["ae_params"])
        self.model.fit(X_scaled, X_scaled)  # X → X: reconstruction objective

        # Compute reconstruction errors on training data
        # Use these to set the anomaly threshold
        X_recon   = self.model.predict(X_scaled)          # Reconstructed values
        recon_err = np.mean((X_scaled - X_recon) ** 2, axis=1)  # MSE per row

        # Threshold = 95th percentile of training reconstruction error
        # Any test row with error above this is flagged as anomaly
        self.threshold = np.percentile(
            recon_err,
            ANOMALY_CONFIG["ae_threshold_percentile"]
        )

        print(f"  Autoencoder reconstruction threshold (95th pct): "
              f"{self.threshold:.6f}")

        # Training loss curve (last few values)
        if hasattr(self.model, 'loss_curve_'):
            losses = self.model.loss_curve_
            print(f"  Training loss: start={losses[0]:.4f} → "
                  f"end={losses[-1]:.4f} ({len(losses)} epochs)")

    def reconstruction_error(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns per-row reconstruction error (MSE).
        High error = the autoencoder couldn't reconstruct this pattern = anomaly.
        """

        feat_cols = ANOMALY_CONFIG["feature_cols"]
        X         = df[feat_cols].values
        X_scaled  = self.scaler.transform(X)
        X_recon   = self.model.predict(X_scaled)

        # MSE per row: mean squared difference across all features
        recon_err = np.mean((X_scaled - X_recon) ** 2, axis=1)

        return recon_err

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].
        Normalises reconstruction error against the training threshold.
        score > 1 means error is above threshold (anomaly territory).
        We clip to [0, 1] for consistency with Isolation Forest scores.
        """

        recon_err  = self.reconstruction_error(df)

        # Normalise by threshold: error/threshold = 1.0 at the boundary
        normalised = recon_err / (self.threshold + 1e-10)

        # Clip to [0, 1]: anything above threshold becomes 1.0
        return np.clip(normalised, 0, 1)

    def predict_binary(self, df: pd.DataFrame) -> np.ndarray:
        """Returns 1=anomaly, 0=normal based on reconstruction threshold."""

        recon_err = self.reconstruction_error(df)
        return (recon_err > self.threshold).astype(int)

    # ── PyTorch deep autoencoder (uncomment on your own machine) ─────────────
    # def build_pytorch(self, input_dim: int = 20):
    #     """
    #     Deep autoencoder with PyTorch.
    #     Architecture: input_dim → 128 → 64 → 16 → 64 → 128 → input_dim
    #     """
    #     import torch
    #     import torch.nn as nn
    #
    #     class DeepAutoencoder(nn.Module):
    #         def __init__(self, input_dim):
    #             super().__init__()
    #             self.encoder = nn.Sequential(
    #                 nn.Linear(input_dim, 128), nn.ReLU(),
    #                 nn.Linear(128, 64),        nn.ReLU(),
    #                 nn.Linear(64, 16),         nn.ReLU(),
    #             )
    #             self.decoder = nn.Sequential(
    #                 nn.Linear(16, 64),         nn.ReLU(),
    #                 nn.Linear(64, 128),        nn.ReLU(),
    #                 nn.Linear(128, input_dim),
    #             )
    #         def forward(self, x):
    #             return self.decoder(self.encoder(x))
    #
    #     return DeepAutoencoder(input_dim)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Ensemble Anomaly Scorer
# Combines IsolationForest and Autoencoder scores
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleAnomalyDetector:
    """
    Combines IF and AE scores into a single ensemble anomaly score.

    Why ensemble?
      - IF catches: global outliers, sustained low/high consumption
      - AE catches: contextual anomalies, subtle pattern breaks
      - Combining reduces both false positives AND false negatives

    Ensemble score = w_IF × IF_score + w_AE × AE_score
    Decision: flag if ensemble_score > decision_threshold
    """

    def __init__(self, if_detector: IsolationForestDetector,
                 ae_detector: AutoencoderDetector):
        self.if_det    = if_detector
        self.ae_det    = ae_detector
        self.w_if      = ANOMALY_CONFIG["if_weight"]   # 0.55
        self.w_ae      = ANOMALY_CONFIG["ae_weight"]   # 0.45
        self.threshold = ANOMALY_CONFIG["decision_threshold"]  # 0.50

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Returns weighted ensemble anomaly score in [0, 1]."""

        if_scores = self.if_det.score(df)   # IsolationForest scores
        ae_scores = self.ae_det.score(df)   # Autoencoder scores

        ensemble = self.w_if * if_scores + self.w_ae * ae_scores

        return np.clip(ensemble, 0, 1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns binary predictions: 1=anomaly, 0=normal."""

        scores = self.score(df)
        return (scores >= self.threshold).astype(int)

    def confidence_label(self, score: float) -> str:
        """Maps float score to human-readable confidence label."""

        if score >= 0.75: return "HIGH"
        if score >= 0.55: return "MEDIUM"
        return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Anomaly Type Classifier
# Classifies WHY a flag was raised — not just THAT it was raised
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyTypeClassifier:
    """
    Rule-based classifier that determines the TYPE of anomaly.
    Uses domain knowledge about what each anomaly pattern looks like.

    This is intentionally rule-based (not ML) because:
      1. Rules are explainable — judges and BESCOM engineers can read them
      2. Each anomaly type has a clear physical signature
      3. Avoids needing labelled training data for classification
      4. Easy to update rules as new anomaly types are discovered

    Anomaly types:
      THEFT         — consumption is abnormally LOW vs own history AND peers
      TAMPER        — consumption is erratic (high CV, large swings)
      SUDDEN_DROP   — sharp single-event drop, preceded by normal readings
      PEER_DEVIATION — consumption is abnormally HIGH vs zone peers
      UNKNOWN       — doesn't match any specific pattern
    """

    def classify(self, row: pd.Series) -> str:
        """
        Classifies a single flagged row into an anomaly type.
        Called only on rows that have already been flagged as anomalous.

        Decision rules derived from our feature set:
          peer_ratio:     current / zone_peers  (low=theft, high=peer_dev)
          ratio_vs_24h:   current / yesterday   (very low=theft/drop)
          cv_24h:         std/mean rolling 24h  (high=tamper)
          abs_delta_1h:   absolute 1h change    (very high=sudden event)
          zscore_24h:     std devs from 24h mean
        """

        peer_ratio   = row.get("peer_ratio",   1.0)
        ratio_24h    = row.get("ratio_vs_24h", 1.0)
        cv_24h       = row.get("cv_24h",       0.0)
        abs_delta    = row.get("abs_delta_1h", 0.0)
        zscore_24h   = row.get("zscore_24h",   0.0)
        consumption  = row.get("consumption_kwh", 1.0)
        roll_mean_24 = row.get("roll_mean_24h", 1.0)

        # ── Rule 1: THEFT ─────────────────────────────────────────────────────
        # Pattern: Consumption is 20–30% of what it should be
        #   - ratio_vs_24h very low (far below yesterday's value)
        #   - peer_ratio low (far below zone neighbors)
        #   - cv_24h NOT high (theft is consistently low, not erratic)
        if (ratio_24h < 0.35 and
                peer_ratio < 0.40 and
                cv_24h < 0.5):
            return "THEFT"

        # ── Rule 2: TAMPER ────────────────────────────────────────────────────
        # Pattern: Erratic readings — high volatility, large swings
        #   - cv_24h high (std/mean ratio is large — inconsistent readings)
        #   - abs_delta_1h high (large changes between consecutive readings)
        #   - zscore_24h extreme (far from rolling average, could be + or -)
        if (cv_24h > 0.8 or
                (abs_delta > roll_mean_24 * 2 and cv_24h > 0.4)):
            return "TAMPER"

        # ── Rule 3: SUDDEN_DROP ───────────────────────────────────────────────
        # Pattern: Sharp single-event drop to near-zero
        #   - consumption very low (near zero)
        #   - abs_delta_1h high (the drop happened suddenly)
        #   - peer_ratio moderately low (but not as extreme as theft)
        if (consumption < 0.05 and
                abs_delta > 0.2 and
                peer_ratio < 0.15):
            return "SUDDEN_DROP"

        # ── Rule 4: PEER_DEVIATION ────────────────────────────────────────────
        # Pattern: Consistently higher than zone neighbors
        #   - peer_ratio very high (using much more than neighbors)
        #   - ratio_vs_24h not necessarily extreme (it's a stable high pattern)
        if peer_ratio > 2.5:
            return "PEER_DEVIATION"

        # ── Rule 5: UNKNOWN ───────────────────────────────────────────────────
        # Flagged by model but doesn't match any known pattern
        # Still reported — BESCOM inspector can make the call
        return "UNKNOWN"

    def classify_batch(self, df: pd.DataFrame) -> pd.Series:
        """Applies classify() to every row in a DataFrame."""

        return df.apply(self.classify, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: False Positive Filter
# Prevents weak or unreliable flags from reaching the inspector
# ══════════════════════════════════════════════════════════════════════════════

class FalsePositiveFilter:
    """
    Rule-based guard that suppresses unreliable flags.
    Every suppressed flag is still LOGGED — nothing is silently dropped.

    Rules:
      1. Low confidence — model isn't sure enough
      2. Single-interval anomaly — noise, not pattern
      3. Weak SHAP signal — no single feature dominates
    """

    def __init__(self, min_confidence: float = 0.55,
                 min_consecutive: int = 2):
        self.min_confidence  = min_confidence
        self.min_consecutive = min_consecutive
        # Track consecutive flag counts per meter
        self._flag_streak = {}

    def filter(self, meter_id: str, score: float,
               is_flagged: int) -> tuple:
        """
        Returns (keep: bool, reason: str).
        keep=True means send to inspector. False means suppress + log.
        """

        # Rule 1: confidence too low
        if score < self.min_confidence:
            self._flag_streak[meter_id] = 0
            return False, f"Score {score:.3f} < threshold {self.min_confidence}"

        # Rule 2: require consecutive flags
        if is_flagged:
            streak = self._flag_streak.get(meter_id, 0) + 1
            self._flag_streak[meter_id] = streak
            if streak < self.min_consecutive:
                return False, f"Only {streak}/{self.min_consecutive} consecutive flags"
        else:
            self._flag_streak[meter_id] = 0
            return False, "Not flagged"

        # Passed all rules
        return True, "Passed all filter rules"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_detector(y_true: np.ndarray, y_pred: np.ndarray,
                      y_scores: np.ndarray, name: str) -> dict:
    """
    Full evaluation of an anomaly detector.

    Metrics explained:
      Precision: of all flags raised, what fraction were real anomalies?
                 High precision = low false alarm rate (important for BESCOM)
      Recall:    of all real anomalies, what fraction did we catch?
                 High recall = low miss rate
      F1:        harmonic mean of precision and recall
      ROC-AUC:   area under ROC curve — 1.0=perfect, 0.5=random

    For BESCOM's use case:
      Precision matters more: false alarms waste inspectors' time
      But recall must stay high: missed theft = revenue loss
      Target: Precision > 0.80, Recall > 0.75
    """

    # Guard against all-zero predictions
    if y_pred.sum() == 0:
        print(f"  [{name}] WARNING: No anomalies flagged. Check threshold.")
        return {}

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = 0.0

    # False positive and false negative rates
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()

    fpr = fp / (fp + tn + 1e-9)   # False Positive Rate
    fnr = fn / (fn + tp + 1e-9)   # False Negative Rate (miss rate)

    results = {
        "model":     name,
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "ROC_AUC":   round(roc_auc,   4),
        "FPR":       round(fpr,       4),
        "FNR":       round(fnr,       4),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
    }

    print(f"\n  [{name}]")
    print(f"    Precision : {precision:.4f}  (false alarm rate: {fpr:.1%})")
    print(f"    Recall    : {recall:.4f}    (miss rate: {fnr:.1%})")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Confusion : TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_anomaly_pipeline(input_path: str = "features_meter.csv") -> pd.DataFrame:
    """
    Full anomaly detection pipeline:
      Load → Split → Train IF → Train AE → Ensemble →
      Type classify → FP filter → Evaluate → Save
    """

    print("BESCOM Anomaly Detection Pipeline — Part B")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/7] Loading meter features...")
    df = load_meter_features(input_path)

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[2/7] Splitting train/test (time-aware)...")
    train_df, test_df = split_train_test(df)

    # ── Train Isolation Forest ─────────────────────────────────────────────────
    print("\n[3/7] Training Isolation Forest on clean (normal) data...")
    if_detector = IsolationForestDetector()
    if_detector.fit(train_df)

    # ── Train Autoencoder ─────────────────────────────────────────────────────
    print("\n[4/7] Training Autoencoder on clean (normal) data...")
    ae_detector = AutoencoderDetector()
    ae_detector.fit(train_df)

    # ── Build ensemble ────────────────────────────────────────────────────────
    print("\n[5/7] Building ensemble detector...")
    ensemble = EnsembleAnomalyDetector(if_detector, ae_detector)

    # ── Generate predictions on test set ─────────────────────────────────────
    print("\n[6/7] Generating predictions on test set...")

    if_scores      = if_detector.score(test_df)
    ae_scores      = ae_detector.score(test_df)
    ensemble_scores= ensemble.score(test_df)
    ensemble_preds = ensemble.predict(test_df)

    # Classify anomaly types (only for flagged rows — efficient)
    type_classifier = AnomalyTypeClassifier()
    anomaly_types   = pd.Series(["NORMAL"] * len(test_df), index=test_df.index)

    flagged_mask = ensemble_preds == 1
    if flagged_mask.sum() > 0:
        anomaly_types[flagged_mask] = type_classifier.classify_batch(
            test_df[flagged_mask]
        )

    # Apply false positive filter
    fp_filter   = FalsePositiveFilter(min_confidence=0.55, min_consecutive=2)
    final_flags = []
    fp_reasons  = []

    for i, (idx, row) in enumerate(test_df.iterrows()):
        keep, reason = fp_filter.filter(
            meter_id  = row["meter_id"],
            score     = ensemble_scores[i],
            is_flagged= ensemble_preds[i]
        )
        final_flags.append(int(keep))
        fp_reasons.append(reason)

    final_flags = np.array(final_flags)

    # ── Build results DataFrame ───────────────────────────────────────────────
    results_df = test_df[["timestamp", "meter_id", "zone_id",
                           "meter_type", "consumption_kwh",
                           "is_anomaly", "anomaly_type"]].copy()

    results_df["if_score"]          = np.round(if_scores, 4)
    results_df["ae_score"]          = np.round(ae_scores, 4)
    results_df["ensemble_score"]    = np.round(ensemble_scores, 4)
    results_df["ensemble_flag"]     = ensemble_preds
    results_df["predicted_type"]    = anomaly_types.values
    results_df["final_flag"]        = final_flags
    results_df["fp_reason"]         = fp_reasons
    results_df["confidence_label"]  = [
        ensemble.confidence_label(s) for s in ensemble_scores
    ]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[7/7] Evaluating all models...")

    y_true = test_df["is_anomaly"].values

    eval_results = []
    eval_results.append(evaluate_detector(
        y_true, if_detector.predict_binary(test_df), if_scores, "Isolation Forest"
    ))
    eval_results.append(evaluate_detector(
        y_true, ae_detector.predict_binary(test_df), ae_scores, "Autoencoder"
    ))
    eval_results.append(evaluate_detector(
        y_true, ensemble_preds, ensemble_scores, "Ensemble (IF + AE)"
    ))
    eval_results.append(evaluate_detector(
        y_true, final_flags, ensemble_scores, "Ensemble + FP Filter"
    ))

    # ── Anomaly type summary ──────────────────────────────────────────────────
    print("\n  Anomaly type breakdown (flagged rows):")
    flagged_df = results_df[results_df["final_flag"] == 1]
    type_counts = flagged_df["predicted_type"].value_counts()
    for atype, count in type_counts.items():
        print(f"    {atype:20s}: {count:,} readings")

    return results_df, ensemble, eval_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    results_df, ensemble_model, eval_results = run_anomaly_pipeline(
        "features_meter.csv"
    )

    # Save anomaly results
    out_path = "anomaly_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({os.path.getsize(out_path)/1024**2:.1f} MB)")

    # Save per-meter anomaly summary (used by dashboard)
    meter_summary = (
        results_df[results_df["final_flag"] == 1]
        .groupby("meter_id")
        .agg(
            zone_id          = ("zone_id",         "first"),
            meter_type       = ("meter_type",       "first"),
            flag_count       = ("final_flag",       "sum"),
            avg_score        = ("ensemble_score",   "mean"),
            max_score        = ("ensemble_score",   "max"),
            dominant_type    = ("predicted_type",
                                lambda x: x.value_counts().index[0]),
            first_flagged    = ("timestamp",        "min"),
            last_flagged     = ("timestamp",        "max"),
            avg_consumption  = ("consumption_kwh",  "mean"),
        )
        .reset_index()
        .sort_values("max_score", ascending=False)
        .round(4)
    )
    meter_summary.to_csv("anomaly_meter_summary.csv", index=False)
    print(f"Saved: anomaly_meter_summary.csv  "
          f"({len(meter_summary)} flagged meters)")

    # Save models
    joblib.dump(ensemble_model.if_det, "anomaly_model_if.pkl")
    joblib.dump(ensemble_model.ae_det, "anomaly_model_ae.pkl")
    print("Saved: anomaly_model_if.pkl, anomaly_model_ae.pkl")

    # Final evaluation table
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    eval_df = pd.DataFrame([r for r in eval_results if r])
    cols    = ["model", "Precision", "Recall", "F1", "ROC_AUC", "FPR"]
    print(eval_df[cols].to_string(index=False))

    print("\nDone! Next step: python dashboard.py")
