"""
explainability.py
=================
The "trust layer" that sits between the ML models and the dashboard.

Answers three questions for every flag:
  1. WHY did the model flag this meter?        → SHAP values
  2. HOW confident is the model?               → Confidence scorer
  3. Is this a real alert or noise?            → False positive filter + Audit log

Components:
  AnomalyExplainer     — wraps any trained model with SHAP
  ConfidenceScorer     — converts raw model scores to 0–1
  FalsePositiveFilter  — rule-based guard before alert reaches inspector
  AuditLogger          — writes every decision to a JSONL file
  run_explainability_pipeline — puts it all together for one flagged reading
"""

import shap                     # SHAP = SHapley Additive exPlanations
import numpy as np              # Numerical operations
import pandas as pd             # DataFrames
import json                     # Serialize audit records to JSON
import os                       # File path operations
from datetime import datetime   # Timestamp every audit entry
from sklearn.ensemble import IsolationForest
import joblib                   # Save / load trained models


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SHAP Explainer
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyExplainer:
    """
    Wraps a trained model with SHAP so every prediction comes
    with a human-readable reason — not just a score.

    SHAP (SHapley Additive exPlanations) comes from cooperative game theory.
    Each feature is treated as a "player" and SHAP computes how much each
    player contributed to the final prediction.

    Positive SHAP value → feature pushed the model TOWARD flagging.
    Negative SHAP value → feature pushed the model AWAY from flagging.
    """

    def __init__(self, model, feature_names: list):
        self.model        = model
        self.feature_names= feature_names

        # shap.Explainer auto-detects model type (tree, linear, kernel)
        # and picks the right SHAP algorithm automatically.
        # For IsolationForest → uses TreeExplainer under the hood.
        self.explainer = shap.Explainer(model)

    def explain(self, X: pd.DataFrame) -> dict:
        """
        Given a single row (one meter's features at one timestamp),
        returns a dict of { feature_name: shap_value } sorted by impact.

        Only returns top 5 features — keeps output readable for inspectors.
        """

        # shap_values shape: (n_samples, n_features)
        shap_values = self.explainer(X)

        # Take first row (.values is the raw numpy array)
        values = shap_values.values[0]

        # Zip feature names with SHAP values
        explanation = dict(zip(self.feature_names, values))

        # Sort by absolute impact — abs() because -0.9 is as important as +0.9
        sorted_explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Return only top 5 most influential features
        return dict(list(sorted_explanation.items())[:5])

    def human_readable(self, shap_dict: dict) -> str:
        """
        Converts a SHAP dict into a plain-English sentence for the dashboard.

        Example output:
          "Flag driven by: low ratio_vs_24h (-0.89), low peer_ratio (-0.82)"
        """

        parts = []
        for feature, value in shap_dict.items():
            direction = "high" if value > 0 else "low"
            parts.append(f"{direction} {feature} ({value:+.2f})")
            # :+.2f always shows + or - sign with 2 decimal places

        return "Flag driven by: " + ", ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Confidence Scorer
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceScorer:
    """
    Converts raw IsolationForest scores to a clean 0–1 confidence value.

    IsolationForest.score_samples() output:
      More negative  = more anomalous  (e.g. -0.45)
      Around 0       = borderline      (e.g. -0.02)
      More positive  = more normal     (e.g. +0.30)

    We flip and normalise this:
      confidence = clip(0.5 - raw_score, 0, 1)
      raw = -0.5  →  confidence = 1.0  (very anomalous)
      raw =  0.5  →  confidence = 0.0  (very normal)
    """

    def score(self, model, X: pd.DataFrame) -> float:
        """Returns anomaly confidence between 0.0 and 1.0."""

        raw_scores = model.score_samples(X)   # shape (n_samples,)
        raw        = raw_scores[0]            # single row

        confidence = float(np.clip(0.5 - raw, 0, 1))
        return round(confidence, 3)

    def label(self, score: float) -> str:
        """Maps float score to HIGH / MEDIUM / LOW label."""

        if score >= 0.75: return "HIGH"
        if score >= 0.50: return "MEDIUM"
        return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: False Positive Filter
# ══════════════════════════════════════════════════════════════════════════════

class FalsePositiveFilter:
    """
    Rule-based guard that runs BEFORE an alert reaches an inspector.
    Suppresses unreliable flags while logging every suppression.

    Why rule-based instead of ML?
      - Rules are readable and auditable by BESCOM engineers
      - No training data needed for the filter itself
      - Easy to add / remove / update rules
      - Traceable: every suppression has a named reason

    Rules:
      1. Confidence below threshold     → not confident enough to alert
      2. Only 1 consecutive flag        → likely noise, not a pattern
      3. Known maintenance window       → expected outage, not suspicious
      4. Weak SHAP signal               → model confused about why it flagged
    """

    def __init__(self,
                 confidence_threshold: float = 0.55,
                 min_consecutive_flags: int = 2,
                 known_maintenance_slots: list = None):

        self.confidence_threshold   = confidence_threshold
        self.min_consecutive_flags  = min_consecutive_flags
        self.known_maintenance_slots= known_maintenance_slots or []

        # Tracks consecutive flag count per meter
        # Key: meter_id, Value: int count
        self._flag_memory: dict = {}

    def should_suppress(self,
                         meter_id: str,
                         timestamp: datetime,
                         confidence: float,
                         shap_dict: dict) -> tuple:
        """
        Returns (suppress: bool, reason: str)

        suppress=True  → log but do NOT send to inspector
        suppress=False → send to inspector alert queue

        The reason string is stored in the audit log — nothing is silent.
        """

        # ── Rule 1: Low confidence ─────────────────────────────────────────
        if confidence < self.confidence_threshold:
            return True, (
                f"Confidence {confidence:.2f} below threshold "
                f"{self.confidence_threshold}"
            )

        # ── Rule 2: Known maintenance window ──────────────────────────────
        date_str = timestamp.strftime("%Y-%m-%d")
        if (meter_id, date_str) in self.known_maintenance_slots:
            return True, (
                f"Meter {meter_id} in scheduled maintenance on {date_str}"
            )

        # ── Rule 3: Consecutive flags check ───────────────────────────────
        # A single spike could be a data error, kettle, or power surge.
        # Require N flags in a row before alerting.
        count = self._flag_memory.get(meter_id, 0) + 1
        self._flag_memory[meter_id] = count

        if count < self.min_consecutive_flags:
            return True, (
                f"Only {count}/{self.min_consecutive_flags} "
                f"consecutive flags — waiting for pattern"
            )

        # ── Rule 4: Weak SHAP signal ───────────────────────────────────────
        # If no single feature dominates the SHAP explanation,
        # the model is confused about why it flagged — suppress.
        top_shap = max(abs(v) for v in shap_dict.values()) if shap_dict else 0
        if top_shap < 0.1:
            return True, (
                f"Top SHAP value {top_shap:.3f} too small — weak signal"
            )

        # ── Passed all rules → confirmed alert ────────────────────────────
        self._flag_memory[meter_id] = 0   # Reset streak after confirmed alert
        return False, "Passed all filter rules"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Audit Logger
# ══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Writes every model decision — confirmed or suppressed — to a JSONL file.

    WHY JSONL (JSON Lines) format?
      - One JSON object per line → easy to append without loading whole file
      - grep-friendly: grep "THEFT" audit_log.jsonl
      - Survives partial writes (no corruption if process is killed mid-write)
      - Works with pandas: pd.read_json("audit_log.jsonl", lines=True)
      - Easily queryable by date, meter, decision type

    Every record contains:
      timestamp, meter_id, confidence, confidence_label,
      top_shap_features, human_explanation,
      suppressed, suppress_reason, logged_at
    """

    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = log_path

        # Create empty file if it doesn't exist yet
        if not os.path.exists(log_path):
            open(log_path, "w").close()

    def log(self,
            meter_id: str,
            timestamp: datetime,
            confidence: float,
            confidence_label: str,
            shap_dict: dict,
            human_reason: str,
            suppressed: bool,
            suppress_reason: str):
        """Appends one audit record to the JSONL file."""

        record = {
            "timestamp":         timestamp.isoformat(),
            "meter_id":          meter_id,
            "confidence":        confidence,
            "confidence_label":  confidence_label,     # "HIGH" / "MEDIUM" / "LOW"
            "top_shap_features": shap_dict,            # e.g. {"ratio_vs_24h": -0.89}
            "human_explanation": human_reason,         # plain English for inspector
            "suppressed":        suppressed,           # True = FP filter blocked it
            "suppress_reason":   suppress_reason,      # why it was blocked
            "logged_at":         datetime.now().isoformat(),
        }

        # "a" = append mode — never overwrites existing entries
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_recent(self, n: int = 50) -> list:
        """
        Returns the N most recent audit records as a list of dicts.
        Used by the dashboard's audit trail tab.
        """

        with open(self.log_path, "r") as f:
            lines = f.readlines()

        # Parse each line as JSON, skip blank lines
        records = [json.loads(line) for line in lines if line.strip()]

        # Return last N (most recent are at the bottom)
        return records[-n:]

    def to_dataframe(self) -> pd.DataFrame:
        """Loads the full audit log into a pandas DataFrame for analysis."""

        records = self.get_recent(n=999999)   # Load all
        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Main pipeline — put it all together
# ══════════════════════════════════════════════════════════════════════════════

def run_explainability_pipeline(model,
                                 X_row: pd.DataFrame,
                                 meter_id: str,
                                 timestamp: datetime,
                                 feature_names: list,
                                 log_path: str = "audit_log.jsonl") -> dict:
    """
    Single function to call for each flagged meter reading.
    Runs all 4 stages and returns a structured result dict
    that the dashboard renders directly.

    Args:
        model         : trained IsolationForest (or any sklearn model)
        X_row         : single-row DataFrame with feature values
        meter_id      : e.g. "MTR_0007"
        timestamp     : datetime of the reading
        feature_names : list of column names matching X_row
        log_path      : path to audit JSONL file

    Returns dict with:
        meter_id, timestamp, confidence, confidence_label,
        explanation, top_features, alert_sent, suppress_reason
    """

    # ── Stage 1: SHAP explanation ──────────────────────────────────────────
    explainer    = AnomalyExplainer(model, feature_names)
    shap_dict    = explainer.explain(X_row)
    human_reason = explainer.human_readable(shap_dict)

    # ── Stage 2: Confidence score ──────────────────────────────────────────
    scorer           = ConfidenceScorer()
    confidence       = scorer.score(model, X_row)
    confidence_label = scorer.label(confidence)

    # ── Stage 3: False positive filter ────────────────────────────────────
    fp_filter          = FalsePositiveFilter(
        confidence_threshold  = 0.55,
        min_consecutive_flags = 2
    )
    suppressed, suppress_reason = fp_filter.should_suppress(
        meter_id, timestamp, confidence, shap_dict
    )

    # ── Stage 4: Audit log ─────────────────────────────────────────────────
    logger = AuditLogger(log_path=log_path)
    logger.log(
        meter_id         = meter_id,
        timestamp        = timestamp,
        confidence       = confidence,
        confidence_label = confidence_label,
        shap_dict        = shap_dict,
        human_reason     = human_reason,
        suppressed       = suppressed,
        suppress_reason  = suppress_reason,
    )

    # ── Return structured result for dashboard ─────────────────────────────
    return {
        "meter_id":         meter_id,
        "timestamp":        timestamp.isoformat(),
        "confidence":       confidence,
        "confidence_label": confidence_label,
        "explanation":      human_reason,
        "top_features":     shap_dict,
        "alert_sent":       not suppressed,   # True = inspector sees this
        "suppress_reason":  suppress_reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Entry point — demo run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Demo: loads the trained Isolation Forest and runs one flagged reading
    through the full explainability pipeline.

    Run: python explainability.py
    Requires: anomaly_model_if.pkl and features_meter.csv to exist.
    """

    print("BESCOM Explainability Pipeline — Demo Run")
    print("=" * 60)

    # Load saved Isolation Forest model
    if not os.path.exists("anomaly_model_if.pkl"):
        print("anomaly_model_if.pkl not found. Run anomaly_detector.py first.")
        exit(1)

    if_model = joblib.load("anomaly_model_if.pkl")
    print("Loaded: anomaly_model_if.pkl")

    # Load a sample flagged reading from features_meter.csv
    features_df = pd.read_csv("features_meter.csv", parse_dates=["timestamp"])

    feature_cols = [
        "consumption_kwh", "zscore_24h", "zscore_7d",
        "ratio_vs_24h", "ratio_vs_7d", "peer_ratio", "peer_zscore",
        "roll_mean_24h", "roll_std_24h", "cv_24h",
        "delta_1h", "abs_delta_1h", "pct_change_1step",
        "lag_24h", "lag_7d",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    # Pick a row that was actually flagged as anomalous
    flagged = features_df[features_df["is_anomaly"] == 1].head(1)

    if len(flagged) == 0:
        print("No flagged rows found in features_meter.csv")
        exit(1)

    sample      = flagged.iloc[0]
    X_row       = flagged[feature_cols]
    meter_id    = sample["meter_id"]
    timestamp   = sample["timestamp"].to_pydatetime()

    print(f"\nRunning explainability pipeline on: {meter_id} @ {timestamp}")
    print(f"Anomaly type (ground truth): {sample.get('anomaly_type', 'unknown')}")

    # Run the pipeline
    result = run_explainability_pipeline(
        model         = if_model.model,      # the raw sklearn model inside wrapper
        X_row         = X_row,
        meter_id      = meter_id,
        timestamp     = timestamp,
        feature_names = feature_cols,
    )

    # Print result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Meter:            {result['meter_id']}")
    print(f"Confidence:       {result['confidence']} ({result['confidence_label']})")
    print(f"Explanation:      {result['explanation']}")
    print(f"Alert sent:       {result['alert_sent']}")
    print(f"Filter reason:    {result['suppress_reason']}")
    print(f"\nTop SHAP features:")
    for feat, val in result["top_features"].items():
        bar = "█" * int(abs(val) * 20)
        direction = "→ anomaly" if val > 0 else "→ normal"
        print(f"  {feat:30s} {val:+.3f}  {bar}  ({direction})")

    print(f"\nAudit log written to: audit_log.jsonl")
    print("\nDone!")
