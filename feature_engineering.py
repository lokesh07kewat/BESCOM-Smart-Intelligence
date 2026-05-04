"""
feature_engineering.py
======================
Transforms raw smart meter readings into rich features for ML models.

Why this matters:
  Raw data: timestamp, meter_id, consumption_kwh
  After this file: 25+ features capturing time patterns, rolling statistics,
  lag comparisons, peer-group deviation, and gap indicators.

Input:  smart_meter_data.csv   (from data_generator.py)
Output: features.csv           (ready for demand_forecast.py and anomaly_detector.py)

Feature groups we build:
  A. Time features          — hour, day, month, is_weekend, is_holiday
  B. Lag features           — consumption 1h ago, 24h ago, 1-week ago
  C. Rolling statistics     — mean, std, min, max over past windows
  D. Peer-group deviation   — how does this meter compare to zone neighbors?
  E. Rate-of-change         — how fast is consumption changing?
  F. Gap / data quality     — is this meter missing readings?
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Time Features
# Extract rich calendar information from the timestamp column.
# These are free features — no historical data needed, just the timestamp.
# ══════════════════════════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar-based features derived purely from the timestamp.

    These capture cyclical patterns — the model needs to know that
    hour 23 and hour 0 are close together, not 23 apart.
    We encode hours and months as sin/cos pairs for this reason.
    """

    # Ensure timestamp column is parsed as datetime type
    # If it's already datetime, this is a no-op. If it's a string, it gets parsed.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Basic calendar extractions ────────────────────────────────────────────

    df["hour"]          = df["timestamp"].dt.hour          # 0–23
    df["day_of_week"]   = df["timestamp"].dt.dayofweek     # 0=Mon, 6=Sun
    df["day_of_month"]  = df["timestamp"].dt.day           # 1–31
    df["month"]         = df["timestamp"].dt.month         # 1–12
    df["quarter"]       = df["timestamp"].dt.quarter       # 1–4
    df["week_of_year"]  = df["timestamp"].dt.isocalendar().week.astype(int)

    # Binary flags — easy for models to use directly
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)  # 1 if Sat or Sun
    df["is_month_start"]= df["timestamp"].dt.is_month_start.astype(int)
    df["is_month_end"]  = df["timestamp"].dt.is_month_end.astype(int)

    # Time of day buckets — helps models segment the day without learning it from scratch
    # 0=night (0-5), 1=morning (6-11), 2=afternoon (12-17), 3=evening (18-23)
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # ── Cyclical encoding ──────────────────────────────────────────────────────
    # Problem: Hour 23 and Hour 0 are only 1 hour apart in reality,
    # but numerically they are 23 apart. Linear encoding breaks this.
    # Solution: Encode as (sin, cos) pair on a circle.
    #   sin(2π × hour/24) and cos(2π × hour/24)
    # Now hour 23 and hour 0 are close in 2D space. ✓

    # Hour cyclical encoding (period = 24 hours)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Day of week cyclical encoding (period = 7 days)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Month cyclical encoding (period = 12 months)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Indian public holidays (approximate) ─────────────────────────────────
    # BESCOM serves Bangalore — consumption drops sharply on major holidays
    indian_holidays_2024 = [
        "2024-01-26",   # Republic Day
        "2024-03-25",   # Holi
        "2024-03-29",   # Good Friday
    ]
    holiday_dates = pd.to_datetime(indian_holidays_2024).normalize()

    # Normalize timestamps to date-only for comparison
    df["is_holiday"] = df["timestamp"].dt.normalize().isin(holiday_dates).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Lag Features
# "What was this meter consuming at the same time yesterday / last week?"
# These are among the most powerful features for anomaly detection:
# if a meter usually uses 1.5 kWh on Tuesday evenings but today shows 0.1,
# the lag comparison catches it immediately.
# ══════════════════════════════════════════════════════════════════════════════

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lagged consumption values per meter.

    CRITICAL: We use groupby(meter_id) before applying shift().
    Without this, the lag for meter MTR_0001 at 00:00 would look at the
    last row of MTR_0000 — completely wrong cross-meter contamination.

    Lag periods (in 15-min intervals):
      4   intervals = 1 hour ago
      96  intervals = 24 hours ago (same time yesterday)
      192 intervals = 48 hours ago
      672 intervals = 7 days ago (same time last week)
    """

    # Sort first — shift() depends on order
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Group by meter so lags stay within each meter's own history
    grp = df.groupby("meter_id")["consumption_kwh"]

    # shift(n) moves values DOWN by n rows — so row i gets the value from row i-n
    # This means: "what was consumption n intervals ago for THIS meter?"

    df["lag_1h"]   = grp.shift(4)    # 4 × 15min = 1 hour ago
    df["lag_24h"]  = grp.shift(96)   # 96 × 15min = 24 hours ago
    df["lag_48h"]  = grp.shift(192)  # 192 × 15min = 48 hours ago
    df["lag_7d"]   = grp.shift(672)  # 672 × 15min = 7 days ago

    # ── Lag ratios ────────────────────────────────────────────────────────────
    # Raw lag tells you the absolute past value.
    # Ratio tells you HOW MUCH has changed — more useful for anomaly detection.
    # We add a small epsilon (1e-6) in the denominator to prevent division by zero.

    eps = 1e-6   # tiny number to prevent ZeroDivisionError

    # ratio > 1 means current consumption is HIGHER than that lag
    # ratio < 1 means current consumption is LOWER (could be theft/drop)
    df["ratio_vs_24h"] = df["consumption_kwh"] / (df["lag_24h"] + eps)
    df["ratio_vs_7d"]  = df["consumption_kwh"] / (df["lag_7d"]  + eps)

    # ── Difference features ───────────────────────────────────────────────────
    # Absolute difference — useful when the ratio might be misleading
    # (e.g. going from 0.001 to 0.002 gives ratio=2 but diff is tiny)

    df["diff_vs_24h"] = df["consumption_kwh"] - df["lag_24h"]
    df["diff_vs_7d"]  = df["consumption_kwh"] - df["lag_7d"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Rolling Statistics
# "What has this meter's average / volatility been over the past N hours?"
# Rolling features smooth out noise and reveal sustained patterns vs spikes.
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling window statistics per meter.

    Windows (in 15-min intervals):
      8   = 2 hours
      24  = 6 hours
      96  = 24 hours (1 day)
      336 = 7 days (1 week)

    min_periods=1: compute stats even if the window isn't full yet
    (important at the start of each meter's history — don't produce NaNs)
    """

    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Helper: for a given window size and aggregation, compute per-meter rolling stat
    # We use transform(lambda) where the lambda applies rolling inside each group.
    def roll_agg(window: int, agg: str):
        """Compute rolling aggregation per meter and return as a Series aligned to df index."""
        return df.groupby("meter_id")["consumption_kwh"].transform(
            lambda x: getattr(x.rolling(window=window, min_periods=1), agg)()
        )

    # ── 2-hour rolling stats (short window — catches sudden spikes) ───────────
    df["roll_mean_2h"]  = roll_agg(8, "mean")
    df["roll_std_2h"]   = roll_agg(8, "std").fillna(0)
    # fillna(0) for std: if all values in window are identical, std=NaN → fill 0

    # ── 24-hour rolling stats (daily baseline) ────────────────────────────────
    df["roll_mean_24h"] = roll_agg(96, "mean")
    df["roll_std_24h"]  = roll_agg(96, "std").fillna(0)
    df["roll_min_24h"]  = roll_agg(96, "min")
    df["roll_max_24h"]  = roll_agg(96, "max")

    # ── 7-day rolling stats (weekly baseline — captures weekly cycles) ─────────
    df["roll_mean_7d"]  = roll_agg(336, "mean")
    df["roll_std_7d"]   = roll_agg(336, "std").fillna(0)

    # ── Z-score: how many standard deviations from the rolling mean? ──────────
    # This is the core anomaly signal:
    #   z = (current - rolling_mean) / rolling_std
    # |z| > 3 is a classic anomaly threshold (3-sigma rule)
    # Positive z = spike above normal; Negative z = drop below normal

    eps = 1e-6
    df["zscore_24h"] = (
        (df["consumption_kwh"] - df["roll_mean_24h"]) /
        (df["roll_std_24h"] + eps)
    )

    df["zscore_7d"] = (
        (df["consumption_kwh"] - df["roll_mean_7d"]) /
        (df["roll_std_7d"] + eps)
    )

    # ── Coefficient of variation (CV) — measures consumption volatility ───────
    # CV = std / mean — dimensionless measure of variability
    # High CV = erratic consumption (tamper signal)
    # Low CV  = very stable consumption (could indicate theft — meter is bypassed)
    df["cv_24h"] = df["roll_std_24h"] / (df["roll_mean_24h"] + eps)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Peer-Group Deviation Features
# "How does this meter compare to other meters in the same zone?"
# This is our most powerful theft/anomaly signal because:
#   - A thief can bypass their own meter but not their neighbors' meters
#   - If zone_2 averages 0.8 kWh and one meter shows 0.05 kWh → suspicious
# ══════════════════════════════════════════════════════════════════════════════

def add_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each meter at each timestamp, computes how it deviates from
    the zone-level average and median.

    IMPORTANT: We exclude the meter itself from its zone average.
    (Otherwise a very anomalous meter would inflate its own zone average.)
    This is called a "leave-one-out" zone mean.
    """

    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Step 1: Compute zone total consumption at each timestamp
    # groupby(['zone_id', 'timestamp']) → for each zone at each time point
    zone_stats = df.groupby(["zone_id", "timestamp"])["consumption_kwh"].agg(
        zone_sum   = "sum",    # total kWh across all meters in zone
        zone_mean  = "mean",   # average kWh per meter in zone
        zone_std   = "std",    # std deviation within zone
        zone_count = "count",  # number of meters in zone
        zone_median= "median"  # median kWh in zone (robust to outliers)
    ).reset_index()

    # Step 2: Join zone stats back to the main DataFrame
    # merge on zone_id + timestamp so each row gets its zone's stats
    df = df.merge(zone_stats, on=["zone_id", "timestamp"], how="left")

    eps = 1e-6

    # Step 3: Leave-one-out zone mean
    # Formula: (zone_sum - this_meter) / (zone_count - 1)
    # Gives the average of ALL OTHER meters in this zone at this timestamp
    df["peer_mean_excl"] = (
        (df["zone_sum"] - df["consumption_kwh"]) /
        (df["zone_count"] - 1 + eps)
    )

    # Step 4: Peer deviation features
    # How much does this meter differ from its zone peers?

    # Absolute difference from peer mean
    df["peer_diff"]  = df["consumption_kwh"] - df["peer_mean_excl"]

    # Ratio vs peer mean — directional signal
    # ratio < 0.3 and sustained = strong theft indicator
    # ratio > 3.0 and sustained = strong peer_deviation indicator
    df["peer_ratio"] = df["consumption_kwh"] / (df["peer_mean_excl"] + eps)

    # Peer z-score: how many std deviations from zone average?
    df["peer_zscore"] = (
        (df["consumption_kwh"] - df["zone_mean"]) /
        (df["zone_std"].fillna(eps) + eps)
    )

    # Drop intermediate columns we don't need downstream
    # (zone_sum and zone_count were only needed for the leave-one-out calc)
    df = df.drop(columns=["zone_sum", "zone_count"])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Rate-of-Change Features
# "How fast is consumption changing right now?"
# Theft and tampering often produce unusually rapid changes.
# A normal household's consumption changes gradually; bypassing a meter
# causes an instant step-change.
# ══════════════════════════════════════════════════════════════════════════════

def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes first and second derivatives of consumption per meter.

    First derivative  (delta): how much did it change in last 15 min?
    Second derivative (accel): is the rate of change itself accelerating?

    High |delta| = rapid change (spike or crash)
    High |accel| = change is getting faster (escalating behavior)
    """

    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("meter_id")["consumption_kwh"]

    # diff(1) = current_value - previous_value (one 15-min step back)
    # Positive = consumption went up; Negative = consumption went down
    df["delta_1step"] = grp.diff(1)

    # diff(4) = change over last 1 hour
    df["delta_1h"]    = grp.diff(4)

    # Second derivative: rate-of-change of rate-of-change
    # If delta is increasing rapidly → consumption is accelerating
    df["accel_1step"] = df.groupby("meter_id")["delta_1step"].diff(1)

    # Absolute delta — for models that can't use sign information directly
    df["abs_delta_1h"] = df["delta_1h"].abs()

    # Percentage change vs 1 step ago (normalised by prior value)
    eps = 1e-6
    df["pct_change_1step"] = df["delta_1step"] / (grp.shift(1) + eps)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Gap / Data Quality Features
# "Is this meter sending regular readings, or are there suspicious gaps?"
# In real BESCOM deployments, meters occasionally miss transmissions.
# But if a meter stops reporting RIGHT AFTER a manual reading → suspicious.
# ══════════════════════════════════════════════════════════════════════════════

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects gaps in meter reporting and flags them as features.

    In our synthetic dataset, all readings are present (no actual gaps).
    In production, you'd join against the expected time index per meter
    and flag missing slots. We simulate the detection logic here.
    """

    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Compute time gap between consecutive readings per meter (in minutes)
    df["prev_timestamp"] = df.groupby("meter_id")["timestamp"].shift(1)

    df["gap_minutes"] = (
        (df["timestamp"] - df["prev_timestamp"])
        .dt.total_seconds() / 60
    )

    # Flag gaps larger than expected interval (15 min + small tolerance)
    # In production: gaps > 30 min = missed reading
    df["has_gap"]        = (df["gap_minutes"] > 20).astype(int)

    # Cumulative gap count per meter — chronic gappers are more suspicious
    df["cumulative_gaps"] = df.groupby("meter_id")["has_gap"].cumsum()

    # Drop the helper column (prev_timestamp was only needed for gap calc)
    df = df.drop(columns=["prev_timestamp"])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Zone-Level Aggregates (for Demand Forecasting — Part A)
# Part A needs features at the zone/feeder level, not just individual meters.
# These features feed the demand forecast model.
# ══════════════════════════════════════════════════════════════════════════════

def build_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a zone-level feature DataFrame for demand forecasting.
    Aggregates individual meter readings up to feeder/zone level.

    Returns a separate DataFrame (one row per zone per timestamp)
    that demand_forecast.py will use directly.
    """

    # Aggregate meter readings to zone level
    zone_df = df.groupby(["zone_id", "timestamp"]).agg(
        total_consumption  = ("consumption_kwh", "sum"),    # Total zone load (kW)
        mean_consumption   = ("consumption_kwh", "mean"),   # Average per meter
        std_consumption    = ("consumption_kwh", "std"),    # Spread within zone
        meter_count        = ("meter_id",        "nunique"),# Active meter count
        max_consumption    = ("consumption_kwh", "max"),    # Peak single meter
        min_consumption    = ("consumption_kwh", "min"),    # Lowest single meter
    ).reset_index()

    # Add time features to zone-level df (same logic as meter-level)
    zone_df["timestamp"]  = pd.to_datetime(zone_df["timestamp"])
    zone_df["hour"]       = zone_df["timestamp"].dt.hour
    zone_df["day_of_week"]= zone_df["timestamp"].dt.dayofweek
    zone_df["month"]      = zone_df["timestamp"].dt.month
    zone_df["is_weekend"] = (zone_df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for zone-level model as well
    zone_df["hour_sin"]   = np.sin(2 * np.pi * zone_df["hour"] / 24)
    zone_df["hour_cos"]   = np.cos(2 * np.pi * zone_df["hour"] / 24)
    zone_df["dow_sin"]    = np.sin(2 * np.pi * zone_df["day_of_week"] / 7)
    zone_df["dow_cos"]    = np.cos(2 * np.pi * zone_df["day_of_week"] / 7)

    # Rolling zone-level demand (last 24h zone total)
    zone_df = zone_df.sort_values(["zone_id", "timestamp"])
    zone_df["zone_roll_mean_24h"] = (
        zone_df.groupby("zone_id")["total_consumption"]
        .transform(lambda x: x.rolling(96, min_periods=1).mean())
    )

    # Zone demand lag (24h ago) — used as baseline in forecast
    zone_df["zone_lag_24h"] = (
        zone_df.groupby("zone_id")["total_consumption"].shift(96)
    )
    zone_df["zone_lag_7d"]  = (
        zone_df.groupby("zone_id")["total_consumption"].shift(672)
    )

    return zone_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Final Feature Selection & Cleaning
# Drops columns not needed for ML, handles NaNs introduced by lags/rolling.
# ══════════════════════════════════════════════════════════════════════════════

# These are the features our ML models will actually use
ANOMALY_FEATURES = [
    # Consumption
    "consumption_kwh",
    # Time features
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday", "time_of_day",
    # Lag features
    "lag_1h", "lag_24h", "lag_48h", "lag_7d",
    "ratio_vs_24h", "ratio_vs_7d", "diff_vs_24h", "diff_vs_7d",
    # Rolling stats
    "roll_mean_2h", "roll_std_2h",
    "roll_mean_24h", "roll_std_24h", "roll_min_24h", "roll_max_24h",
    "roll_mean_7d", "roll_std_7d",
    "zscore_24h", "zscore_7d", "cv_24h",
    # Peer features
    "peer_diff", "peer_ratio", "peer_zscore",
    "zone_mean", "zone_std", "zone_median",
    # Rate of change
    "delta_1step", "delta_1h", "accel_1step", "abs_delta_1h", "pct_change_1step",
    # Gap features
    "has_gap", "cumulative_gaps",
    # Gap minutes (useful signal even when not a "gap")
    "gap_minutes",
]

FORECAST_FEATURES = [
    "total_consumption", "mean_consumption", "std_consumption",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "month",
    "zone_roll_mean_24h", "zone_lag_24h", "zone_lag_7d",
]


def clean_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Final cleaning step before saving:
      1. Keep only needed columns (+ metadata for joins)
      2. Fill NaNs introduced by lag/rolling with forward fill then 0
      3. Clip extreme values (Isolation Forest is sensitive to outliers)
    """

    # Metadata columns to keep alongside features (not fed to model)
    meta_cols = ["timestamp", "meter_id", "zone_id", "meter_type",
                 "anomaly_type", "is_anomaly"]

    # Keep only columns that actually exist in df
    # (some may be missing if an earlier step failed)
    available_features = [c for c in feature_cols if c in df.columns]
    available_meta     = [c for c in meta_cols if c in df.columns]

    df_out = df[available_meta + available_features].copy()

    # Determine groupby key — meter-level uses meter_id, zone-level uses zone_id
    group_key = "meter_id" if "meter_id" in df_out.columns else "zone_id"

    # Fill NaNs: first forward-fill within each group (uses previous reading)
    # then fill remaining NaNs (at the very start) with 0
    df_out[available_features] = (
        df_out.groupby(group_key)[available_features]
        .transform(lambda x: x.ffill().fillna(0))
    )

    # Clip extreme ratio values — ratios can blow up to 1000+ for near-zero lags
    # We cap at 20x to keep features on a reasonable scale
    ratio_cols = [c for c in available_features if "ratio" in c or "zscore" in c]
    df_out[ratio_cols] = df_out[ratio_cols].clip(-20, 20)

    # Clip pct_change to avoid -1 to 100+ blowup
    if "pct_change_1step" in df_out.columns:
        df_out["pct_change_1step"] = df_out["pct_change_1step"].clip(-5, 5)

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_summary(df: pd.DataFrame, name: str):
    """Prints a clean summary of the feature DataFrame."""

    print(f"\n{'='*60}")
    print(f"FEATURE SUMMARY: {name}")
    print(f"{'='*60}")
    print(f"  Shape:          {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage:   {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    null_counts = df.isnull().sum()
    nulls_present = null_counts[null_counts > 0]
    if len(nulls_present) == 0:
        print(f"  NaN values:     None ✓")
    else:
        print(f"  NaN values:     {len(nulls_present)} columns have NaNs")
        print(nulls_present.to_string())

    print(f"\n  Feature columns:")
    feature_cols = [c for c in df.columns
                    if c not in ["timestamp","meter_id","zone_id",
                                 "meter_type","anomaly_type","is_anomaly"]]
    for col in feature_cols:
        vals = df[col]
        print(f"    {col:30s}  min={vals.min():8.3f}  mean={vals.mean():8.3f}  max={vals.max():8.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Main Pipeline
# Runs all feature engineering steps in order.
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(input_path: str = "smart_meter_data.csv") -> tuple:
    """
    Runs the full feature engineering pipeline.

    Returns:
        meter_features_df  — one row per (meter, timestamp), 25+ features
        zone_features_df   — one row per (zone, timestamp), for Part A forecast
    """

    print("BESCOM Feature Engineering Pipeline")
    print("=" * 60)

    # ── Load raw data ──────────────────────────────────────────────────────────
    print("\n[1/8] Loading raw data...")
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    print(f"      Loaded {len(df):,} rows, {df.shape[1]} columns")

    # ── Time features ──────────────────────────────────────────────────────────
    print("[2/8] Adding time features (cyclical encoding, holidays)...")
    df = add_time_features(df)
    print(f"      +{sum(1 for c in df.columns if 'sin' in c or 'cos' in c or 'holiday' in c)} new columns")

    # ── Lag features ───────────────────────────────────────────────────────────
    print("[3/8] Adding lag features (1h, 24h, 48h, 7d)...")
    df = add_lag_features(df)
    print(f"      +lag_1h, lag_24h, lag_48h, lag_7d, ratio_vs_24h, ratio_vs_7d, ...")

    # ── Rolling stats ──────────────────────────────────────────────────────────
    print("[4/8] Computing rolling statistics (2h, 24h, 7d windows)...")
    df = add_rolling_features(df)
    print(f"      +roll_mean, roll_std, zscore, cv across 3 windows")

    # ── Peer-group features ────────────────────────────────────────────────────
    print("[5/8] Computing peer-group deviation features...")
    df = add_peer_features(df)
    print(f"      +peer_diff, peer_ratio, peer_zscore, zone_mean, zone_std, zone_median")

    # ── Rate of change ─────────────────────────────────────────────────────────
    print("[6/8] Computing rate-of-change features...")
    df = add_rate_of_change(df)
    print(f"      +delta_1step, delta_1h, accel_1step, abs_delta_1h, pct_change_1step")

    # ── Gap features ───────────────────────────────────────────────────────────
    print("[7/8] Computing gap / data quality features...")
    df = add_gap_features(df)
    print(f"      +has_gap, cumulative_gaps, gap_minutes")

    # ── Zone-level features for Part A ────────────────────────────────────────
    print("[8/8] Building zone-level features for demand forecasting...")
    zone_df = build_zone_features(df)
    print(f"      Zone DataFrame: {zone_df.shape[0]:,} rows × {zone_df.shape[1]} columns")

    # ── Clean & select final feature sets ──────────────────────────────────────
    print("\nCleaning and selecting final feature sets...")
    meter_features = clean_features(df, ANOMALY_FEATURES)
    zone_features  = clean_features(zone_df, FORECAST_FEATURES) if FORECAST_FEATURES else zone_df

    return meter_features, zone_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    meter_features, zone_features = run_pipeline("smart_meter_data.csv")

    # Print summaries
    print_feature_summary(meter_features, "Meter-level features (for anomaly detection)")

    # Save outputs
    print("\nSaving feature files...")

    meter_out = "features_meter.csv"
    zone_out  = "features_zone.csv"

    meter_features.to_csv(meter_out, index=False)
    zone_features.to_csv(zone_out, index=False)

    import os
    print(f"Saved: {meter_out}  ({os.path.getsize(meter_out)/1024**2:.1f} MB)")
    print(f"Saved: {zone_out}   ({os.path.getsize(zone_out)/1024**2:.1f} MB)")

    print(f"\nTotal features built: {len(ANOMALY_FEATURES)} (meter-level) + {len(FORECAST_FEATURES)} (zone-level)")
    print("\nDone! Next step: python demand_forecast.py  and  python anomaly_detector.py")
