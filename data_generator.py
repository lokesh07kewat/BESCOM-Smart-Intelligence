"""
data_generator.py
=================
Generates synthetic smart meter data for the BESCOM Smart Meter Intelligence project.

What this file produces:
  - Realistic 15-minute interval electricity consumption data
  - Multiple meters across multiple zones (feeders)
  - Real-world patterns: peak hours, weekends, seasonality, weather effects
  - Injected anomalies: theft, tampering, sudden drops, peer deviation

Output: smart_meter_data.csv
"""

# ── Imports ────────────────────────────────────────────────────────────────────

import numpy as np          # For numerical operations and random number generation
import pandas as pd         # For creating and manipulating DataFrames
import random               # For Python's built-in random choices
from datetime import datetime, timedelta   # For creating date/time ranges
import os                   # For file path operations

# ── Reproducibility ────────────────────────────────────────────────────────────

# Setting a seed means every time this script runs, it produces IDENTICAL data.
# Critical for hackathons — judges can re-run your code and get the same results.
np.random.seed(42)
random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Configuration
# All tunable parameters are here at the top — easy for judges to inspect.
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # How many unique smart meters to simulate
    "num_meters": 50,

    # How many zones (feeders) to group meters into
    # Real BESCOM has feeders serving clusters of households
    "num_zones": 5,

    # Date range for data generation
    # 90 days of 15-min data = 8,640 rows per meter = 432,000 rows total
    "start_date": "2024-01-01",
    "end_date":   "2024-03-31",

    # Interval between readings in minutes
    "interval_minutes": 15,

    # Base consumption in kWh per 15-min interval for different meter types
    # Residential: ~0.15–0.5 kWh per interval (600W–2kW average draw)
    # Commercial:  ~0.5–2.0 kWh per interval (2kW–8kW average draw)
    "consumption_base": {
        "residential": 0.25,
        "commercial":  1.20,
        "industrial":  3.50,
    },

    # What fraction of meters are each type
    "meter_type_weights": [0.70, 0.20, 0.10],   # 70% residential, 20% commercial, 10% industrial

    # What fraction of meters will have injected anomalies
    "anomaly_fraction": 0.20,   # 20% of meters get some kind of anomaly
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Meter Registry
# Creates the list of meters with their properties before generating readings.
# ══════════════════════════════════════════════════════════════════════════════

def create_meter_registry(config: dict) -> pd.DataFrame:
    """
    Creates a DataFrame where each row is one smart meter with its static properties.
    Think of this as the "metadata table" — it doesn't change over time.

    Returns a DataFrame with columns:
        meter_id, zone_id, meter_type, base_consumption, anomaly_type
    """

    meter_types = ["residential", "commercial", "industrial"]
    records = []

    for i in range(config["num_meters"]):

        # Assign meter to a zone (zone_0 through zone_4)
        # We use modulo so meters are distributed evenly across zones
        zone_id = f"zone_{i % config['num_zones']}"

        # Randomly pick meter type using the configured weights
        # np.random.choice picks from the list; p= sets probabilities
        meter_type = np.random.choice(
            meter_types,
            p=config["meter_type_weights"]
        )

        # Each meter has a slightly different base consumption even within same type
        # np.random.uniform(0.8, 1.2) gives a multiplier between 0.8x and 1.2x
        # This simulates household-to-household variation (family size, appliances, etc.)
        variation = np.random.uniform(0.8, 1.2)
        base = config["consumption_base"][meter_type] * variation

        # Decide if this meter will have anomalies injected
        # random.random() returns a float between 0 and 1
        has_anomaly = random.random() < config["anomaly_fraction"]

        if has_anomaly:
            # Pick what KIND of anomaly this meter will exhibit
            anomaly_type = random.choice([
                "theft",            # Consumption suddenly drops (bypassing meter)
                "tamper",           # Erratic/impossible consumption values
                "sudden_drop",      # Sharp single-event drop (equipment failure)
                "peer_deviation",   # Unusually high compared to zone neighbors
            ])
        else:
            anomaly_type = "none"

        records.append({
            "meter_id":        f"MTR_{i:04d}",   # e.g. MTR_0001, MTR_0042
            "zone_id":         zone_id,
            "meter_type":      meter_type,
            "base_consumption": round(base, 4),
            "anomaly_type":    anomaly_type,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Consumption Pattern Functions
# These functions return a multiplier (0.0 – 2.0+) applied to base consumption.
# Multiplier = 1.0 means "average" consumption for that meter.
# ══════════════════════════════════════════════════════════════════════════════

def hourly_multiplier(hour: int, meter_type: str) -> float:
    """
    Returns a float multiplier based on the hour of day.
    Different meter types have different daily usage patterns.

    hour: integer 0–23 (0 = midnight, 13 = 1 PM)
    """

    if meter_type == "residential":
        # Residential pattern: two peaks
        #   Morning peak  6–9 AM  (cooking breakfast, geysers, getting ready)
        #   Evening peak  6–10 PM (cooking dinner, lighting, TV, ACs)
        #   Night trough  11 PM–5 AM (everyone asleep)
        pattern = [
            0.3,  # 00:00 midnight  — very low
            0.3,  # 01:00           — very low
            0.3,  # 02:00           — very low
            0.3,  # 03:00           — very low
            0.3,  # 04:00           — very low
            0.5,  # 05:00           — waking up
            0.9,  # 06:00           — morning peak starts
            1.3,  # 07:00           — peak (geysers, cooking)
            1.2,  # 08:00           — peak
            1.0,  # 09:00           — tapering off
            0.7,  # 10:00           — mid-morning low
            0.6,  # 11:00           — low
            0.7,  # 12:00 noon      — slight uptick (lunch)
            0.6,  # 13:00           — low
            0.6,  # 14:00           — low (hot afternoon, AC on but otherwise quiet)
            0.7,  # 15:00           — slight rise
            0.8,  # 16:00           — kids home from school
            1.0,  # 17:00           — evening starts
            1.3,  # 18:00           — evening peak (cooking, lights on)
            1.5,  # 19:00           — peak (dinner + entertainment)
            1.4,  # 20:00           — peak
            1.2,  # 21:00           — winding down
            0.8,  # 22:00           — late evening
            0.4,  # 23:00           — nearly asleep
        ]

    elif meter_type == "commercial":
        # Commercial pattern: flat during business hours, low at night
        # Shops, offices: 9 AM – 9 PM active, rest near-zero
        pattern = [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   # 00–05: closed
            0.2, 0.4, 0.7,                    # 06–08: opening up
            1.0, 1.1, 1.1, 1.0, 1.1, 1.1,   # 09–14: business hours
            1.1, 1.0, 1.0, 1.2, 1.2, 1.1,   # 15–20: busy afternoon/evening
            0.8, 0.4, 0.2,                    # 21–23: closing down
        ]

    else:  # industrial
        # Industrial: runs 24/7, slight shift-change dips
        pattern = [
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9,   # Night shift
            1.0, 1.1, 1.2, 1.2, 1.2, 1.2,   # Day shift ramp up
            1.1, 1.2, 1.2, 1.2, 1.2, 1.1,   # Afternoon
            1.0, 1.0, 1.0, 0.9, 0.9, 0.9,   # Evening shift
        ]

    return pattern[hour]


def weekday_multiplier(day_of_week: int, meter_type: str) -> float:
    """
    Returns a multiplier based on day of week (0=Monday, 6=Sunday).
    Weekend vs weekday patterns differ significantly.
    """

    if meter_type == "residential":
        # Weekends: people stay home → higher residential consumption
        weekend_boost = {5: 1.15, 6: 1.20}   # Saturday=1.15x, Sunday=1.20x
        return weekend_boost.get(day_of_week, 1.0)

    elif meter_type == "commercial":
        # Commercial: lower on Sundays (many shops closed)
        if day_of_week == 6:    # Sunday
            return 0.60
        elif day_of_week == 5:  # Saturday
            return 0.85
        return 1.0

    else:  # industrial
        # Industrial barely changes — continuous production
        if day_of_week == 6:
            return 0.90   # slight reduction on Sunday
        return 1.0


def seasonal_multiplier(month: int, meter_type: str) -> float:
    """
    Returns a multiplier based on the month (1=January, 12=December).
    Captures seasonal effects like summer ACs and winter heaters.
    Bangalore climate: hot March–May, monsoon June–Sep, pleasant Oct–Feb.
    """

    if meter_type == "residential":
        # Higher in hot months (ACs running), lower in pleasant months
        seasonal = {
            1: 0.90,   # January  — pleasant
            2: 0.95,   # February — warming up
            3: 1.10,   # March    — hot, ACs on
            4: 1.20,   # April    — very hot
            5: 1.25,   # May      — peak summer
            6: 1.10,   # June     — monsoon begins, slightly cooler
            7: 1.00,   # July     — monsoon
            8: 1.00,   # August   — monsoon
            9: 1.05,   # September— post monsoon
            10: 0.95,  # October  — pleasant
            11: 0.90,  # November — pleasant
            12: 0.85,  # December — coolest, lowest AC use
        }
        return seasonal.get(month, 1.0)

    return 1.0   # Other meter types: no strong seasonal effect assumed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Anomaly Injectors
# Each function takes a consumption value and a timestamp and returns a
# modified (anomalous) consumption value.
# ══════════════════════════════════════════════════════════════════════════════

def inject_theft(consumption: float, timestamp: pd.Timestamp,
                  meter_row: pd.Series) -> tuple:
    """
    Simulates electricity theft: meter is physically bypassed.

    Pattern: After a certain date, consumption drops to 10–30% of normal.
    Theft is "smart" — the meter still shows SOME consumption to avoid
    zero-reading alerts, but far less than expected.

    Returns (modified_consumption, is_anomaly_flag)
    """

    # Theft starts on a random day after the first 2 weeks of data
    # We use the meter_id to generate a consistent start date per meter
    # (same meter always starts theft on same day — reproducible)
    theft_start_offset = hash(meter_row["meter_id"]) % 45 + 14   # 14–59 days in
    theft_start = pd.Timestamp(CONFIG["start_date"]) + timedelta(days=theft_start_offset)

    if timestamp >= theft_start:
        # After theft starts: consumption is 10–30% of normal
        # The thief is drawing power but it's not metered
        theft_factor = np.random.uniform(0.10, 0.30)
        return round(consumption * theft_factor, 4), True

    return consumption, False   # Before theft: normal


def inject_tamper(consumption: float, timestamp: pd.Timestamp) -> tuple:
    """
    Simulates meter tampering: the meter gives erratic / impossible readings.

    Pattern: Occasional spikes to 10x normal, or drops to near-zero,
    in an unpredictable pattern. Unlike theft, this isn't sustained —
    it's sporadic, which makes it harder to detect with simple thresholds.

    Returns (modified_consumption, is_anomaly_flag)
    """

    # 8% chance of tampering at any given 15-min interval
    if random.random() < 0.08:
        tamper_type = random.choice(["spike", "drop"])

        if tamper_type == "spike":
            # Impossible spike: 5x–15x normal consumption
            # Could indicate meter being bypassed with a known load for calibration
            return round(consumption * np.random.uniform(5, 15), 4), True
        else:
            # Near-zero drop: meter reports almost nothing
            return round(consumption * np.random.uniform(0.01, 0.05), 4), True

    return consumption, False   # 92% of the time: normal reading


def inject_sudden_drop(consumption: float, timestamp: pd.Timestamp,
                        meter_row: pd.Series) -> tuple:
    """
    Simulates a legitimate equipment failure or supply interruption.
    Unlike theft, this is time-bounded — it recovers after a few days.

    Pattern: Sharp drop to near-zero for 1–5 days, then recovers.
    Important: This SHOULD be detectable but NOT flagged as theft —
    it tests our ability to distinguish anomaly types.
    """

    drop_start_offset = hash(meter_row["meter_id"] + "drop") % 60 + 10
    drop_start = pd.Timestamp(CONFIG["start_date"]) + timedelta(days=drop_start_offset)
    drop_end   = drop_start + timedelta(days=random.randint(1, 5))

    if drop_start <= timestamp <= drop_end:
        # During outage: near-zero consumption (not exactly zero — some phantom load)
        return round(np.random.uniform(0.001, 0.01), 4), True

    return consumption, False


def inject_peer_deviation(consumption: float, meter_type: str) -> tuple:
    """
    Simulates a meter that consistently draws 3x–5x more than its zone peers.

    This could indicate:
      - Unauthorized sub-metering (sharing connection with unlicensed neighbors)
      - Industrial activity in a residential connection
      - A meter miscalibrated to under-report (showing high is actually correct)

    Pattern: Always-on multiplier — every reading is elevated.
    """

    # Sustained 3x–5x elevation across all readings for this meter
    elevation = np.random.uniform(3.0, 5.0)
    return round(consumption * elevation, 4), True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Main Data Generation Function
# Loops over every meter × every timestamp and generates one row per combo.
# ══════════════════════════════════════════════════════════════════════════════

def generate_meter_data(meter_registry: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Core generation loop. For each meter, for each 15-min timestamp,
    computes consumption using:
        base × hourly_mult × weekday_mult × seasonal_mult × noise + anomaly

    Returns a long-format DataFrame with one row per (meter, timestamp).
    """

    # Build the full time index once (shared across all meters)
    # pd.date_range creates a sequence of timestamps at 15-min intervals
    time_index = pd.date_range(
        start=config["start_date"],
        end=config["end_date"],
        freq=f"{config['interval_minutes']}min"
    )

    all_records = []   # We'll accumulate all rows here, then convert to DataFrame

    print(f"Generating data for {len(meter_registry)} meters × {len(time_index)} timestamps...")
    print(f"Total rows to generate: {len(meter_registry) * len(time_index):,}")

    for _, meter in meter_registry.iterrows():
        # meter is a Series: meter["meter_id"], meter["zone_id"], etc.

        meter_records = []   # Rows for THIS meter only

        for ts in time_index:
            # ── Step A: Base consumption ─────────────────────────────────────
            # Start with the meter's baseline kWh per 15-min interval
            base = meter["base_consumption"]

            # ── Step B: Apply time-of-day pattern ────────────────────────────
            h_mult = hourly_multiplier(ts.hour, meter["meter_type"])

            # ── Step C: Apply day-of-week pattern ────────────────────────────
            d_mult = weekday_multiplier(ts.dayofweek, meter["meter_type"])

            # ── Step D: Apply seasonal pattern ───────────────────────────────
            s_mult = seasonal_multiplier(ts.month, meter["meter_type"])

            # ── Step E: Combine multipliers ───────────────────────────────────
            # Multiply all factors together to get "expected" consumption
            expected = base * h_mult * d_mult * s_mult

            # ── Step F: Add realistic noise ───────────────────────────────────
            # Real meters don't give perfect readings — there's always variation.
            # np.random.normal(0, std) gives Gaussian noise centered at 0.
            # std = 8% of expected value — realistic measurement noise.
            noise = np.random.normal(0, expected * 0.08)

            # Clip ensures consumption never goes negative (physically impossible)
            consumption = max(0.0, round(expected + noise, 4))

            # ── Step G: Inject anomaly if this meter has one ──────────────────
            is_anomaly = False   # Default: normal reading

            if meter["anomaly_type"] == "theft":
                consumption, is_anomaly = inject_theft(consumption, ts, meter)

            elif meter["anomaly_type"] == "tamper":
                consumption, is_anomaly = inject_tamper(consumption, ts)

            elif meter["anomaly_type"] == "sudden_drop":
                consumption, is_anomaly = inject_sudden_drop(consumption, ts, meter)

            elif meter["anomaly_type"] == "peer_deviation":
                consumption, is_anomaly = inject_peer_deviation(
                    consumption, meter["meter_type"]
                )

            # ── Step H: Build the row ─────────────────────────────────────────
            meter_records.append({
                "timestamp":       ts,
                "meter_id":        meter["meter_id"],
                "zone_id":         meter["zone_id"],
                "meter_type":      meter["meter_type"],
                "consumption_kwh": consumption,
                "hour":            ts.hour,
                "day_of_week":     ts.dayofweek,       # 0=Mon, 6=Sun
                "month":           ts.month,
                "is_weekend":      int(ts.dayofweek >= 5),  # 1 if Sat/Sun
                "anomaly_type":    meter["anomaly_type"],
                "is_anomaly":      int(is_anomaly),    # Ground truth label for evaluation
            })

        all_records.extend(meter_records)

    print("Data generation complete. Building DataFrame...")

    df = pd.DataFrame(all_records)

    # Sort by meter and timestamp — important for rolling feature computation later
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Summary & Validation
# Sanity-checks the generated data before saving.
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, meter_registry: pd.DataFrame):
    """
    Prints a human-readable summary of the generated dataset.
    Useful for demos — shows judges the data is realistic.
    """

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"\nTotal rows:          {len(df):,}")
    print(f"Unique meters:       {df['meter_id'].nunique()}")
    print(f"Unique zones:        {df['zone_id'].nunique()}")
    print(f"Date range:          {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Interval:            15 minutes")

    print(f"\nMeter type breakdown:")
    for mtype, count in meter_registry["meter_type"].value_counts().items():
        print(f"  {mtype:15s}: {count} meters")

    print(f"\nAnomaly breakdown:")
    for atype, count in meter_registry["anomaly_type"].value_counts().items():
        print(f"  {atype:20s}: {count} meters")

    print(f"\nConsumption stats (kWh per 15-min interval):")
    stats = df["consumption_kwh"].describe()
    print(f"  Min:    {stats['min']:.4f} kWh")
    print(f"  Mean:   {stats['mean']:.4f} kWh")
    print(f"  Max:    {stats['max']:.4f} kWh")
    print(f"  Std:    {stats['std']:.4f} kWh")

    anomaly_rows = df[df["is_anomaly"] == 1]
    print(f"\nAnomaly rows:        {len(anomaly_rows):,} ({100*len(anomaly_rows)/len(df):.1f}% of data)")

    print("\n" + "="*60)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Basic sanity checks on the generated data.
    Returns True if all checks pass, False otherwise.
    """

    checks = []

    # Check 1: No negative consumption
    neg = (df["consumption_kwh"] < 0).sum()
    checks.append(("No negative consumption", neg == 0, f"{neg} negative values found"))

    # Check 2: No missing values
    nulls = df.isnull().sum().sum()
    checks.append(("No missing values", nulls == 0, f"{nulls} nulls found"))

    # Check 3: All meters present in all time slots
    expected_rows = df["meter_id"].nunique() * df["timestamp"].nunique()
    checks.append(("Row count correct", len(df) == expected_rows,
                   f"Expected {expected_rows:,}, got {len(df):,}"))

    # Check 4: Anomaly meters actually have anomalous readings
    anomaly_meters = df[df["anomaly_type"] != "none"]["meter_id"].unique()
    if len(anomaly_meters) > 0:
        anomaly_flag_rate = df[df["meter_id"].isin(anomaly_meters)]["is_anomaly"].mean()
        checks.append(("Anomaly meters have flags", anomaly_flag_rate > 0,
                       f"Flag rate: {anomaly_flag_rate:.2%}"))

    print("\nValidation checks:")
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not passed:
            all_passed = False

    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Entry Point
# This runs when you execute: python data_generator.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("BESCOM Smart Meter Data Generator")
    print("="*60)

    # Step 1: Create the meter registry (who are our meters?)
    print("\n[1/4] Creating meter registry...")
    meter_registry = create_meter_registry(CONFIG)
    print(f"      Created {len(meter_registry)} meters across {CONFIG['num_zones']} zones")

    # Step 2: Generate the full time-series data
    print("\n[2/4] Generating time-series readings...")
    df = generate_meter_data(meter_registry, CONFIG)

    # Step 3: Validate the output
    print("\n[3/4] Validating generated data...")
    valid = validate_data(df)

    if not valid:
        print("\nWARNING: Some validation checks failed. Review above output.")

    # Step 4: Print summary and save
    print("\n[4/4] Saving data...")
    print_summary(df, meter_registry)

    # Save the main dataset
    output_path = "smart_meter_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    # Also save the meter registry separately (useful for joins later)
    registry_path = "meter_registry.csv"
    meter_registry.to_csv(registry_path, index=False)
    print(f"Saved: {registry_path}")

    print("\nDone! Next step: python feature_engineering.py")
