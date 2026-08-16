# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 06:01:23 2026
Project Title: Industrial IoT Telemetry Pipeline and Engine Optimization Study
Purpose: Implement an advanced architectural evaluation comparing Pandas (Eager) and Polars (Lazy) engines.
Author: Travis Asher

Scenario: A regional smart-grid sensor network emits messy, unstructured log arrays. The objective
is to clean the telemetry strings, parse geographical spatial coordinates embedded inside text
flags, classify risk levels without using iterative loops, and calculate isolated ranking metrics
per zone via partitioned window functions.
"""

import csv
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# Raw telemetry stream initialization
raw_data = {
    " SENSOR_ID  ": [
        "  sn-101 ", " sn-102", "sn-103 ", "SN-101", "sn-104", 
        "sn-102 ", "  sn-105 ", "sn-101", "  SN-103", "sn-104 ", 
        " sn-105", "sn-102", "  sn-101 ", "sn-103 ", "sn-104"
    ],
    "raw_logs": [
        "ERROR|LOC:40.7128,-74.0060|VOLT:240.5",
        "WARN|LOC:39.9526,-75.1652|VOLT:missing",
        "OK|LOC:34.0522,-118.2437|VOLT:120.2",
        "ERROR|LOC:40.7128,-74.0060|VOLT:285.0",
        "OK|LOC:41.8781,-87.6298|VOLT:118.9",
        "ERROR|LOC:39.9526,-75.1652|VOLT:310.1",
        "WARN|LOC:null|VOLT:190.4",
        "OK|LOC:40.7128,-74.0060|VOLT:238.1",
        "WARN|LOC:34.0522,-118.2437|VOLT:244.9",
        "ERROR|LOC:41.8781,-87.6298|VOLT:262.3",
        "OK|LOC:45.5017,-73.5673|VOLT:121.5",
        "OK|LOC:39.9526,-75.1652|VOLT:220.4",
        "WARN|LOC:40.7128,-74.0060|VOLT:255.2",
        "ERROR|LOC:34.0522,-118.2437|VOLT:missing",
        "WARN|LOC:41.8781,-87.6298|VOLT:249.8"
    ],
    "timestamp": [
        "2026-08-01 10:00", "2026-08-01 10:05", "2026-08-01 10:10", "2026-08-01 10:15", "2026-08-01 10:20",
        "2026-08-01 10:25", "2026-08-01 10:30", "2026-08-01 10:35", "2026-08-01 10:40", "2026-08-01 10:45",
        "2026-08-01 10:50", "2026-08-01 10:55", "2026-08-01 11:00", "2026-08-01 11:05", "2026-08-01 11:10"
    ]
}

# ================================================================================================
# %%% Phase 0: Data Preparation and Ingestion
# ================================================================================================
# Establish the pipeline directory paths
script_dir = Path.cwd()
csv_file_path = script_dir / "sensor_telemetry_raw.csv"

# Convert the raw dictionary dataset to a local CSV file:
    # (1) Open a local file handler in write-only mode.
    # (2) Pivot the key-value dictionary elements vertically using zip().
    # (3) Ingest the dictionary keys as the CSV column headers.
    # (4) Write the transposed records down into individual dataset rows.
dt_val = list(raw_data.values())
tr_dt_val = list(zip(*dt_val))

with open(csv_file_path, "w", newline="") as new_file:
    writer = csv.writer(new_file)
    writer.writerow(list(raw_data.keys()))
    writer.writerows(tr_dt_val)

# Read the generated CSV file into both processing frameworks
df_telemetry_source = pd.read_csv(csv_file_path)
lf_telemetry_source = pl.scan_csv(csv_file_path)

# ================================================================================================
# END PHASE 0
# ================================================================================================

#### CORE EXECUTION PIPELINES ####
# All transformations below are executed utilizing a strict single method chain for Pandas, and a
# strict single expression query plan for Polars.



# ================================================================================================
# %%PART 1: Pandas Implementation with Pandas DataFrame Objects
# ================================================================================================

# %%% Phase 1: Ingestion, Header Normalization, and Token Extraction
# Standardize column headers to lowercase and strip all structural whitespace.
# Clean up sensor_id strings to establish a uniform uppercase format ("SN-101") with no 
#   trailing padding.
# Parse the raw_logs string column using Regular Expressions to extract three clean features:
    # (1) status: The alphabetic string preceding the first pipe (|) delimiter.
    # (2) latitude: The numeric coordinate text isolated immediately following the LOC:
    #       flag.
    # (3) voltage: The floating-point decimal trailing the VOLT: flag, coercing non-numeric 
    #       text like "missing" into true null values.
    # (4) timestamp: The raw temporal text string coerced into a native datetime object.
df_parsed_features = (
    df_telemetry_source
    .rename(columns=lambda c: c.strip().lower())
    .astype({"sensor_id": str, "raw_logs": str})
    .assign(
        timestamp = lambda df: pd.to_datetime(df["timestamp"]),
        sensor_id = lambda df: df["sensor_id"].str.strip().str.upper(),
        status = lambda df: df["raw_logs"].str.extract(r"(\A(?:[a-zA-Z])+)\|", expand=False),
        latitude = lambda df: pd.to_numeric(
            df["raw_logs"].str.extract(r":((?:(?:\d+\.\d+))|(?:[a-zA-Z]*))", expand=False),
            errors="coerce"),
        voltage = lambda df: pd.to_numeric(df["raw_logs"].str.extract(
            r"VOLT:((?:\d+\.\d+)?|(?:[a-zA-Z]*))\Z", expand=False), errors="coerce")
        )
    )

# Display the parsed checkpoint DataFrame and schema types
print(f"Phase 1 Cleaned Pandas DataFrame:\n {df_parsed_features}\n")
print(f"Data Types:\n {df_parsed_features.dtypes}\n")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 2: Priority Conditionals and Risk Profile Classification
# Derive a new categorical attribute named risk_profile based on the operational thresholds:
    # If status is "ERROR" and voltage exceeds 250.0 --> label it "CRITICAL".
    # If status is "ERROR" or "WARN" --> label it "HIGH".
    # Otherwise --> label it "STABLE".
    # Engine Constraints: Pandas must execute this vectorization step within .assign() via
    #   np.select().
df_classified_risk = (
    df_parsed_features
    .assign(
        risk_profile = lambda df: np.select(
            condlist=[
                (df["status"] == "ERROR") & (df["voltage"] > 250),
                df["status"].str.fullmatch(r"ERROR|WARN")
            ], 
            choicelist=["CRITICAL", "HIGH"], 
            default="STABLE"
            )
        )
    )

# Isolate target metrics to audit classification accuracy
print(f"Phase 2 Classified Telemetry Audit:\n {
    df_classified_risk[['sensor_id', 'status', 'voltage', 'risk_profile']]}\n")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 3: Spatial Filtering and Anomaly Data Cleansing
# Filter out and drop all rows containing missing values or null entries inside the extracted
#   latitude and voltage columns.
df_sanitized_stream = (
    df_classified_risk
    .query("not voltage.isna() and not latitude.isna()")
    )

# Isolate target records post-cleansing for terminal verification
df_nulls_removed = df_sanitized_stream[["sensor_id", "timestamp", "status", "latitude", 
                                        "voltage", "risk_profile"]]
print(f"Phase 3 Nulls Removed:\n {df_nulls_removed}\n")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 4: Partitioned Analytics and Windowed Voltage Ranking
# Compute a relative ranking metric across the dataset while fully preserving the row density of
#   individual log records (no grouping or structural row reduction):
    # Create a column named voltage_rank_in_sensor that ranks the row's voltage relative to that
    #   specific sensor_id, ordered from highest to lowest voltage.
    # Engine Constraints: Pandas must leverage .groupby().rank().
    # Output a Pandas DataFrame object that displays the number of each unique sensor_id values
df_analytical_base = (
    df_sanitized_stream
    .assign(
        voltage_rank_in_sensor = lambda df: df.groupby(["sensor_id"])["voltage"]
        .rank(ascending=False)
        )
    )

# Isolate the final computed attributes for analytical evaluation and output the grouped Pandas
#   DataFrame displaying the amount of unique sensor_ids; finally, output final Pandas DataFrame
df_partitioned = (
    df_analytical_base[["sensor_id", "voltage", "voltage_rank_in_sensor"]]
    )

uniq_grpd_sensor_pd = (
    df_analytical_base
    .groupby(["sensor_id"]).agg(amt_sensors = pd.NamedAgg(column="raw_logs", aggfunc="count"))
    .sort_values("sensor_id")
    )

df_pandas_final = df_analytical_base[["sensor_id", "timestamp", "status", "latitude", 
                                      "voltage", "risk_profile", "voltage_rank_in_sensor"]]

print(f"Phase 4 Analytic Voltage Evaluation:\n {df_partitioned}\n")
print(f"Phase 4 Unique Sensors Grouped Pandas DataFrame:\n {uniq_grpd_sensor_pd}\n")
print("\n-------------------------------------------------------------------------------\n")
print("----------------------------------FINAL PART 1---------------------------------")
print("\n-------------------------------------------------------------------------------\n")
print(f"Part 1 Final Pandas DataFrame:\n {df_pandas_final}\n")

# ================================================================================================
# END PART 1
# ================================================================================================



# ================================================================================================
# %%PART 2: Polars Implementation with Polars LazyFrame Objects
# ================================================================================================

# %%% Phase 1: Ingestion, Header Normalization, and Token Extraction
# Standardize column headers to lowercase and strip all structural whitespace.
# Clean up sensor_id strings to establish a uniform uppercase format ("SN-101") with no 
#   trailing padding.
# Parse the raw_logs string column using Regular Expressions to extract three clean features:
    # (1) status: The alphabetic string preceding the first pipe (|) delimiter.
    # (2) latitude: The numeric coordinate text isolated immediately following the LOC:
    #       flag.
    # (3) voltage: The floating-point decimal trailing the VOLT: flag, coercing non-numeric 
    #       text like "missing" into true null values.
    # (4) timestamp: The raw temporal text string coerced into a native datetime object.
lf_parsed_features = (
    lf_telemetry_source
    .rename({c: c.strip().lower() for c in lf_telemetry_source.collect_schema().names()})
    .with_columns([
        pl.col("sensor_id").str.to_uppercase().str.strip_chars(),
        pl.col("timestamp").str.to_datetime(),
        pl.col("raw_logs").str.extract(r"^([a-zA-Z]+)\|").alias("status"),
        pl.col("raw_logs").str.extract(r"LOC:([^,|]+)").str.to_decimal(scale=4).alias("latitude"),
        pl.col("raw_logs").str.extract(r"VOLT:((?:\d+\.\d+)|[a-zA-Z]+)$")
        .str.to_decimal(scale=1).alias("voltage")
        ])
    )

# Display the parsed checkpoint DataFrame and schema types
pl_parsed = lf_parsed_features.collect()
pl_parsed = pl_parsed.select("sensor_id","raw_logs","timestamp","status","latitude","voltage")
print(f"Phase 1 Cleaned Polars DataFrame:\n {pl_parsed}")
print(f"Data Types:\n {pl_parsed.dtypes}\n")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 2: Priority Conditionals and Risk Profile Classification
# Derive a new categorical attribute named risk_profile based on the operational thresholds:
    # If status is "ERROR" and voltage exceeds 250.0 --> label it "CRITICAL".
    # If status is "ERROR" or "WARN" --> label it "HIGH".
    # Otherwise --> label it "STABLE".
    # Engine Constraints: Polars must utilize native pl.when().then().otherwise() syntax.
lf_classified_risk = (
    lf_parsed_features.with_columns(
        risk_profile = pl.when((pl.col("status") == "ERROR") & (pl.col("voltage") > 250))
        .then(pl.lit("CRITICAL"))
        .when(pl.col("status").is_in(["ERROR", "WARN"]))
        .then(pl.lit("HIGH"))
        .otherwise(pl.lit("STABLE"))
        )
    )

# Isolate target metrics to audit classification accuracy
pl_risk = lf_classified_risk.collect()
pl_risk = pl_risk.select("sensor_id","status","voltage","risk_profile")
print(f"Phase 2 Classified Telemetry Audit:\n {pl_risk}")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 3: Spatial Filtering and Anomaly Data Cleansing
# Filter out and drop all rows containing missing values or null entries inside the extracted
#   latitude and voltage columns.
lf_sanitized_stream = (
    lf_classified_risk.drop_nulls()
    )

# Isolate target records post-cleansing for terminal verification
pl_sanitized = lf_sanitized_stream.collect()
pl_sanitized = pl_sanitized.select("sensor_id","latitude","voltage")
print(f"Phase 3 Nulls Removed::\n {pl_sanitized}")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 4: Partitioned Analytics and Windowed Voltage Ranking
# Compute a relative ranking metric across the dataset while fully preserving the row density of
#   individual log records (no grouping or structural row reduction):
    # Create a column named voltage_rank_in_sensor that ranks the row's voltage relative to that
    #   specific sensor_id, ordered from highest to lowest voltage.
    # Engine Constraints: Polars must use an expression block evaluated over the partition via
    #   .over().
    # Output a Polars DataFrame object that displays the number of each unique sensor_id values.
    #   NOT required to be part of the single expression Polars query chain
lf_analytical_base = (
    lf_sanitized_stream.with_columns(
        voltage_rank_in_sensor = pl.col("voltage").rank("ordinal", descending=True)
        .over("sensor_id")
        )
    )

# Isolate the final computed attributes for analytical evaluation and output the grouped Polars
#   DataFrame displaying the amount of unique sensor_ids
pl_partitioned = lf_analytical_base.collect()
pl_partitioned = pl_partitioned.select("sensor_id","voltage","voltage_rank_in_sensor")

uniq_grpd_sensor_pl = (
    lf_analytical_base
    .group_by(["sensor_id"])
    .agg(amt_sensors=pl.len())
    .sort("sensor_id")
    .collect()
    )

print(f"Phase 4 Analytic Voltage Evaluation:\n {pl_partitioned}\n")
print(f"Phase 4 Unique Sensors Grouped Polars DataFrame:\n {uniq_grpd_sensor_pl}\n")
print("\n-------------------------------------------------------------------------------\n")


# %%% Phase 5: Query Plan Inspection and Lazy Engine Performance Audit
# Evaluate the compiled query graph of the finalized Polars pipeline:
    # Output the optimized logical execution tree to the terminal console using the .explain()
    #   inspector to audit predicate pushdowns executed by the Polars optimization tree.
    # Output the full, final Polars DataFrame.
df_polars_final = (
    lf_analytical_base.collect()
    .select("sensor_id", "timestamp", "status", "latitude", "voltage", "risk_profile",
            "voltage_rank_in_sensor")
)

# Print the final Polars LazyFrame optimization network and the collected DataFrame
print("----------------------------------FINAL PART 2---------------------------------")
print("\n-------------------------------------------------------------------------------\n")
print(f"Part 2: Polars LazyFrame Optimization Explanation:\n {lf_analytical_base.explain()}")
print("\n-------------------------------------------------------------------------------\n")
print(f"Part 2: Polars Final DataFrame:\n {df_polars_final}\n")

# ================================================================================================
# END PART 2
# ================================================================================================



# ================================================================================================
# END OF PIPELINE SCRIPT DOCUMENT
# ================================================================================================