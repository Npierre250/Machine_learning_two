"""
Batch Sensor Analyzer — Autoencoder Anomaly Detection on Historical Data
========================================================================
Apache Flink (PyFlink) batch job that:

1. Reads the historical CSV file (factory_sensor_simulator_2040.csv).
2. Scores every row with the trained autoencoder.
3. Compares AE results with the original rule-based approach.
4. Writes enriched results + comparison summary to PostgreSQL.

Run AFTER the autoencoder has been trained (autoencoder_model.json exists).
"""

import os
import json
import numpy as np
from datetime import datetime

from pyflink.table import (
    EnvironmentSettings, TableEnvironment, DataTypes,
    TableDescriptor, Schema,
)
from pyflink.table.expressions import col, lit
from pyflink.table.udf import udf


# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_DIR   = os.environ.get("MODEL_DIR", "/app/autoencoder")
MODEL_PATH  = os.path.join(MODEL_DIR, "autoencoder_model.json")
STATS_PATH  = os.path.join(MODEL_DIR, "normalisation_stats.json")
THRESH_PATH = os.path.join(MODEL_DIR, "anomaly_threshold.json")

# Load globally so UDFs can reference without re-reading per row
with open(MODEL_PATH)  as f: _MODEL_DICT = json.load(f)
with open(STATS_PATH)  as f: _STATS      = json.load(f)
with open(THRESH_PATH) as f: _THRESH     = json.load(f)

_MIN       = np.array(_STATS["min"])
_MAX       = np.array(_STATS["max"])
_RNG_NORM  = np.where((_MAX - _MIN) > 0, _MAX - _MIN, 1.0)
_THRESHOLD = _THRESH["threshold"]
_LAYERS    = _MODEL_DICT["layers"]


def _activate(z, act):
    if act == "relu":
        return np.maximum(0, z)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return z


def _forward(x):
    h = x
    for layer in _LAYERS:
        W = np.array(layer["W"])
        b = np.array(layer["b"])
        h = _activate(h @ W + b, layer["activation"])
    return h


def _recon_error(temp, vib, sound, power, hours, errors):
    raw = np.array([temp, vib, sound, power, hours, errors])
    x   = ((raw - _MIN) / _RNG_NORM).reshape(1, -1)
    return float(np.mean((_forward(x) - x) ** 2))


# ── PyFlink UDFs ──────────────────────────────────────────────────────────────

@udf(result_type=DataTypes.DOUBLE())
def ae_recon_error(temp, vib, sound, power, hours, errors):
    """Returns autoencoder reconstruction error for one sensor reading."""
    return _recon_error(float(temp), float(vib), float(sound),
                        float(power), float(hours), float(errors))


@udf(result_type=DataTypes.BOOLEAN())
def ae_is_anomaly(temp, vib, sound, power, hours, errors):
    """Returns True if reading is an autoencoder-detected anomaly."""
    return _recon_error(float(temp), float(vib), float(sound),
                        float(power), float(hours), float(errors)) > _THRESHOLD


@udf(result_type=DataTypes.BOOLEAN())
def rule_is_fault(temp, vib, sound):
    """Original rule-based fault detection from the base project."""
    return float(temp) > 75.0 or float(vib) > 20.0 or float(sound) > 80.0


# ── PostgreSQL config ─────────────────────────────────────────────────────────

PG_URL  = "jdbc:postgresql://postgres:5432/flink_db"
PG_USER = os.environ.get("PG_USER", "flink_user")
PG_PASS = os.environ.get("PG_PASS", "flink_password")

CSV_PATH = os.environ.get(
    "CSV_PATH",
    "/app/data/factory_sensor_simulator_2040.csv"
)


def main():
    print("=" * 60)
    print("  Flink Batch Job — Autoencoder IoT Anomaly Detection")
    print("=" * 60)
    print(f"  CSV input  : {CSV_PATH}")
    print(f"  Model dir  : {MODEL_DIR}")
    print(f"  AE threshold: {_THRESHOLD:.6f}")
    print("=" * 60)

    settings = EnvironmentSettings.in_batch_mode()
    t_env    = TableEnvironment.create(settings)

    t_env.get_config().set("parallelism.default", "2")

    # Register UDFs
    t_env.create_temporary_function("ae_recon_error", ae_recon_error)
    t_env.create_temporary_function("ae_is_anomaly",  ae_is_anomaly)
    t_env.create_temporary_function("rule_is_fault",  rule_is_fault)

    # ── Source: CSV file ──────────────────────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TEMPORARY TABLE sensor_csv (
            machine_id        STRING,
            machine_type      STRING,
            temperature       DOUBLE,
            vibration         DOUBLE,
            sound_level       DOUBLE,
            power_consumption DOUBLE,
            operating_hours   DOUBLE,
            error_count       DOUBLE
        ) WITH (
            'connector' = 'filesystem',
            'path'      = '{CSV_PATH}',
            'format'    = 'csv',
            'csv.field-delimiter' = ',',
            'csv.ignore-parse-errors' = 'true',
            'csv.allow-comments' = 'true'
        )
    """)

    # ── Sink 1: enriched per-row results ──────────────────────────────────
    t_env.execute_sql(f"""
        CREATE TEMPORARY TABLE ae_batch_results (
            machine_id        STRING,
            machine_type      STRING,
            temperature       DOUBLE,
            vibration         DOUBLE,
            sound_level       DOUBLE,
            power_consumption DOUBLE,
            operating_hours   DOUBLE,
            error_count       DOUBLE,
            recon_error       DOUBLE,
            ae_anomaly        BOOLEAN,
            rule_fault        BOOLEAN,
            agreement         BOOLEAN
        ) WITH (
            'connector'  = 'jdbc',
            'url'        = '{PG_URL}',
            'table-name' = 'ae_batch_results',
            'username'   = '{PG_USER}',
            'password'   = '{PG_PASS}'
        )
    """)

    print("\n[1/3] Scoring all rows with autoencoder …")
    t_env.execute_sql("""
        INSERT INTO ae_batch_results
        SELECT
            machine_id,
            machine_type,
            temperature,
            vibration,
            sound_level,
            power_consumption,
            operating_hours,
            error_count,
            ae_recon_error(temperature, vibration, sound_level,
                           power_consumption, operating_hours, error_count)
                AS recon_error,
            ae_is_anomaly(temperature, vibration, sound_level,
                          power_consumption, operating_hours, error_count)
                AS ae_anomaly,
            rule_is_fault(temperature, vibration, sound_level)
                AS rule_fault,
            ae_is_anomaly(temperature, vibration, sound_level,
                          power_consumption, operating_hours, error_count)
            = rule_is_fault(temperature, vibration, sound_level)
                AS agreement
        FROM sensor_csv
    """).wait()
    print("      Done.")

    # ── Sink 2: aggregated comparison per machine type ────────────────────
    t_env.execute_sql(f"""
        CREATE TEMPORARY TABLE ae_batch_summary (
            machine_type         STRING,
            total_readings       BIGINT,
            ae_anomaly_count     BIGINT,
            rule_fault_count     BIGINT,
            agreement_count      BIGINT,
            avg_recon_error      DOUBLE,
            max_recon_error      DOUBLE,
            avg_temperature      DOUBLE,
            avg_vibration        DOUBLE,
            avg_sound_level      DOUBLE
        ) WITH (
            'connector'  = 'jdbc',
            'url'        = '{PG_URL}',
            'table-name' = 'ae_batch_summary',
            'username'   = '{PG_USER}',
            'password'   = '{PG_PASS}'
        )
    """)

    print("\n[2/3] Writing summary statistics per machine type …")
    t_env.execute_sql("""
        INSERT INTO ae_batch_summary
        SELECT
            machine_type,
            COUNT(*)                    AS total_readings,
            SUM(CASE WHEN ae_anomaly  THEN 1 ELSE 0 END) AS ae_anomaly_count,
            SUM(CASE WHEN rule_fault  THEN 1 ELSE 0 END) AS rule_fault_count,
            SUM(CASE WHEN agreement   THEN 1 ELSE 0 END) AS agreement_count,
            AVG(recon_error)            AS avg_recon_error,
            MAX(recon_error)            AS max_recon_error,
            AVG(temperature)            AS avg_temperature,
            AVG(vibration)              AS avg_vibration,
            AVG(sound_level)            AS avg_sound_level
        FROM ae_batch_results
        GROUP BY machine_type
    """).wait()
    print("      Done.")

    print("\n[3/3] Batch job complete.")
    print(f"      AE threshold used: {_THRESHOLD:.6f}")
    print("      Results in tables: ae_batch_results, ae_batch_summary")
    print("\nQuery example:")
    print("  SELECT machine_type, ae_anomaly_count, rule_fault_count,")
    print("         agreement_count::float/total_readings AS agreement_rate")
    print("  FROM ae_batch_summary;")


if __name__ == "__main__":
    main()
