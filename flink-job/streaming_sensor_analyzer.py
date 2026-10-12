"""
Streaming Sensor Analyzer — Autoencoder Anomaly Detection
==========================================================
Apache Flink (PyFlink) DataStream job that:

1. Reads live sensor readings from a TCP socket (from the data generator).
2. Parses and normalises each reading using pre-computed stats.
3. Runs the trained autoencoder to compute reconstruction error per reading.
4. Flags readings whose error exceeds the learned anomaly threshold.
5. Aggregates 30-second tumbling-window statistics per machine type.
6. Writes both per-event anomaly alerts and windowed stats to PostgreSQL.

Sensor features used by the autoencoder (must match training):
  temperature, vibration, sound_level, power_consumption,
  operating_hours, error_count
"""

import os
import json
import numpy as np
from datetime import datetime

from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.functions import MapFunction, KeyedProcessFunction
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.datastream.functions import ProcessWindowFunction
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from pyflink.datastream.connectors import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions


# ── Autoencoder weights loader (pure-NumPy, no ML framework needed) ───────────

class AutoencoderInference:
    """
    Lightweight autoencoder inference engine.
    Loads the JSON weights produced by train_autoencoder.py and
    runs forward passes on numpy arrays — zero extra dependencies.
    """

    def __init__(self, model_path: str, stats_path: str, threshold_path: str):
        with open(model_path) as f:
            model_dict = json.load(f)
        with open(stats_path) as f:
            stats = json.load(f)
        with open(threshold_path) as f:
            thresh_dict = json.load(f)

        self.layers = model_dict["layers"]
        self.features = stats["features"]
        self.min = np.array(stats["min"])
        self.max = np.array(stats["max"])
        self.threshold = thresh_dict["threshold"]
        self.rng = np.where((self.max - self.min) > 0, self.max - self.min, 1.0)

    def _activate(self, z: np.ndarray, act: str) -> np.ndarray:
        if act == "relu":
            return np.maximum(0, z)
        elif act == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        return z  # linear

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.layers:
            W = np.array(layer["W"])
            b = np.array(layer["b"])
            h = self._activate(h @ W + b, layer["activation"])
        return h

    def normalise(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.min) / self.rng

    def reconstruction_error(self, raw: np.ndarray) -> float:
        x = self.normalise(raw.reshape(1, -1))
        recon = self._forward(x)
        return float(np.mean((recon - x) ** 2))

    def is_anomaly(self, raw: np.ndarray) -> bool:
        return self.reconstruction_error(raw) > self.threshold

    def encode(self, raw: np.ndarray) -> list:
        """Return 2-D latent representation (useful for logging)."""
        x = self.normalise(raw.reshape(1, -1))
        # Forward through encoder layers only (first 2)
        h = x
        for layer in self.layers[:2]:
            W = np.array(layer["W"])
            b = np.array(layer["b"])
            h = self._activate(h @ W + b, layer["activation"])
        return h.flatten().tolist()


# ── Model paths (Docker volume /app/autoencoder) ──────────────────────────────

MODEL_DIR   = os.environ.get("MODEL_DIR", "/app/autoencoder")
MODEL_PATH  = os.path.join(MODEL_DIR, "autoencoder_model.json")
STATS_PATH  = os.path.join(MODEL_DIR, "normalisation_stats.json")
THRESH_PATH = os.path.join(MODEL_DIR, "anomaly_threshold.json")

# ── PostgreSQL config ─────────────────────────────────────────────────────────

PG_URL  = "jdbc:postgresql://postgres:5432/flink_db"
PG_USER = os.environ.get("PG_USER", "flink_user")
PG_PASS = os.environ.get("PG_PASS", "flink_password")

# ── Socket source config ──────────────────────────────────────────────────────

FLINK_HOST = os.environ.get("FLINK_HOST", "jobmanager")
FLINK_PORT = int(os.environ.get("FLINK_PORT", "9000"))


# ── Flink Map function: parse CSV line → enriched tuple ──────────────────────

class ParseAndScoreFunction(MapFunction):
    """
    Parses a raw CSV sensor line, runs the autoencoder, and returns a tuple:
    (timestamp, machine_id, machine_type, temperature, vibration, sound_level,
     power_consumption, operating_hours, error_count,
     recon_error, is_anomaly, latent_z1, latent_z2, rule_fault)
    """

    def __init__(self, model_path, stats_path, thresh_path):
        self._model_path  = model_path
        self._stats_path  = stats_path
        self._thresh_path = thresh_path
        self._ae = None   # lazy-loaded inside the task (serialisation-safe)

    def open(self, runtime_context):
        self._ae = AutoencoderInference(
            self._model_path, self._stats_path, self._thresh_path
        )

    def map(self, line: str):
        try:
            parts = line.strip().split(",")
            if len(parts) < 8:
                return None

            machine_id   = parts[0].strip()
            machine_type = parts[1].strip()
            temperature  = float(parts[2])
            vibration    = float(parts[3])
            sound_level  = float(parts[4])
            power        = float(parts[5])
            op_hours     = float(parts[6])
            error_count  = float(parts[7])

            # ── Autoencoder scoring ──────────────────────────────────────
            features = np.array([
                temperature, vibration, sound_level,
                power, op_hours, error_count
            ])
            recon_error = self._ae.reconstruction_error(features)
            ae_anomaly  = recon_error > self._ae.threshold
            latent      = self._ae.encode(features)
            z1, z2      = latent[0], latent[1]

            # ── Legacy rule-based fault (kept for comparison) ────────────
            rule_fault = (
                temperature > 75.0 or
                vibration   > 20.0 or
                sound_level > 80.0
            )

            ts = datetime.utcnow().isoformat()
            return (
                ts, machine_id, machine_type,
                temperature, vibration, sound_level, power, op_hours, error_count,
                recon_error, int(ae_anomaly), z1, z2, int(rule_fault)
            )
        except Exception:
            return None


# ── Flink Window function: 30-second stats per machine type ──────────────────

class WindowStatsFunction(ProcessWindowFunction):
    """
    Aggregates readings inside a 30-second tumbling window per machine_type.
    Outputs (window_end, machine_type, count, avg_recon_error,
             anomaly_count, avg_temp, avg_vib, avg_sound).
    """

    def process(self, key, context, elements):
        items = list(elements)
        if not items:
            return
        window_end       = datetime.utcfromtimestamp(
            context.window().end / 1000
        ).isoformat()
        count            = len(items)
        avg_recon_error  = float(np.mean([e[9]  for e in items]))
        anomaly_count    = int(sum(e[10] for e in items))
        avg_temp         = float(np.mean([e[3]  for e in items]))
        avg_vib          = float(np.mean([e[4]  for e in items]))
        avg_sound        = float(np.mean([e[5]  for e in items]))
        yield (
            window_end, key, count,
            avg_recon_error, anomaly_count,
            avg_temp, avg_vib, avg_sound
        )


# ── Main Flink job ────────────────────────────────────────────────────────────

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_stream_time_characteristic(TimeCharacteristic.ProcessingTime)
    env.set_parallelism(2)

    # Add PostgreSQL JDBC driver (must be in /opt/flink/lib/)
    env.add_jars("file:///opt/flink/lib/flink-connector-jdbc.jar",
                 "file:///opt/flink/lib/postgresql-42.7.3.jar")

    print("=" * 60)
    print("  Flink Streaming Job — Autoencoder IoT Anomaly Detection")
    print("=" * 60)
    print(f"  Connecting to data socket  : {FLINK_HOST}:{FLINK_PORT}")
    print(f"  Autoencoder model dir      : {MODEL_DIR}")
    print(f"  PostgreSQL                 : {PG_URL}")
    print("=" * 60)

    # ── Source: TCP socket ─────────────────────────────────────────────────
    raw_stream = env.socket_text_stream(FLINK_HOST, FLINK_PORT)

    # ── Map: parse + AE score ─────────────────────────────────────────────
    scored = (
        raw_stream
        .map(ParseAndScoreFunction(MODEL_PATH, STATS_PATH, THRESH_PATH),
             output_type=Types.TUPLE([
                 Types.STRING(),   # ts
                 Types.STRING(),   # machine_id
                 Types.STRING(),   # machine_type
                 Types.DOUBLE(),   # temperature
                 Types.DOUBLE(),   # vibration
                 Types.DOUBLE(),   # sound_level
                 Types.DOUBLE(),   # power
                 Types.DOUBLE(),   # op_hours
                 Types.DOUBLE(),   # error_count
                 Types.DOUBLE(),   # recon_error
                 Types.INT(),      # ae_anomaly
                 Types.DOUBLE(),   # z1
                 Types.DOUBLE(),   # z2
                 Types.INT(),      # rule_fault
             ]))
        .filter(lambda x: x is not None)
    )

    # ── Sink 1: per-event anomaly alerts → PostgreSQL ─────────────────────
    scored.add_sink(
        JdbcSink.sink(
            """
            INSERT INTO ae_sensor_events
              (event_ts, machine_id, machine_type,
               temperature, vibration, sound_level,
               power_consumption, operating_hours, error_count,
               recon_error, ae_anomaly, latent_z1, latent_z2, rule_fault)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            JdbcStatementBuilder(14),
            JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                .with_url(PG_URL)
                .with_driver_name("org.postgresql.Driver")
                .with_user_name(PG_USER)
                .with_password(PG_PASS)
                .build(),
            JdbcExecutionOptions.builder()
                .with_batch_interval_ms(1000)
                .with_batch_size(200)
                .with_max_retries(3)
                .build(),
        )
    )

    # ── Sink 2: 30-second windowed stats per machine_type → PostgreSQL ────
    windowed_stats = (
        scored
        .key_by(lambda x: x[2])   # key by machine_type
        .window(TumblingProcessingTimeWindows.of(Time.seconds(30)))
        .process(WindowStatsFunction(),
                 output_type=Types.TUPLE([
                     Types.STRING(),  # window_end
                     Types.STRING(),  # machine_type
                     Types.INT(),     # count
                     Types.DOUBLE(),  # avg_recon_error
                     Types.INT(),     # anomaly_count
                     Types.DOUBLE(),  # avg_temp
                     Types.DOUBLE(),  # avg_vib
                     Types.DOUBLE(),  # avg_sound
                 ]))
    )

    windowed_stats.add_sink(
        JdbcSink.sink(
            """
            INSERT INTO ae_window_stats
              (window_end, machine_type, reading_count,
               avg_recon_error, anomaly_count,
               avg_temperature, avg_vibration, avg_sound_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            JdbcStatementBuilder(8),
            JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                .with_url(PG_URL)
                .with_driver_name("org.postgresql.Driver")
                .with_user_name(PG_USER)
                .with_password(PG_PASS)
                .build(),
            JdbcExecutionOptions.builder()
                .with_batch_interval_ms(5000)
                .with_batch_size(50)
                .with_max_retries(3)
                .build(),
        )
    )

    # ── Side-effect: print anomalies to stdout for live demo ──────────────
    scored.filter(lambda x: x[10] == 1).map(
        lambda x: (
            f"🚨 ANOMALY | {x[0]} | {x[2]:20s} | "
            f"T={x[3]:.1f}°C V={x[4]:.1f}mm/s S={x[5]:.1f}dB | "
            f"recon_err={x[9]:.5f} | z=({x[11]:.3f},{x[12]:.3f})"
        ),
        output_type=Types.STRING()
    ).print()

    env.execute("Flink AE Streaming Sensor Analyzer")


# ── Thin JDBC statement builder helper ───────────────────────────────────────

class JdbcStatementBuilder:
    """Simple wrapper to set PreparedStatement parameters from a tuple."""

    def __init__(self, n_params: int):
        self._n = n_params

    def __call__(self, stmt, record):
        for i, val in enumerate(record[: self._n]):
            if isinstance(val, float):
                stmt.setDouble(i + 1, val)
            elif isinstance(val, int):
                stmt.setInt(i + 1, val)
            else:
                stmt.setString(i + 1, str(val))


if __name__ == "__main__":
    main()
