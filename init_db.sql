-- ============================================================
--  Database initialisation — Flink AE Sensor Processing
-- ============================================================

-- ── Streaming: per-event anomaly alerts ──────────────────────
CREATE TABLE IF NOT EXISTS ae_sensor_events (
    id                SERIAL PRIMARY KEY,
    event_ts          TEXT,
    machine_id        TEXT,
    machine_type      TEXT,
    temperature       DOUBLE PRECISION,
    vibration         DOUBLE PRECISION,
    sound_level       DOUBLE PRECISION,
    power_consumption DOUBLE PRECISION,
    operating_hours   DOUBLE PRECISION,
    error_count       DOUBLE PRECISION,
    recon_error       DOUBLE PRECISION,  -- Autoencoder reconstruction error
    ae_anomaly        INTEGER,           -- 1 = AE flagged anomaly
    latent_z1         DOUBLE PRECISION,  -- Latent space coordinate 1
    latent_z2         DOUBLE PRECISION,  -- Latent space coordinate 2
    rule_fault        INTEGER,           -- 1 = legacy rule-based fault
    inserted_at       TIMESTAMP DEFAULT NOW()
);

-- ── Streaming: 30-second windowed aggregates per machine type ─
CREATE TABLE IF NOT EXISTS ae_window_stats (
    id               SERIAL PRIMARY KEY,
    window_end       TEXT,
    machine_type     TEXT,
    reading_count    INTEGER,
    avg_recon_error  DOUBLE PRECISION,
    anomaly_count    INTEGER,
    avg_temperature  DOUBLE PRECISION,
    avg_vibration    DOUBLE PRECISION,
    avg_sound_level  DOUBLE PRECISION,
    inserted_at      TIMESTAMP DEFAULT NOW()
);

-- ── Batch: enriched per-row results ──────────────────────────
CREATE TABLE IF NOT EXISTS ae_batch_results (
    id                SERIAL PRIMARY KEY,
    machine_id        TEXT,
    machine_type      TEXT,
    temperature       DOUBLE PRECISION,
    vibration         DOUBLE PRECISION,
    sound_level       DOUBLE PRECISION,
    power_consumption DOUBLE PRECISION,
    operating_hours   DOUBLE PRECISION,
    error_count       DOUBLE PRECISION,
    recon_error       DOUBLE PRECISION,
    ae_anomaly        BOOLEAN,
    rule_fault        BOOLEAN,
    agreement         BOOLEAN,           -- AE and rule agree?
    inserted_at       TIMESTAMP DEFAULT NOW()
);

-- ── Batch: summary per machine type ──────────────────────────
CREATE TABLE IF NOT EXISTS ae_batch_summary (
    id                   SERIAL PRIMARY KEY,
    machine_type         TEXT,
    total_readings       BIGINT,
    ae_anomaly_count     BIGINT,
    rule_fault_count     BIGINT,
    agreement_count      BIGINT,
    avg_recon_error      DOUBLE PRECISION,
    max_recon_error      DOUBLE PRECISION,
    avg_temperature      DOUBLE PRECISION,
    avg_vibration        DOUBLE PRECISION,
    avg_sound_level      DOUBLE PRECISION,
    inserted_at          TIMESTAMP DEFAULT NOW()
);

-- ── Useful analysis views ─────────────────────────────────────

-- AE vs rule comparison by machine type
CREATE OR REPLACE VIEW ae_vs_rule_comparison AS
SELECT
    machine_type,
    total_readings,
    ae_anomaly_count,
    rule_fault_count,
    agreement_count,
    ROUND(100.0 * ae_anomaly_count  / NULLIF(total_readings,0), 2) AS ae_anomaly_rate_pct,
    ROUND(100.0 * rule_fault_count  / NULLIF(total_readings,0), 2) AS rule_fault_rate_pct,
    ROUND(100.0 * agreement_count   / NULLIF(total_readings,0), 2) AS agreement_rate_pct,
    ROUND(avg_recon_error::numeric, 6)                             AS avg_recon_error,
    ROUND(max_recon_error::numeric, 6)                             AS max_recon_error
FROM ae_batch_summary
ORDER BY ae_anomaly_count DESC;

-- Anomaly-only stream events
CREATE OR REPLACE VIEW stream_anomalies AS
SELECT
    event_ts, machine_id, machine_type,
    ROUND(temperature::numeric, 2)   AS temp_c,
    ROUND(vibration::numeric, 2)     AS vib_mms,
    ROUND(sound_level::numeric, 2)   AS sound_db,
    ROUND(recon_error::numeric, 6)   AS recon_error,
    ROUND(latent_z1::numeric, 4)     AS z1,
    ROUND(latent_z2::numeric, 4)     AS z2,
    rule_fault,
    inserted_at
FROM ae_sensor_events
WHERE ae_anomaly = 1
ORDER BY inserted_at DESC;

-- Recent 30-second windows
CREATE OR REPLACE VIEW recent_window_stats AS
SELECT
    window_end, machine_type, reading_count,
    anomaly_count,
    ROUND(100.0 * anomaly_count / NULLIF(reading_count,0), 1) AS anomaly_rate_pct,
    ROUND(avg_recon_error::numeric, 6)  AS avg_recon_error,
    ROUND(avg_temperature::numeric, 2)  AS avg_temp,
    ROUND(avg_vibration::numeric, 2)    AS avg_vib,
    ROUND(avg_sound_level::numeric, 2)  AS avg_sound
FROM ae_window_stats
ORDER BY inserted_at DESC
LIMIT 50;

-- Notify on startup
DO $$
BEGIN
    RAISE NOTICE 'Flink AE Sensor Processing DB initialised at %', NOW();
END$$;
