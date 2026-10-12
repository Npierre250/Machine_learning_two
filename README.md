# Apache Flink — Industrial IoT Sensor Processing with Autoencoder Anomaly Detection

[![Apache Flink](https://img.shields.io/badge/Apache%20Flink-1.18.1-orange)](https://flink.apache.org/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)](https://www.postgresql.org/)

A real-time industrial IoT pipeline built on **Apache Flink** that uses a
**pure-NumPy Autoencoder** to detect machine anomalies — no external ML
framework required. The autoencoder learns what "normal" sensor readings look
like, then flags deviations during live streaming and historical batch processing.

---

## What's New vs the Base Project

| Feature | Base Project | This Project |
|---|---|---|
| Anomaly detection | Rule-based thresholds | **Autoencoder reconstruction error** |
| ML model | None | Symmetric AE (6→4→2→4→6) |
| Latent space | N/A | 2-D (loggable, visualisable) |
| Batch comparison | Rule stats only | **AE vs rule agreement analysis** |
| DB schema | 2 tables | **5 tables + 3 analysis views** |
| Training | N/A | Auto-trains on first `docker-compose up` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Network                        │
│                                                             │
│  ┌──────────────┐     ┌─────────────────────────────────┐  │
│  │  ae-trainer  │────▶│  /app/autoencoder/ (volume)      │  │
│  │  (runs once) │     │  ├── autoencoder_model.json       │  │
│  └──────────────┘     │  ├── normalisation_stats.json     │  │
│                        │  └── anomaly_threshold.json       │  │
│                        └──────────────┬──────────────────┘  │
│                                       │ read                 │
│  ┌──────────────┐   socket   ┌────────▼──────────────────┐  │
│  │data-generator│──────────▶│     flink-client           │  │
│  │  (CSV/TCP)   │  TCP:9000  │  streaming_sensor_analyzer │  │
│  └──────────────┘            │  batch_sensor_analyzer     │  │
│                              └────────┬──────────────────┘  │
│                                       │ JDBC                 │
│  ┌──────────────────────────────────┐ │                      │
│  │  Flink Cluster                   │ │                      │
│  │  JobManager  +  3×TaskManager    │◀┘                      │
│  │  Web UI: localhost:8081           │                        │
│  └──────────────────────────────────┘                        │
│                                                             │
│  ┌──────────────────────────────────┐                        │
│  │  PostgreSQL :5435                │                        │
│  │  ├── ae_sensor_events            │                        │
│  │  ├── ae_window_stats             │                        │
│  │  ├── ae_batch_results            │                        │
│  │  ├── ae_batch_summary            │                        │
│  │  └── views: stream_anomalies,    │                        │
│  │            recent_window_stats,  │                        │
│  │            ae_vs_rule_comparison │                        │
│  └──────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Autoencoder Design

```
Sensor input (6 features)
  temperature, vibration, sound_level,
  power_consumption, operating_hours, error_count

         ┌──────────┐
  x ──▶  │  Enc-1   │  6 → 4  ReLU
         └────┬─────┘
              │
         ┌────▼─────┐
         │  Enc-2   │  4 → 2  ReLU   ← Latent space (z1, z2)
         └────┬─────┘
              │
         ┌────▼─────┐
         │  Dec-1   │  2 → 4  ReLU
         └────┬─────┘
              │
         ┌────▼─────┐
  x̂ ◀──  │  Dec-2   │  4 → 6  Sigmoid
         └──────────┘

Loss = MSE(x, x̂)
Threshold = μ_normal + 3σ_normal  (computed after training on healthy data)
```

Training is done on **healthy sensor data only**. At inference time, anomalous
readings produce high reconstruction error because the encoder cannot compress
them into the "normal" latent region.

---

## Sensor Features & Fault Thresholds

| Feature | Normal Range | Fault Threshold |
|---|---|---|
| Temperature | 20–72 °C | > 75 °C |
| Vibration | 0–19 mm/s | > 20 mm/s |
| Sound Level | 40–78 dB | > 80 dB |
| Power Consumption | 1–50 kW | — |
| Operating Hours | 0–8760 h | — |
| Error Count | 0–3 | — |

The autoencoder learns from **all 6 features together**, allowing it to catch
subtle multi-sensor correlations that single-threshold rules miss.

---

## Quick Start

### Requirements
- Docker Desktop (≥ 8 GB RAM recommended)
- Docker Compose v2

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/Flink-sensor-processing-ae.git
cd Flink-sensor-processing-ae
```

### 2. Start everything
```bash
docker-compose up -d
```
This will:
- Start PostgreSQL and initialise the schema
- Train the autoencoder (watch with `docker-compose logs ae-trainer`)
- Start the Flink cluster (JobManager + 3 TaskManagers)
- Start the data generator

### 3. Run the streaming job
```bash
docker exec -it flink-client-pyflink bash
python /app/flink-job/streaming_sensor_analyzer.py
```

You'll see live anomaly alerts like:
```
🚨 ANOMALY | 2040-01-01T08:23:11 | CNC_Machine         | T=89.3°C V=8.1mm/s S=62.0dB | recon_err=0.04821 | z=(1.234,0.876)
```

### 4. Check the Flink Web UI
Open **http://localhost:8081** — you'll see your streaming job running with
input/output rates, backpressure indicators, and task parallelism.

### 5. Run the batch job
```bash
# In the same container
python /app/flink-job/batch_sensor_analyzer.py
```

### 6. Query PostgreSQL results
```bash
docker exec -it flink-postgres psql -U flink_user -d flink_db
```
```sql
-- AE vs rule comparison
SELECT * FROM ae_vs_rule_comparison;

-- Recent streaming anomalies
SELECT * FROM stream_anomalies LIMIT 20;

-- 30-second window stats
SELECT * FROM recent_window_stats;
```

---

## Project Structure

```
Flink-sensor-processing-ae/
├── docker-compose.yml                  # Full stack orchestration
├── Dockerfile.client                   # PyFlink client image
├── Dockerfile.generator                # Data generator image
├── Dockerfile.trainer                  # Autoencoder trainer image
├── requirements.txt
├── simulate_factory_data.py            # Sensor data simulator (stream + batch)
├── init_db.sql                         # PostgreSQL schema + analysis views
├── demo.ps1                            # PowerShell demo automation
│
├── autoencoder/
│   └── train_autoencoder.py            # Pure-NumPy AE training (no ML framework)
│       ├── DenseLayer                  # Forward + backprop with NumPy
│       ├── Autoencoder                 # Encoder-Decoder, train_step, to_dict/from_dict
│       └── train()                     # Generates data, trains, saves JSON weights
│
└── flink-job/
    ├── streaming_sensor_analyzer.py    # Flink DataStream job (PyFlink)
    │   ├── AutoencoderInference        # Loads JSON weights, runs forward pass
    │   ├── ParseAndScoreFunction       # Flink MapFunction (parse + AE score)
    │   └── WindowStatsFunction         # 30-second tumbling window aggregation
    └── batch_sensor_analyzer.py        # Flink Table API batch job
        ├── ae_recon_error (UDF)        # Flink UDF wrapping AE inference
        ├── ae_is_anomaly  (UDF)
        └── rule_is_fault  (UDF)
```

---

## Key Commands

```bash
# Start / stop
docker-compose up -d
docker-compose down -v                  # also removes volumes

# Watch autoencoder training
docker-compose logs -f ae-trainer

# Interactive Flink client shell
docker exec -it flink-client-pyflink bash

# Connect to database
docker exec -it flink-postgres psql -U flink_user -d flink_db

# PowerShell demo helpers
.\demo.ps1 -StreamJob
.\demo.ps1 -BatchJob
.\demo.ps1 -QueryDB
.\demo.ps1 -Logs
.\demo.ps1 -Stop
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ae-trainer` exits with code 1 | Run `docker-compose logs ae-trainer` to see the error |
| Flink job can't find model files | Ensure `ae-trainer` completed before client starts |
| Out of memory | Give Docker at least 8 GB RAM in Docker Desktop settings |
| Port 8081 conflict | Change `"8081:8081"` in docker-compose.yml |
| Port 5435 conflict | Change `"5435:5432"` in docker-compose.yml |
| JDBC connector missing | Rebuild Dockerfile.client (`docker-compose build flink-client`) |

---

## License

MIT — see [LICENSE](LICENSE) for details.

Built on top of [MbarekTech/Flink-sensor-processing](https://github.com/MbarekTech/Flink-sensor-processing).
