<#
.SYNOPSIS
    Demo script — Apache Flink Autoencoder IoT Anomaly Detection
.DESCRIPTION
    Automates the full demo lifecycle:
      1. Start the stack
      2. Wait for the autoencoder to finish training
      3. Launch the streaming job inside the Flink client container
      4. Tail anomaly output and query PostgreSQL results
#>

param(
    [switch]$Stop,
    [switch]$Logs,
    [switch]$QueryDB,
    [switch]$StreamJob,
    [switch]$BatchJob
)

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host "`n$("=" * 60)" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host "$("=" * 60)`n" -ForegroundColor Cyan
}

# ── Stop ─────────────────────────────────────────────────────────────────────
if ($Stop) {
    Write-Header "Stopping all containers"
    docker-compose down -v
    exit 0
}

# ── Show logs ─────────────────────────────────────────────────────────────────
if ($Logs) {
    Write-Header "Showing ae-trainer logs"
    docker-compose logs ae-trainer
    exit 0
}

# ── Query DB ─────────────────────────────────────────────────────────────────
if ($QueryDB) {
    Write-Header "Querying PostgreSQL results"

    Write-Host "[1] Stream anomalies (last 10)" -ForegroundColor Yellow
    docker exec flink-postgres psql -U flink_user -d flink_db -c `
        "SELECT event_ts, machine_type, ROUND(temperature::numeric,1) AS temp, ROUND(recon_error::numeric,5) AS err FROM stream_anomalies LIMIT 10;"

    Write-Host "`n[2] Window stats (last 5 windows)" -ForegroundColor Yellow
    docker exec flink-postgres psql -U flink_user -d flink_db -c `
        "SELECT * FROM recent_window_stats LIMIT 5;"

    Write-Host "`n[3] AE vs Rule comparison" -ForegroundColor Yellow
    docker exec flink-postgres psql -U flink_user -d flink_db -c `
        "SELECT * FROM ae_vs_rule_comparison;"
    exit 0
}

# ── Run streaming job ─────────────────────────────────────────────────────────
if ($StreamJob) {
    Write-Header "Submitting Streaming Job to Flink"
    docker exec -it flink-client-pyflink `
        python /app/flink-job/streaming_sensor_analyzer.py
    exit 0
}

# ── Run batch job ─────────────────────────────────────────────────────────────
if ($BatchJob) {
    Write-Header "Submitting Batch Job to Flink"
    docker exec -it flink-client-pyflink `
        python /app/flink-job/batch_sensor_analyzer.py
    exit 0
}

# ── Full demo start ───────────────────────────────────────────────────────────
Write-Header "Apache Flink — Autoencoder IoT Anomaly Detection DEMO"

Write-Host "[1/5] Building and starting all services …" -ForegroundColor Green
docker-compose up -d --build

Write-Host "`n[2/5] Waiting for PostgreSQL to be healthy …" -ForegroundColor Green
$retries = 0
while ($retries -lt 20) {
    $status = docker inspect --format='{{.State.Health.Status}}' flink-postgres 2>$null
    if ($status -eq "healthy") { break }
    Write-Host "      Postgres not ready yet ($retries/20) …"
    Start-Sleep -Seconds 3
    $retries++
}
Write-Host "      PostgreSQL is ready!" -ForegroundColor Green

Write-Host "`n[3/5] Waiting for autoencoder training to complete …" -ForegroundColor Green
$retries = 0
while ($retries -lt 60) {
    $state = docker inspect --format='{{.State.Status}}' ae-trainer 2>$null
    if ($state -eq "exited") {
        $exit_code = docker inspect --format='{{.State.ExitCode}}' ae-trainer
        if ($exit_code -eq "0") {
            Write-Host "      Autoencoder trained successfully!" -ForegroundColor Green
            break
        } else {
            Write-Host "      ERROR: ae-trainer exited with code $exit_code" -ForegroundColor Red
            docker-compose logs ae-trainer
            exit 1
        }
    }
    Write-Host "      Training in progress ($retries/60) …"
    Start-Sleep -Seconds 5
    $retries++
}

Write-Host "`n[4/5] Flink Web UI is available at: http://localhost:8081" -ForegroundColor Yellow

Write-Host "`n[5/5] Starting streaming job …" -ForegroundColor Green
Write-Host "      Run this command in a separate terminal to see live anomalies:" -ForegroundColor Cyan
Write-Host "      docker exec -it flink-client-pyflink python /app/flink-job/streaming_sensor_analyzer.py" -ForegroundColor White

Write-Host "`n── Query results ─────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "  .\demo.ps1 -QueryDB       — query PostgreSQL for results"
Write-Host "  .\demo.ps1 -StreamJob     — run streaming job"
Write-Host "  .\demo.ps1 -BatchJob      — run batch job on historical CSV"
Write-Host "  .\demo.ps1 -Logs          — show autoencoder training logs"
Write-Host "  .\demo.ps1 -Stop          — stop and remove all containers"
Write-Host ""
