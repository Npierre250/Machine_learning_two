"""
Factory Sensor Data Simulator
==============================
Generates realistic industrial IoT sensor readings and streams them
over a TCP socket to the Flink job.

Machines emit readings every ~0.5 seconds.
~15% of readings are intentional anomalies (exceeding safe thresholds),
so the autoencoder has something interesting to detect in the demo.

Sensor fields (CSV order):
  machine_id, machine_type, temperature, vibration, sound_level,
  power_consumption, operating_hours, error_count
"""

import socket
import time
import random
import argparse
import csv
import os
from datetime import datetime, timedelta

# ── Machine catalogue ─────────────────────────────────────────────────────────

MACHINE_TYPES = {
    "CNC_Machine":     {"temp": (30, 68), "vib": (1, 16),  "sound": (55, 75), "power": (5,  35)},
    "Conveyor_Belt":   {"temp": (20, 60), "vib": (2, 17),  "sound": (50, 70), "power": (1,  15)},
    "Hydraulic_Press": {"temp": (35, 72), "vib": (3, 19),  "sound": (60, 78), "power": (10", 50)},
    "Robotic_Arm":     {"temp": (25, 65), "vib": (1, 15),  "sound": (45, 72), "power": (3,  25)},
    "Pump_Station":    {"temp": (30, 70), "vib": (2, 18),  "sound": (55, 76), "power": (8,  40)},
}

# Fix the syntax error above (typo in original)
MACHINE_TYPES = {
    "CNC_Machine":     {"temp": (30, 68), "vib": (1, 16),  "sound": (55, 75), "power": (5,  35)},
    "Conveyor_Belt":   {"temp": (20, 60), "vib": (2, 17),  "sound": (50, 70), "power": (1,  15)},
    "Hydraulic_Press": {"temp": (35, 72), "vib": (3, 19),  "sound": (60, 78), "power": (10, 50)},
    "Robotic_Arm":     {"temp": (25, 65), "vib": (1, 15),  "sound": (45, 72), "power": (3,  25)},
    "Pump_Station":    {"temp": (30, 70), "vib": (2, 18),  "sound": (55, 76), "power": (8,  40)},
}

MACHINES = [
    {"id": f"{mtype[:3].upper()}-{i:03d}", "type": mtype}
    for mtype, _ in MACHINE_TYPES.items()
    for i in range(1, 4)          # 3 machines of each type = 15 total
]


# ── Single reading generator ──────────────────────────────────────────────────

def generate_reading(machine: dict, fault_probability: float = 0.15) -> str:
    """
    Returns a CSV line for one sensor reading.
    With fault_probability chance the reading is an anomaly
    (one or more sensors exceed safe thresholds).
    """
    mtype  = machine["type"]
    ranges = MACHINE_TYPES[mtype]
    is_fault = random.random() < fault_probability

    temp  = random.uniform(*ranges["temp"])
    vib   = random.uniform(*ranges["vib"])
    sound = random.uniform(*ranges["sound"])

    if is_fault:
        # Spike one or more sensors above threshold
        fault_type = random.choice(["temp", "vib", "sound", "multi"])
        if fault_type in ("temp",  "multi"):
            temp  = random.uniform(76, 105)    # over 75 °C threshold
        if fault_type in ("vib",   "multi"):
            vib   = random.uniform(21,  40)    # over 20 mm/s
        if fault_type in ("sound", "multi"):
            sound = random.uniform(82, 110)    # over 80 dB

    power   = random.uniform(*ranges["power"])
    hours   = random.uniform(0, 8760)
    errors  = random.randint(0, 10) if is_fault else random.randint(0, 3)

    return (
        f"{machine['id']},{mtype},"
        f"{temp:.2f},{vib:.2f},{sound:.2f},"
        f"{power:.2f},{hours:.2f},{errors}"
    )


# ── Stream mode (socket to Flink) ─────────────────────────────────────────────

def run_stream(host: str, port: int, interval: float, fault_prob: float):
    """Opens a TCP socket server and streams readings to Flink."""
    print(f"[Stream] Starting socket server on {host}:{port}")
    print(f"         Fault probability: {fault_prob*100:.0f}%  |  Interval: {interval}s")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print(f"[Stream] Waiting for Flink to connect …")
        conn, addr = srv.accept()
        print(f"[Stream] Flink connected from {addr}")
        count = 0
        try:
            while True:
                machine = random.choice(MACHINES)
                line    = generate_reading(machine, fault_prob)
                conn.sendall((line + "\n").encode())
                count += 1
                if count % 50 == 0:
                    print(f"[Stream] Sent {count} readings …")
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            print("[Stream] Flink disconnected.")


# ── Batch mode (CSV file for batch Flink job) ─────────────────────────────────

def run_batch(output_path: str, n_rows: int, fault_prob: float):
    """Writes a CSV file for the batch Flink job."""
    print(f"[Batch] Generating {n_rows} rows → {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Simulate a timeline spanning 2040 to make it feel like real historical data
    start = datetime(2040, 1, 1)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header as a comment (Flink's csv.allow-comments=true will skip it)
        f.write("# machine_id,machine_type,temperature,vibration,sound_level,"
                "power_consumption,operating_hours,error_count\n")
        for i in range(n_rows):
            machine = MACHINES[i % len(MACHINES)]
            line    = generate_reading(machine, fault_prob)
            f.write(line + "\n")
            if (i + 1) % 10000 == 0:
                print(f"[Batch]  {i+1}/{n_rows} rows written …")
    print(f"[Batch] Done. File: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory sensor data simulator")
    parser.add_argument("--mode",        default=os.environ.get("MODE", "stream"),
                        choices=["stream", "batch"])
    parser.add_argument("--host",        default=os.environ.get("FLINK_HOST", "0.0.0.0"))
    parser.add_argument("--port",        default=int(os.environ.get("FLINK_PORT", "9000")), type=int)
    parser.add_argument("--interval",    default=0.5,    type=float,
                        help="Seconds between readings (stream mode)")
    parser.add_argument("--fault-prob",  default=0.15,   type=float,
                        help="Fraction of readings that are anomalies (0–1)")
    parser.add_argument("--n-rows",      default=50000,  type=int,
                        help="Number of rows to generate (batch mode)")
    parser.add_argument("--output",      default="/app/data/factory_sensor_simulator_2040.csv")
    args = parser.parse_args()

    if args.mode == "stream":
        run_stream(args.host, args.port, args.interval, args.fault_prob)
    else:
        run_batch(args.output, args.n_rows, args.fault_prob)
