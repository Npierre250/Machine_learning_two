"""
Autoencoder Training Script for Industrial IoT Sensor Anomaly Detection
=======================================================================
Trains a lightweight autoencoder on NORMAL sensor data (no faults).
The trained model is saved and later loaded by the Flink streaming job
to compute reconstruction error on live sensor readings.

Architecture:
  Input (6 features) → Encoder (6→4→2) → Latent (2) → Decoder (2→4→6) → Output (6)
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from datetime import datetime

# ── Pure-NumPy Autoencoder ────────────────────────────────────────────────────

class DenseLayer:
    """Single fully-connected layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        # Xavier initialisation
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation
        # Cache for backprop
        self._x = None
        self._z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(0, self._z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(self._z, -50, 50)))
        elif self.activation == "linear":
            return self._z
        raise ValueError(f"Unknown activation: {self.activation}")

    def backward(self, grad_out: np.ndarray, lr: float):
        if self.activation == "relu":
            grad_act = (self._z > 0).astype(float)
        elif self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(self._z, -50, 50)))
            grad_act = s * (1 - s)
        else:  # linear
            grad_act = np.ones_like(self._z)

        delta = grad_out * grad_act
        grad_W = self._x.T @ delta
        grad_b = delta.sum(axis=0)
        grad_x = delta @ self.W.T

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_x


class Autoencoder:
    """
    Symmetric autoencoder for sensor anomaly detection.
    Input/Output dimension = 6 sensor features.
    Latent dimension = 2 (bottleneck forces meaningful compression).
    """

    def __init__(self, input_dim: int = 6, latent_dim: int = 2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        hidden = 4
        # Encoder
        self.enc1 = DenseLayer(input_dim, hidden, "relu")
        self.enc2 = DenseLayer(hidden, latent_dim, "relu")
        # Decoder
        self.dec1 = DenseLayer(latent_dim, hidden, "relu")
        self.dec2 = DenseLayer(hidden, input_dim, "sigmoid")
        self.layers = [self.enc1, self.enc2, self.dec1, self.dec2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = self.enc1.forward(x)
        return self.enc2.forward(h)

    def train_step(self, x: np.ndarray, lr: float) -> float:
        recon = self.forward(x)
        loss = np.mean((recon - x) ** 2)          # MSE reconstruction loss
        grad = 2 * (recon - x) / x.shape[0]       # dL/d(recon)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return loss

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Returns per-sample MSE reconstruction error."""
        recon = self.forward(x)
        return np.mean((recon - x) ** 2, axis=1)

    def to_dict(self) -> dict:
        """Serialize weights to plain dict for JSON/pickle persistence."""
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "layers": [
                {"W": l.W.tolist(), "b": l.b.tolist(), "activation": l.activation}
                for l in self.layers
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Autoencoder":
        ae = cls(d["input_dim"], d["latent_dim"])
        for layer, ld in zip(ae.layers, d["layers"]):
            layer.W = np.array(ld["W"])
            layer.b = np.array(ld["b"])
            layer.activation = ld["activation"]
        return ae


# ── Data helpers ─────────────────────────────────────────────────────────────

FEATURES = [
    "temperature",
    "vibration",
    "sound_level",
    "power_consumption",
    "operating_hours",
    "error_count",
]

# Fault thresholds from the original project (used only to filter training data)
FAULT_THRESHOLDS = {
    "temperature": 75.0,
    "vibration": 20.0,
    "sound_level": 80.0,
}


def generate_normal_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic NORMAL (healthy) factory sensor data.
    Mimics the same ranges as simulate_factory_data.py but keeps
    readings below fault thresholds so the autoencoder learns normal behaviour.
    """
    rng = np.random.default_rng(seed)
    machine_types = ["CNC_Machine", "Conveyor_Belt", "Hydraulic_Press",
                     "Robotic_Arm", "Pump_Station"]

    rows = []
    for _ in range(n_samples):
        mtype = rng.choice(machine_types)
        # Healthy ranges (deliberately below fault thresholds)
        temp   = rng.uniform(20, 70)          # threshold 75 °C
        vib    = rng.uniform(0,  18)          # threshold 20 mm/s
        sound  = rng.uniform(40, 76)          # threshold 80 dB
        power  = rng.uniform(1,  50)
        hours  = rng.uniform(0,  8760)
        errors = rng.integers(0, 5)
        rows.append({
            "machine_type": mtype,
            "temperature": temp,
            "vibration": vib,
            "sound_level": sound,
            "power_consumption": power,
            "operating_hours": hours,
            "error_count": float(errors),
            "is_fault": 0,
        })
    return pd.DataFrame(rows)


def normalise(df: pd.DataFrame, stats: dict = None):
    """Min-max normalise the 6 feature columns. Returns (X, stats)."""
    X = df[FEATURES].values.astype(np.float64)
    if stats is None:
        stats = {
            "min": X.min(axis=0).tolist(),
            "max": X.max(axis=0).tolist(),
        }
    mn = np.array(stats["min"])
    mx = np.array(stats["max"])
    rng = np.where((mx - mn) > 0, mx - mn, 1.0)
    return (X - mn) / rng, stats


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    n_samples: int = 5000,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 0.005,
    latent_dim: int = 2,
    output_dir: str = "/app/autoencoder",
):
    print("=" * 60)
    print("  Autoencoder Training — Industrial IoT Sensor Anomaly")
    print("=" * 60)

    print(f"\n[1/4] Generating {n_samples} normal sensor samples …")
    df = generate_normal_data(n_samples)
    X, stats = normalise(df)
    print(f"      Features: {FEATURES}")
    print(f"      Shape: {X.shape}")

    print(f"\n[2/4] Building autoencoder  (latent_dim={latent_dim}) …")
    ae = Autoencoder(input_dim=len(FEATURES), latent_dim=latent_dim)

    print(f"\n[3/4] Training ({epochs} epochs, batch={batch_size}, lr={lr}) …")
    rng = np.random.default_rng(0)
    history = []
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X), batch_size):
            batch = X[idx[start:start + batch_size]]
            epoch_loss += ae.train_step(batch, lr)
            n_batches += 1
        avg = epoch_loss / n_batches
        history.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{epochs}  loss={avg:.6f}")

    print("\n[4/4] Computing anomaly threshold on normal data …")
    errors = ae.reconstruction_error(X)
    # Threshold = mean + 3*std on normal data (catches 99.7% of normal under Gaussian assumption)
    threshold = float(np.mean(errors) + 3 * np.std(errors))
    print(f"      Recon error — mean={np.mean(errors):.6f}  std={np.std(errors):.6f}")
    print(f"      Anomaly threshold (μ+3σ): {threshold:.6f}")

    # ── Persist artefacts ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    model_path     = os.path.join(output_dir, "autoencoder_model.json")
    stats_path     = os.path.join(output_dir, "normalisation_stats.json")
    threshold_path = os.path.join(output_dir, "anomaly_threshold.json")
    history_path   = os.path.join(output_dir, "training_history.json")

    with open(model_path, "w") as f:
        json.dump(ae.to_dict(), f)

    with open(stats_path, "w") as f:
        json.dump({**stats, "features": FEATURES}, f)

    with open(threshold_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "trained_at": datetime.utcnow().isoformat(),
        }, f)

    with open(history_path, "w") as f:
        json.dump({"loss": history}, f)

    print(f"\n✓ Model saved       → {model_path}")
    print(f"✓ Norm stats saved  → {stats_path}")
    print(f"✓ Threshold saved   → {threshold_path}")
    print(f"✓ History saved     → {history_path}")
    print("\nTraining complete.\n")
    return ae, stats, threshold


if __name__ == "__main__":
    train()
