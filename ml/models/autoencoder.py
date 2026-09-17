"""PyTorch MLP autoencoder over the flattened ``[60, 5]`` window tensor.

Architecture: ``300 -> 64 -> 16 -> 64 -> 300`` with ReLU and a linear output
(inputs are z-scored per channel with the train-set scaler). Reconstruction MSE
is calibrated to ``s_ae`` in ``[0, 1]`` using the 98th percentile of the nominal
holdout error as 1.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

WINDOW_N = 60
N_CHANNELS = 5
INPUT_DIM = WINDOW_N * N_CHANNELS

HIDDEN_1 = 64
LATENT = 16
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
CALIB_PERCENTILE = 98.0
RANDOM_STATE = 42

WEIGHTS_NAME = "autoencoder.pt"
META_NAME = "autoencoder_meta.json"
SCALER_NAME = "scaler.joblib"


class MLPAutoencoder(nn.Module):
    """300 -> 64 -> 16 -> 64 -> 300 MLP autoencoder."""

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, LATENT),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def fit_scaler(x_seq: np.ndarray) -> StandardScaler:
    """Per-channel mean/std scaler fitted on ``(N, 60, 5)`` nominal windows."""
    flat = np.asarray(x_seq, dtype=float).reshape(-1, N_CHANNELS)
    scaler = StandardScaler()
    scaler.fit(flat)
    scaler.scale_ = np.where(scaler.scale_ < 1e-9, 1e-9, scaler.scale_)
    return scaler


def transform(x_seq: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """z-score ``(N, 60, 5)`` (or ``(60, 5)``) and flatten to ``(N, 300)``."""
    arr = np.asarray(x_seq, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    n = arr.shape[0]
    scaled = scaler.transform(arr.reshape(-1, N_CHANNELS))
    return scaled.reshape(n, -1)


@dataclass
class AutoencoderScorer:
    """Autoencoder + scaler + error calibration."""

    model: MLPAutoencoder
    scaler: StandardScaler
    err_p98: float
    input_dim: int = INPUT_DIM

    # --- scoring ---------------------------------------------------------- #
    def errors(self, x_seq: np.ndarray) -> np.ndarray:
        flat = transform(x_seq, self.scaler)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(flat.astype(np.float32))
            recon = self.model(tensor)
            err = torch.mean((tensor - recon) ** 2, dim=1)
        return err.numpy().astype(float)

    def score(self, x_seq: np.ndarray) -> np.ndarray:
        denom = self.err_p98 if self.err_p98 > 0 else 1e-9
        return np.clip(self.errors(x_seq) / denom, 0.0, 1.0)

    def score_one(self, window: np.ndarray) -> float:
        return float(self.score(np.asarray(window, dtype=float))[0])

    # --- construction ----------------------------------------------------- #
    @classmethod
    def fit(
        cls,
        x_seq_nominal: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        holdout_frac: float = 0.2,
        random_state: int = RANDOM_STATE,
        verbose: bool = False,
    ) -> "AutoencoderScorer":
        arr = np.asarray(x_seq_nominal, dtype=float)
        if arr.ndim != 3:
            raise ValueError("x_seq_nominal must be (N, 60, 5)")
        torch.manual_seed(random_state)
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(arr))
        n_hold = max(1, int(round(holdout_frac * len(arr))))
        hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]
        if len(fit_idx) == 0:
            fit_idx, hold_idx = idx, idx

        scaler = fit_scaler(arr[fit_idx])
        train_x = torch.from_numpy(transform(arr[fit_idx], scaler).astype(np.float32))

        model = MLPAutoencoder(input_dim=train_x.shape[1])
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(train_x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimiser.step()
                total += float(loss.item()) * batch.shape[0]
            if verbose:
                print(f"[ae] epoch {epoch + 1}/{epochs} loss={total / max(1, len(dataset)):.6f}")

        scorer = cls(model=model, scaler=scaler, err_p98=1.0, input_dim=train_x.shape[1])
        holdout_err = scorer.errors(arr[hold_idx])
        err_p98 = float(np.percentile(holdout_err, CALIB_PERCENTILE))
        scorer.err_p98 = err_p98 if err_p98 > 0 else 1e-9
        return scorer

    # --- persistence ------------------------------------------------------ #
    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        weights = out / WEIGHTS_NAME
        meta = out / META_NAME
        scaler_path = out / SCALER_NAME
        torch.save(self.model.state_dict(), weights)
        meta.write_text(
            json.dumps(
                {
                    "input_dim": int(self.input_dim),
                    "window_n": WINDOW_N,
                    "n_channels": N_CHANNELS,
                    "hidden_1": HIDDEN_1,
                    "latent": LATENT,
                    "err_p98": float(self.err_p98),
                },
                indent=2,
            )
            + "\n"
        )
        joblib.dump(self.scaler, scaler_path)
        return {"weights": weights, "meta": meta, "scaler": scaler_path}

    @classmethod
    def load(cls, out_dir: str | Path) -> "AutoencoderScorer":
        out = Path(out_dir)
        meta = json.loads((out / META_NAME).read_text())
        model = MLPAutoencoder(input_dim=int(meta["input_dim"]))
        model.load_state_dict(torch.load(out / WEIGHTS_NAME, map_location="cpu"))
        model.eval()
        scaler = joblib.load(out / SCALER_NAME)
        return cls(
            model=model,
            scaler=scaler,
            err_p98=float(meta["err_p98"]),
            input_dim=int(meta["input_dim"]),
        )
