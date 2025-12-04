"""
Baseline evaluation for Black-Scholes-style jump diffusion.

This script simulates jump-diffusion paths using the dataset metadata
(drift/vol/jump params) and reports the MSE vs. the saved dataset.
It avoids the `JumpSDEIto` API (not available in some torchsde builds) and
uses an Euler-style simulator with Poisson jumps.
"""

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch

# repo paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TRAINING_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_data")


def _get_latest_dataset_id(dataset: str) -> int:
    overview = os.path.join(TRAINING_DATA_PATH, "dataset_overview.csv")
    if not os.path.exists(overview):
        return None
    df = pd.read_csv(overview, index_col=0)
    df = df.loc[df["name"] == dataset]
    if len(df) == 0:
        return None
    return int(df["id"].max())


def load_dataset_tensors(dataset: str, dataset_id: int, device):
    """Load dataset tensors directly from saved files; if dataset_id is None, pick latest."""
    if dataset_id is None:
        dataset_id = _get_latest_dataset_id(dataset)
        if dataset_id is None:
            raise ValueError(f"No dataset found for {dataset}. Please generate it first.")
    base = os.path.join(TRAINING_DATA_PATH, f"{dataset}-{dataset_id}")
    data_file = os.path.join(base, "data.npy")
    metadata_file = os.path.join(base, "metadata.txt")
    if not os.path.exists(data_file) or not os.path.exists(metadata_file):
        raise ValueError(f"Dataset files not found at {base}")
    with open(metadata_file, "r") as f:
        hp = json.load(f)
    with open(data_file, "rb") as f:
        stock_paths = torch.tensor(np.load(f), dtype=torch.float32, device=device)
        _ = np.load(f)  # observed_dates, unused
        _ = np.load(f)  # nb_obs, unused
        # ignore further arrays if present
    dt = float(hp["dt"])
    return stock_paths, dt, hp, dataset_id


def simulate_jump_diffusion(
        y0: torch.Tensor, dt: float, steps: int,
        params: Tuple[float, float, float, float, float],
        compensate_jumps: bool = True, device: torch.device = torch.device("cpu")):
    """
    Euler-style simulation of jump-diffusion with log-normal jumps.
    y0: [batch, dim]
    returns: [batch, dim, steps]
    """
    mu, vol, lam, j_mean, j_std = params
    batch, dim = y0.shape
    paths = torch.zeros(batch, dim, steps, device=device)
    paths[:, :, 0] = y0
    kappa = torch.exp(torch.tensor(j_mean + 0.5 * j_std ** 2, device=device)) - 1.0
    drift = mu
    if compensate_jumps:
        drift = drift - lam * kappa.item()
    drift = torch.tensor(drift, device=device)
    vol = torch.tensor(vol, device=device)
    lam = torch.tensor(lam, device=device)
    j_mean = torch.tensor(j_mean, device=device)
    j_std = torch.tensor(j_std, device=device)

    for k in range(1, steps):
        prev = paths[:, :, k - 1]
        # Brownian increment
        dW = torch.randn(batch, dim, device=device) * (dt ** 0.5)
        # Poisson jump counts
        jump_counts = torch.poisson(lam * dt * torch.ones(batch, dim, device=device))
        jump_multiplier = torch.ones(batch, dim, device=device)
        has_jump = jump_counts > 0
        if has_jump.any():
            max_count = int(jump_counts.max().item())
            for j in range(max_count):
                mask = jump_counts > j
                if not mask.any():
                    continue
                jump_draw = torch.exp(j_mean + j_std * torch.randn(batch, dim, device=device))
                jump_multiplier = torch.where(mask, jump_multiplier * jump_draw, jump_multiplier)
        drift_term = (drift - 0.5 * vol ** 2) * dt
        diffusion_term = vol * dW
        paths[:, :, k] = prev * torch.exp(drift_term + diffusion_term) * jump_multiplier
    return paths


def evaluate(dataset: str, dataset_id: int, compensate_jumps: bool, device: str):
    device = torch.device(device)
    paths, dt, hp, dataset_id = load_dataset_tensors(dataset, dataset_id, device)
    _, _, steps = paths.shape
    params = (
        float(hp["drift"]),
        float(hp["volatility"]),
        float(hp.get("jump_intensity", 0.0)),
        float(hp.get("jump_mean", -0.2)),
        float(hp.get("jump_std", 0.25)),
    )
    sim = simulate_jump_diffusion(
        y0=paths[:, :, 0], dt=dt, steps=steps, params=params,
        compensate_jumps=compensate_jumps, device=device)
    mse = torch.mean((sim - paths) ** 2).item()
    return mse, dataset_id


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate jump-diffusion baseline (Euler + Poisson jumps) "
                    "on BlackScholesJumpDiffusion datasets.")
    parser.add_argument("--dataset", type=str,
                        default="BlackScholesJumpDiffusion")
    parser.add_argument("--dataset_id", type=int, default=None,
                        help="If None, latest dataset is used.")
    parser.add_argument("--no_compensate", action="store_true",
                        help="Disable drift compensation for jumps.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    compensate = not args.no_compensate
    mse, dataset_id = evaluate(
        dataset=args.dataset, dataset_id=args.dataset_id,
        compensate_jumps=compensate, device=args.device)
    print(f"MSE vs dataset ({args.dataset}, id={dataset_id}): {mse:.6f}")


if __name__ == "__main__":
    main()
