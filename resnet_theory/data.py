"""MAGIC gamma telescope data (UCI, Bock et al. 2004): 19,020 events, 10 features."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MAGIC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
MAGIC_MD5 = "bb3888bd588c4b15b24351218f74491e"
MAGIC_COLUMNS = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class",
]


@dataclass
class Dataset:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor

    @property
    def dim(self) -> int:
        return int(self.X_train.shape[1])


def magic_file(path: str | Path) -> Path:
    """Path of the data file, downloaded from the UCI repository on first use and checked
    against the MD5 checksum of the file used in the thesis."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MAGIC data from {MAGIC_URL} to {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MAGIC_URL, tmp)
        digest = hashlib.md5(tmp.read_bytes()).hexdigest()
        if digest != MAGIC_MD5:
            tmp.unlink()
            raise RuntimeError(f"Downloaded MAGIC file has MD5 {digest}, expected {MAGIC_MD5}.")
        tmp.replace(path)
    return path


def load_magic(path: str | Path, val_fraction: float, seed: int) -> Dataset:
    """Random train/validation split set by ``seed``, standardised with training statistics.

    Labels: 1 for gamma (signal), 0 for hadron (background).
    """
    df = pd.read_csv(magic_file(path), names=MAGIC_COLUMNS)
    X = df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    y = (df["class"].to_numpy() == "g").astype(np.float32).reshape(-1, 1)

    perm = np.random.default_rng(int(seed)).permutation(X.shape[0])
    n_val = max(1, int(round(val_fraction * X.shape[0])))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

    as_tensor = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return Dataset(as_tensor(X_train), as_tensor(y_train), as_tensor(X_val), as_tensor(y_val))
