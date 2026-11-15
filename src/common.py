"""Shared paths, normalisation, splits and metrics for the BANKING77 project."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
SEED = 42
# Fine-tuned checkpoints: set BANK77_MODEL_DIR to override; defaults to <project>/models.
_SCRATCH = Path("/scratch/f008m8h/bank77/models")  # original run location (home quota is small)
MODEL_DIR = Path(os.environ.get("BANK77_MODEL_DIR", _SCRATCH if _SCRATCH.exists() else ROOT / "models"))
WORD_RE = re.compile(r"\w+(?:'\w+)?")  # regex word-token proxy used in the proposal


def normalize(text: str) -> str:
    """Trim and collapse internal whitespace; keep case, punctuation and negation."""
    return re.sub(r"\s+", " ", str(text)).strip()


def word_len(text: str) -> int:
    return len(WORD_RE.findall(text))


def load_splits():
    """Processed splits written by audit.py: train / val / test (full) with norm text + label id."""
    tr = pd.read_csv(PROC / "train.csv")
    va = pd.read_csv(PROC / "val.csv")
    te = pd.read_csv(PROC / "test.csv")
    labels = json.loads((PROC / "labels.json").read_text())
    return tr, va, te, labels


def topk_acc(probs: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def core_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "top3_accuracy": topk_acc(probs, y, 3),
    }


def selective(probs: np.ndarray, y: np.ndarray, thr: float) -> dict:
    conf = probs.max(1)
    keep = conf >= thr
    pred = probs.argmax(1)
    cov = float(keep.mean())
    sel = float((pred[keep] == y[keep]).mean()) if keep.any() else float("nan")
    return {"threshold": float(thr), "coverage": cov, "selective_accuracy": sel,
            "n_flagged": int((~keep).sum()),
            "errors_caught": int(((~keep) & (pred != y)).sum()),
            "errors_total": int((pred != y).sum())}


def pick_threshold(probs: np.ndarray, y: np.ndarray, target_sel_acc: float = 0.98) -> float:
    """Smallest max-softmax threshold whose validation selective accuracy >= target
    (i.e. the highest-coverage operating point meeting the target)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        keep = conf >= t
        if keep.any() and correct[keep].mean() >= target_sel_acc:
            return float(t)
    return 1.0


def per_class(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, probs.argmax(1), labels=range(len(labels)), zero_division=0)
    return pd.DataFrame({"intent": labels, "precision": p, "recall": r, "f1": f, "support": s})


def risk_coverage(probs: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = [selective(probs, y, t) for t in np.round(np.arange(0.0, 1.0001, 0.01), 2)]
    return pd.DataFrame(rows)
