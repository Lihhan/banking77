"""Shared inference for the demo and demo checks: both models return top-k (intent, score)."""
import json
import pickle

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import MODEL_DIR, PROC, RESULTS, ROOT, normalize

LABELS = json.loads((PROC / "labels.json").read_text())
GLOSSARY = json.loads((ROOT / "data" / "glossary.json").read_text())
THRESHOLDS = {
    "ModernBERT-base": json.loads((RESULTS / "modernbert/final_s42/run.json").read_text())["threshold"],
    "TF-IDF + LogReg baseline": json.loads((RESULTS / "tfidf/selected.json").read_text())["threshold"],
}


class TfidfRouter:
    def __init__(self):
        self.m = pickle.load(open(RESULTS / "tfidf/model.pkl", "rb"))

    def probs(self, text):
        return self.m.predict_proba([normalize(text)])[0]


class BertRouter:
    def __init__(self, path=MODEL_DIR / "final_s42"):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path).to(self.dev).eval()

    @torch.no_grad()
    def probs(self, text):
        enc = self.tok([normalize(text)], truncation=True, max_length=128, return_tensors="pt").to(self.dev)
        return self.m(**enc).logits.float().softmax(-1)[0].cpu().numpy()


def topk(probs, k=3):
    idx = np.argsort(-probs)[:k]
    return [(LABELS[i], float(probs[i])) for i in idx]
