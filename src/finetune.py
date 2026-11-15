"""Week 1/3: ModernBERT-base fine-tuning (bf16, max_len 128, bs 16 x accum 2, <=3 epochs,
best epoch selected by validation macro-F1). --pilot N runs N optimiser steps and logs resources."""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import MODEL_DIR, RESULTS, core_metrics, load_splits, pick_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--pilot", type=int, default=0)
ap.add_argument("--tag", default=None)
ap.add_argument("--save_model", action="store_true")
a = ap.parse_args()
MODEL = "answerdotai/ModernBERT-base"
tag = a.tag or (f"pilot{a.pilot}" if a.pilot else f"lr{a.lr:g}_s{a.seed}")
OUT = RESULTS / "modernbert" / tag; OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
tr, va, te, labels = load_splits()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)}).to(dev)


def collate(batch):
    texts, ys = zip(*batch)
    enc = tok(list(texts), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
    enc["labels"] = torch.tensor(ys)
    return enc


g = torch.Generator().manual_seed(a.seed)
train_dl = DataLoader(list(zip(tr.norm, tr.label)), batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(texts):
    model.eval(); out = []
    for i in range(0, len(texts), 128):
        enc = tok(list(texts[i:i + 128]), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.float().softmax(-1).cpu())
    model.train(); return torch.cat(out).numpy()


steps_per_epoch = math.ceil(len(train_dl) / a.accum)
total = a.pilot or steps_per_epoch * a.epochs
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
torch.cuda.reset_peak_memory_stats()
model.train(); step, t_start, log = 0, time.time(), []
best = {"val_macro_f1": -1}
step_times = []
for ep in range(a.epochs):
    for i, batch in enumerate(train_dl):
        if i % a.accum == 0: t0 = time.time()
        batch = batch.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / a.accum
        loss.backward()
        if (i + 1) % a.accum == 0 or i + 1 == len(train_dl):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            torch.cuda.synchronize(); step_times.append(time.time() - t0)
            if step % 50 == 0: print(f"step {step}/{total} loss {loss.item() * a.accum:.4f}", flush=True)
            if a.pilot and step >= a.pilot: break
    if a.pilot: break
    pv = predict(va.norm.tolist()); m = core_metrics(pv, va.label.values)
    rec = {"epoch": ep + 1, "step": step, **{f"val_{k}": v for k, v in m.items() if k != "n"}}
    log.append(rec); print(rec, flush=True)
    if m["macro_f1"] > best["val_macro_f1"]:  # best-epoch selection on val macro-F1 (fixed epochs, no early stopping)
        best = {**rec, "val_macro_f1": m["macro_f1"]}
        np.save(OUT / "val_probs.npy", pv); np.save(OUT / "test_probs.npy", predict(te.norm.tolist()))
        if a.save_model:  # checkpoints go to /scratch (home has a small quota)
            MD = MODEL_DIR / tag; model.save_pretrained(MD); tok.save_pretrained(MD)

res = {"args": vars(a), "gpu": torch.cuda.get_device_name(), "steps": step,
       "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
       "median_step_s": round(float(np.median(step_times[5:] or step_times)), 4),
       "wall_s": round(time.time() - t_start, 1), "epoch_log": log}
if a.pilot:
    res["steps_per_epoch"] = steps_per_epoch
    res["extrapolated_3ep_train_min"] = round(res["median_step_s"] * steps_per_epoch * 3 / 60, 2)
else:
    res["best"] = best
    res["threshold"] = pick_threshold(np.load(OUT / "val_probs.npy"), va.label.values)
(OUT / "run.json").write_text(json.dumps(res, indent=2)); print(json.dumps({k: v for k, v in res.items() if k != "epoch_log"}))
