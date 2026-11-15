"""Week 1: data audit, duplicate/overlap checks, stratified split, Appendix-A figure."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import FIGS, PROC, RAW, RESULTS, SEED, normalize, word_len

PROC.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
rep = {}
rep["sha256"] = {f: hashlib.sha256((RAW / f).read_bytes()).hexdigest() for f in ["train.csv", "test.csv"]}

tr = pd.read_csv(RAW / "train.csv", keep_default_na=False)
te = pd.read_csv(RAW / "test.csv", keep_default_na=False)
cats = json.loads((RAW / "categories.json").read_text())
labels = sorted(cats)
lid = {l: i for i, l in enumerate(labels)}

rep["rows"] = {"train": len(tr), "test": len(te)}
rep["n_labels"] = {"train": tr.category.nunique(), "test": te.category.nunique(), "categories.json": len(cats)}
rep["labels_match_categories_json"] = set(tr.category) == set(te.category) == set(cats)
rep["empty_fields"] = int((tr.text.str.strip() == "").sum() + (te.text.str.strip() == "").sum()
                          + (tr.category == "").sum() + (te.category == "").sum())

for df in (tr, te):
    df["norm"] = df.text.map(normalize)
    df["wlen"] = df.norm.map(word_len)
    df["label"] = df.category.map(lid)
rep["word_len"] = {s: {"mean": round(float(d.wlen.mean()), 2), "median": float(d.wlen.median()),
                       "max": int(d.wlen.max())} for s, d in [("train", tr), ("test", te)]}
cnt = tr.category.value_counts()
rep["train_support"] = {"min": int(cnt.min()), "min_label": cnt.idxmin(), "max": int(cnt.max()), "max_label": cnt.idxmax()}
rep["test_support_unique"] = sorted(te.category.value_counts().unique().tolist())

# duplicates after normalisation
dup_tr = tr[tr.duplicated("norm", keep="first")]
dup_te = te[te.duplicated("norm", keep="first")]
overlap = sorted(set(tr.norm) & set(te.norm))
ov_rows = []
for t in overlap:
    ov_rows.append({"text": t, "train_labels": sorted(set(tr.category[tr.norm == t])),
                    "test_labels": sorted(set(te.category[te.norm == t]))})
rep["repeated_texts"] = {"train": len(dup_tr), "test": len(dup_te)}
rep["train_label_conflicts_in_dups"] = int((tr.groupby("norm").category.nunique() > 1).sum())
rep["train_test_overlap"] = len(overlap)
rep["overlap_same_label"] = all(r["train_labels"] == r["test_labels"] for r in ov_rows)
rep["overlap_texts"] = ov_rows
rep["repeated_train_texts"] = dup_tr[["norm", "category"]].to_dict("records")
rep["repeated_test_texts"] = dup_te[["norm", "category"]].to_dict("records")

# clean training pool: drop repeated rows, drop overlap texts
pool = tr.drop_duplicates("norm", keep="first")
pool = pool[~pool.norm.isin(overlap)].reset_index(drop=True)
trn, val = train_test_split(pool, test_size=0.10, stratify=pool.label, random_state=SEED)
rep["split"] = {"pool": len(pool), "train": len(trn), "val": len(val),
                "val_min_per_class": int(val.label.value_counts().min()),
                "test_full": len(te), "test_overlap_free": int((~te.norm.isin(overlap)).sum())}

te["in_overlap"] = te.norm.isin(overlap)
cols = ["norm", "category", "label"]
trn[cols].to_csv(PROC / "train.csv", index=False)
val[cols].to_csv(PROC / "val.csv", index=False)
te[cols + ["in_overlap"]].to_csv(PROC / "test.csv", index=False)
(PROC / "labels.json").write_text(json.dumps(labels, indent=1))
(RESULTS / "audit.json").write_text(json.dumps(rep, indent=2))
print(json.dumps({k: v for k, v in rep.items() if k not in ("overlap_texts", "repeated_train_texts", "repeated_test_texts")}, indent=1))
for r in ov_rows: print("OVERLAP:", r)

# ---- Appendix-A style diagnostics figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
c = cnt.sort_values()
ax[0, 0].bar(range(1, 78), c.values, color="#4361ee"); ax[0, 0].set_yscale("log")
ax[0, 0].axhline(40, ls="--", color="#e76f51", label="40 test examples / intent"); ax[0, 0].legend()
ax[0, 0].set(title="1  Class support (train)", xlabel="Intent rank (fewest → most)", ylabel="Examples (log)")
ext = pd.concat([c.head(6), c.tail(6)])
ax[0, 1].barh([l.replace("_", " ") for l in ext.index], ext.values, color=["#f4845f"] * 6 + ["#4361ee"] * 6)
ax[0, 1].set(title="2  Lowest / highest support intents", xlabel="Training examples")
med = tr.groupby("category").wlen.median()
ax[1, 0].scatter(cnt[med.index], med.values, s=14, color="#4361ee"); ax[1, 0].set_xscale("log")
ax[1, 0].set(title="3  Support vs. median query length", xlabel="Training examples (log)", ylabel="Median words")
bins = np.arange(0, 80, 2)
ax[1, 1].hist(tr.wlen, bins=bins, density=True, histtype="step", lw=2, label=f"Train · median {tr.wlen.median():.0f}")
ax[1, 1].hist(te.wlen, bins=bins, density=True, histtype="step", lw=2, label=f"Test · median {te.wlen.median():.0f}")
ax[1, 1].legend(); ax[1, 1].set(title="4  Query length by split", xlabel="Regex word tokens", ylabel="Density")
fig.tight_layout(); fig.savefig(FIGS / "dataset_diagnostics.png", dpi=150)
