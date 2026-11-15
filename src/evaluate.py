"""Week 4: official-test evaluation (full + overlap-free), selective metrics, per-class, confusion, errors."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

from common import FIGS, RESULTS, core_metrics, load_splits, per_class, risk_coverage, selective

tr, va, te, labels = load_splits()
y, yv = te.label.values, va.label.values
nov = ~te.in_overlap.values
rng = np.random.default_rng(42)

MODELS = {
    "TF-IDF + LogReg": ("tfidf", json.load(open(RESULTS / "tfidf/selected.json"))["threshold"]),
    "ModernBERT-base": ("modernbert/final_s42", json.load(open(RESULTS / "modernbert/final_s42/run.json"))["threshold"]),
}


def boot_ci(probs, y, n=2000):
    pred = probs.argmax(1); idx = np.arange(len(y)); acc, f1 = [], []
    from sklearn.metrics import f1_score
    for _ in range(n):
        s = rng.choice(idx, len(idx)); acc.append((pred[s] == y[s]).mean()); f1.append(f1_score(y[s], pred[s], average="macro"))
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))], [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]


rows, P = [], {}
for name, (d, thr) in MODELS.items():
    pt, pv = np.load(RESULTS / d / "test_probs.npy"), np.load(RESULTS / d / "val_probs.npy")
    P[name] = pt
    for subset, mask in [("full official test", np.ones_like(y, bool)), ("overlap-free test", nov)]:
        r = {"model": name, "subset": subset, **core_metrics(pt[mask], y[mask]), **selective(pt[mask], y[mask], thr)}
        if subset.startswith("full"):
            r["acc_ci95"], r["f1_ci95"] = boot_ci(pt, y)
        rows.append(r)
    rows.append({"model": name, "subset": "validation", **core_metrics(pv, yv), **selective(pv, yv, thr)})
    per_class(pt, y, labels).to_csv(RESULTS / f"per_class_{d.split('/')[0]}.csv", index=False)
    risk_coverage(pt, y).to_csv(RESULTS / f"risk_coverage_{d.split('/')[0]}.csv", index=False)
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "main_metrics.csv", index=False)
print(res.drop(columns=["acc_ci95", "f1_ci95"]).round(4).to_string())
print(res[res.subset.str.startswith("full")][["model", "acc_ci95", "f1_ci95"]].to_string())

# seed robustness for ModernBERT
seed_rows = []
for s in [42, 13, 7]:
    pt = np.load(RESULTS / f"modernbert/final_s{s}/test_probs.npy")
    seed_rows.append({"seed": s, **core_metrics(pt, y)})
seeds = pd.DataFrame(seed_rows); seeds.to_csv(RESULTS / "modernbert_seeds.csv", index=False)
print(seeds.round(4).to_string()); print("mean", seeds.mean().round(4).to_dict(), "sd", seeds.std().round(4).to_dict())

# McNemar exact test (paired) ModernBERT vs TF-IDF on full test
a = P["ModernBERT-base"].argmax(1) == y; b = P["TF-IDF + LogReg"].argmax(1) == y
n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
mc = {"bert_right_tfidf_wrong": n10, "tfidf_right_bert_wrong": n01,
      "p_value": float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)}
print("McNemar", mc)

# matched-coverage comparison: selective accuracy when each model auto-routes the same 90% / 80% of queries
matched = []
for cov in [1.0, 0.95, 0.9, 0.8]:
    for name, pt in P.items():
        conf = pt.max(1); k = int(round(cov * len(y))); keep = np.argsort(-conf)[:k]
        matched.append({"coverage": cov, "model": name, "selective_accuracy": float((pt[keep].argmax(1) == y[keep]).mean())})
matched = pd.DataFrame(matched).pivot(index="coverage", columns="model", values="selective_accuracy")
matched.to_csv(RESULTS / "matched_coverage.csv"); print(matched.round(4))

# confusion pairs + errors
summary = {"mcnemar": mc, "thresholds": {k: v[1] for k, v in MODELS.items()}}
for name, pt in P.items():
    pred = pt.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(77)); np.fill_diagonal(cm, 0)
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i, j in zip(*np.unravel_index(np.argsort(-cm, None)[:12], cm.shape))]
    summary[f"top_confusions::{name}"] = pairs
    print(name, "top confusions:"); [print("  ", p) for p in pairs]
json.dump(summary, open(RESULTS / "eval_summary.json", "w"), indent=2)

pt = P["ModernBERT-base"]; pred = pt.argmax(1); top3 = np.argsort(-pt, 1)[:, :3]
err = te.assign(pred=[labels[p] for p in pred], conf=pt.max(1).round(3),
                gold_rank=[int(np.where(np.argsort(-pt[i]) == y[i])[0][0]) + 1 for i in range(len(y))],
                tfidf_pred=[labels[p] for p in P["TF-IDF + LogReg"].argmax(1)])[pred != y]
err = err.sort_values("conf", ascending=False)
err[["norm", "category", "pred", "conf", "gold_rank", "tfidf_pred"]].to_csv(RESULTS / "modernbert_errors.csv", index=False)
print("ModernBERT errors:", len(err))

# ---- figures
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for name, col in zip(P, ["#f4845f", "#4361ee"]):
    rc = risk_coverage(P[name], y)
    ax[0].plot(rc.coverage, rc.selective_accuracy, color=col, lw=2, label=name)
    thr = MODELS[name][1]; s = selective(P[name], y, thr)
    ax[0].scatter([s["coverage"]], [s["selective_accuracy"]], color=col, zorder=5, s=50, edgecolor="k")
    ax[0].annotate(f"τ={thr:.2f}", (s["coverage"], s["selective_accuracy"]), textcoords="offset points", xytext=(-40, 8), color=col)
ax[0].set(xlabel="Coverage (auto-routed fraction)", ylabel="Selective accuracy", title="Coverage vs. selective accuracy (test)")
ax[0].legend(loc="lower left"); ax[0].grid(alpha=.3)
pc_b = pd.read_csv(RESULTS / "per_class_modernbert.csv"); pc_t = pd.read_csv(RESULTS / "per_class_tfidf.csv")
cnt = tr.category.value_counts()
ax[1].scatter(cnt[pc_t.intent].values, pc_t.f1, s=14, color="#f4845f", label="TF-IDF + LogReg")
ax[1].scatter(cnt[pc_b.intent].values, pc_b.f1, s=14, color="#4361ee", label="ModernBERT-base")
ax[1].set_xscale("log"); ax[1].set(xlabel="Training examples (log)", ylabel="Test F1 per intent", title="Per-intent F1 vs. training support")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / "selective_and_perclass.png", dpi=150)

cm = confusion_matrix(y, pt.argmax(1), labels=range(77))
fig, ax = plt.subplots(figsize=(11, 10)); ax.imshow(np.log1p(cm), cmap="Blues")
ax.set_xticks(range(77)); ax.set_yticks(range(77)); ax.set_xticklabels(labels, rotation=90, fontsize=5); ax.set_yticklabels(labels, fontsize=5)
ax.set(xlabel="Predicted", ylabel="Gold", title="ModernBERT-base confusion matrix (test, log scale)")
fig.tight_layout(); fig.savefig(FIGS / "confusion_modernbert.png", dpi=150)
