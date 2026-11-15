"""Week 2: word+char TF-IDF + class-weighted logistic regression; predeclared grid on val macro-F1."""
import itertools
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from common import RESULTS, SEED, core_metrics, load_splits, pick_threshold

tr, va, te, labels = load_splits()
OUT = RESULTS / "tfidf"; OUT.mkdir(parents=True, exist_ok=True)

# predeclared grid: 2 word n-gram x 2 char n-gram x 4 C = 16 configs
GRID = {"word_ngram": [(1, 1), (1, 2)], "char_ngram": [(2, 4), (2, 5)], "C": [1, 3, 10, 30]}


def build(word_ngram, char_ngram, C):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=word_ngram, lowercase=True,
                                 token_pattern=r"(?u)\b\w+(?:'\w+)?\b", sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, lowercase=True,
                                 sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    return Pipeline([("tfidf", feats), ("lr", clf)])


rows, best = [], None
for wn, cn, C in itertools.product(*GRID.values()):
    t0 = time.time()
    m = build(wn, cn, C).fit(tr.norm, tr.label)
    met = core_metrics(m.predict_proba(va.norm), va.label.values)
    row = {"word_ngram": str(wn), "char_ngram": str(cn), "C": C, **{f"val_{k}": v for k, v in met.items() if k != "n"},
           "fit_s": round(time.time() - t0, 1)}
    rows.append(row); print(row, flush=True)
    if best is None or met["macro_f1"] > best[0]:
        best = (met["macro_f1"], m, row)
pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)

_, model, cfg = best
pv = model.predict_proba(va.norm)
thr = pick_threshold(pv, va.label.values)
pt = model.predict_proba(te.norm)
np.save(OUT / "val_probs.npy", pv); np.save(OUT / "test_probs.npy", pt)
pickle.dump(model, open(OUT / "model.pkl", "wb"))
json.dump({"selected": cfg, "threshold": thr}, open(OUT / "selected.json", "w"), indent=2)
print("SELECTED", cfg, "threshold", thr)
