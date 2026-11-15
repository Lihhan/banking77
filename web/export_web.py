"""Build the browser demo assets: fp16 ONNX ModernBERT (for transformers.js) and the TF-IDF model as binary.
Run from the project root:  python web/export_web.py   (needs onnx, onnxruntime; writes web/build/)"""
import json, os, pickle, shutil
import numpy as np, torch, pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
P = Path(__file__).resolve().parents[1]; W = P / "web" / "build"
SRC = Path(os.environ.get("BANK77_MODEL_DIR", P / "models")) / "final_s42"
(W / "model/onnx").mkdir(parents=True, exist_ok=True); (W / "static").mkdir(parents=True, exist_ok=True)
labels = json.loads((P/"data/processed/labels.json").read_text())
# --- model files for transformers.js
for f in ["config.json", "tokenizer.json"]: shutil.copy(SRC/f, W/"model"/f)
tc = json.loads((SRC/"tokenizer_config.json").read_text()); tc["tokenizer_class"] = "PreTrainedTokenizerFast"
for k in ["backend", "is_local", "local_files_only"]: tc.pop(k, None)
tc["model_max_length"] = 128
(W/"model/tokenizer_config.json").write_text(json.dumps(tc, indent=2))
cfg = json.loads((SRC/"config.json").read_text()); cfg["id2label"] = {str(i): l for i, l in enumerate(labels)}
cfg["label2id"] = {l: i for i, l in enumerate(labels)}; (W/"model/config.json").write_text(json.dumps(cfg, indent=2))
tok = AutoTokenizer.from_pretrained(SRC)
m = AutoModelForSequenceClassification.from_pretrained(SRC, attn_implementation="eager").eval()
enc = tok(["my transfer is still pending", "hi"], padding=True, return_tensors="pt")
torch.onnx.export(m, (enc["input_ids"], enc["attention_mask"]), str(W/"model/onnx/model.onnx"),
    input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17, dynamo=False,
    dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}})
# int8 quantisation cost ~2 points of accuracy, so the demo uses fp16 (matches PyTorch: 93.38% test acc)
from onnxruntime.transformers import float16
import onnx
m16 = float16.convert_float_to_float16(onnx.load(str(W/"model/onnx/model.onnx")), keep_io_types=True,
                                       op_block_list=["Cast"] + float16.DEFAULT_OP_BLOCK_LIST, force_fp16_initializers=True)
onnx.save(m16, str(W/"model/onnx/model_fp16.onnx"))
# --- TF-IDF to binary
pipe = pickle.load(open(P/"results/tfidf/model.pkl", "rb"))
meta = {"labels": labels, "parts": []}; arrays = []
for name, v in pipe["tfidf"].transformer_list:
    vocab = sorted(v.vocabulary_, key=v.vocabulary_.get)
    meta["parts"].append({"name": name, "analyzer": v.analyzer, "ngram": list(v.ngram_range), "vocab": vocab})
    arrays.append(v.idf_.astype(np.float32))
lr = pipe["lr"]; meta["n_features"] = int(lr.coef_.shape[1])
arrays += [lr.coef_.astype(np.float32).ravel(), lr.intercept_.astype(np.float32)]
np.concatenate(arrays).tofile(W/"static/tfidf.bin")
(W/"static/tfidf_meta.json").write_text(json.dumps(meta, separators=(",", ":")))
print("done")
