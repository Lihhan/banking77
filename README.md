# BANKING77 Intent Router

Accelerated Computational Linguistics (FA26) course project by Lihan Li and Xizi Luo.
The project compares a TF–IDF + logistic-regression baseline with fine-tuned ModernBERT-base on BANKING77 (77 intents). It includes a hosted browser demo (both models run client-side) and a local Streamlit app.

* Results: [RESULTS.md](RESULTS.md). An interactive version is in [report/banking77_report.html](report/banking77_report.html); open it in any browser.
* **Live demo:** https://huggingface.co/spaces/Lihhan/banking77-intent-router. Both models run in the browser, so no install is needed. The first visit downloads the 300 MB fp16 ModernBERT, which the browser then caches. It has a "Challenging cases" panel (confident error, keyword-free paraphrases, adjacent transfer intents, out-of-scope) that shows the expected intent. A backup recording is in [`demo/demo_video.mp4`](demo/demo_video.mp4) (1.5 min).
* Headline: ModernBERT-base reaches 93.38% test accuracy and 0.934 macro-F1. TF–IDF reaches 91.40% and 0.914.

## Layout

| Path | Contents |
|---|---|
| `src/` | All code. Run every script from inside this folder. |
| `data/raw/` | Original PolyAI CSVs and `categories.json` (CC BY 4.0). |
| `data/processed/` | Normalised train / val / test splits and the label list. |
| `data/glossary.json` | One-line plain-language gloss for each of the 77 intents, used by the demo. |
| `results/` | Metrics, per-class tables, predicted probabilities, error coding, demo checks and logs. `results/tfidf/model.pkl` is the trained baseline. |
| `figures/` | Dataset diagnostics, the selective-accuracy and per-intent F1 plot, and the confusion matrix. |
| `report/` | Self-contained HTML results report. |
| `web/` | Hosted demo: a static page (`index.html`), a JS port of the TF–IDF model (`tfidf.js`), and `export_web.py`, which builds the fp16 ONNX model and the TF–IDF binary. The fp16 model matches PyTorch: 93.38% test accuracy, with the same argmax on 99.9% of queries. |
| `models/` | Fine-tuned ModernBERT checkpoint. It ships in a separate zip because of its size, or you can retrain it (step 4 below, about 2 minutes on one GPU). |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA/CPU
pip install -r requirements.txt
```

The demo and the checks need the ModernBERT checkpoint in `models/final_s42/`. Either unzip `banking77_modernbert_final_s42.zip` into the project root, or set `BANK77_MODEL_DIR` to the folder that contains `final_s42/`.

## Reproduce

```bash
cd src
python audit.py                                          # 1. audit, dedup/overlap, split, diagnostics figure
python baseline.py                                       # 2. TF-IDF grid on validation macro-F1
python finetune.py --pilot 100                           # 3. 100-step resource pilot
python finetune.py --lr 8e-5 --seed 42 --tag final_s42 --save_model   # 4. final model (also seeds 13 and 7)
python evaluate.py                                       # 5. test metrics, selective metrics, per-class, confusion, errors
python demo_check.py                                     # 6. fixed demo inputs
cd .. && streamlit run src/app.py                        # 7. demo at http://localhost:8501
```

The learning-rate check used `python finetune.py --lr {3e-5,5e-5,8e-5}`. All randomness uses seed 42 unless noted. Re-running seed 42 on the same GPU reproduced the validation scores exactly.
