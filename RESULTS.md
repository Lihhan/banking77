# BANKING77 intent routing — experiment results

Everything follows the proposal *"Routing Banking Questions with Fine-Grained Intent Classification"*. Test predictions were never used for any selection. `finetune.py` saves test-set probabilities whenever validation macro-F1 improves, so every run, including the learning-rate check, has a `test_probs.npy`. Model, epoch, learning rate, TF–IDF configuration and review threshold were all chosen on validation only. The test set was scored once, in `src/evaluate.py`, after these were fixed.

## Week 1 — Data audit, split, resource pilot

| Check | Result |
|---|---|
| Source | PolyAI-LDN/task-specific-datasets `banking_data/` (CC BY 4.0) |
| SHA-256 train / test | `b06e26ac…cbc664b` / `d12d6e3b…feeb474d` (full values in `results/audit.json`) |
| Rows | 10,003 train / 3,080 test; 77 labels in each split, same as `categories.json` |
| Empty text/label fields | 0 |
| Regex word length (`\w+(?:'\w+)?`) | mean 11.99 / 11.00, median 10 / 9 (train / test): **same as the proposal** |
| Train support | 35 (`contactless_not_working`) to 187 (`card_payment_fee_charged`); test is 40 per intent |
| Repeated texts after normalisation | 4 in train, 1 in test; no label conflicts |
| Train–test overlap | 7 texts, all with the same label |
| Model-fitting pool | 9,992 → **8,992 train / 1,000 val** (stratified, seed 42) |
| Test sets | full official test (3,080) and overlap-free test (3,073) |
| ModernBERT tokenizer lengths (train) | p50 13, p99 51, p99.9 72, max 93 → `max_length=128` keeps 100% of queries |
| 100-step pilot (RTX PRO 6000 Blackwell, bf16, bs 16×2) | peak allocated **3.5 GB**, 0.094 s/step → **about 1.3 min** projected for 3 epochs |

Figure: `figures/dataset_diagnostics.png` (rebuilds Appendix A).

## Week 2 — TF–IDF + class-weighted logistic regression

The grid was set in advance: 16 configurations (word n-grams {1, 1–2} × char_wb n-grams {2–4, 2–5} × C {1, 3, 10, 30}). Each was scored on validation macro-F1 (`results/tfidf/grid.csv`). All 16 took 4 min on CPU in total.
**Selected:** word 1–2-grams + char 2–4-grams, C = 30 → val acc 0.897, val macro-F1 0.897.
C = 30 is the largest value in the grid, so a larger C might do slightly better. We did not extend the grid after seeing this.

## Week 3 — ModernBERT-base fine-tuning

Settings: bf16, max_len 128, batch 16 × grad-accum 2 (effective 32), AdamW with weight decay 0.01, 10% warm-up then linear decay, at most 3 epochs, best epoch kept by val macro-F1.
The learning-rate check was small (seed 42):

| LR | val acc | val macro-F1 |
|---|---|---|
| 3e-5 | 0.906 | 0.908 |
| 5e-5 | 0.920 | 0.924 |
| **8e-5 (selected)** | **0.921** | **0.924** |

**This is best-epoch selection, not early stopping.** The proposal said "early stopping", but the code trains a fixed 3 epochs and keeps the epoch with the best validation macro-F1. Every run picked epoch 3, and validation was still improving there. The model is therefore probably under-trained, and more epochs might help. We kept the proposal's 3-epoch cap and did not test longer training.
The selected learning rate, 8e-5, is the largest value tried, and it only ties 5e-5 on macro-F1. So the optimum may lie above the range we searched.
Re-running seed 42 reproduced exactly the same numbers. The total GPU time over all runs, including the pilot and three seeds, was **0.16 GPU-hours**, far under the 1–3 h budget.

## Week 4 — Official test evaluation

### Main table (primary run: seed 42)

| Model | Subset | Acc | Macro-F1 | Top-3 acc | τ (from val) | Coverage | Selective acc |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | full test (3,080) | 0.9140 | 0.9140 | 0.9795 | 0.81 | 0.764 | 0.9860 |
| TF-IDF + LogReg | overlap-free (3,073) | 0.9138 | 0.9137 | 0.9795 | 0.81 | 0.764 | 0.9859 |
| **ModernBERT-base** | full test (3,080) | **0.9338** | **0.9340** | **0.9857** | 0.96 | **0.838** | **0.9876** |
| **ModernBERT-base** | overlap-free (3,073) | **0.9336** | **0.9338** | **0.9857** | 0.96 | **0.838** | **0.9876** |

* **95% bootstrap CIs (full test):** TF-IDF acc [0.904, 0.924], macro-F1 [0.903, 0.923]; ModernBERT acc [0.925, 0.942], macro-F1 [0.924, 0.941].
* **Paired McNemar test:** ModernBERT is right where TF-IDF is wrong on 121 queries; the reverse happens on 60. Exact p = 6.8 × 10⁻⁶.
* **Seed robustness (ModernBERT, seeds 42/13/7):** acc 0.9338 / 0.9364 / 0.9393 → **0.9365 ± 0.0028**; macro-F1 **0.9366 ± 0.0027**. Seed 42 was chosen before testing, and it is the lowest of the three.
* **Overlap sensitivity:** both models get all 7 overlap texts right. Removing them changes the scores by at most 0.0003, so the overlap does not inflate the results.
* **Reference point (not directly comparable):** Casanueva et al. (2020) report 93.66% for BERT-large and 93.36% for USE+ConveRT.

### Confidence-based review
The threshold τ is the smallest max-softmax value that reaches ≥ 98% selective accuracy on validation. We chose this 98% target because the proposal does not give a criterion. At the selected τ:
* ModernBERT auto-routes 83.8% of test queries at 98.8% accuracy. The other 498 go to review, and they include 172 of its 204 errors (84%).
* TF-IDF auto-routes only 76.4% at 98.6%.
* When both models auto-route the same share of queries, ModernBERT is ahead at every coverage level:

| Coverage | ModernBERT | TF-IDF |
|---|---|---|
| 100% | 0.934 | 0.914 |
| 95% | 0.961 | 0.940 |
| 90% | 0.977 | 0.961 |
| 80% | 0.991 | 0.984 |

The scores are uncalibrated ranking signals. One check example shows this: ModernBERT gave 1.00 to a wrong label (`card_payment_not_recognised` for a `compromised_card` query).

### Per-class and confusion
Weakest intents for ModernBERT (F1): `balance_not_updated_after_bank_transfer` 0.75, `pending_transfer` 0.81, `transfer_not_received_by_recipient` 0.85, `failed_transfer` 0.85, `topping_up_by_card` 0.86. The transfer-status group is the main source of routing risk.
Most frequent ModernBERT confusions: `pending_transfer → balance_not_updated_after_bank_transfer` (6), `transfer_not_received_by_recipient → balance_not_updated_after_bank_transfer` (4), `card_delivery_estimate → card_arrival` (4).
The lowest-support intents are not the weakest. `contactless_not_working` has 35 training examples and ModernBERT F1 0.97 (figure right panel). The errors come from semantically adjacent labels, not from data scarcity.
Files: `results/per_class_*.csv`, `figures/confusion_modernbert.png`, `figures/selective_and_perclass.png`.

### Qualitative error analysis (30 random ModernBERT errors, seed 42)
| Code | Count |
|---|---|
| Missing context (query underspecified, garbled or with an unclear referent) | 10 |
| Annotation ambiguity (gold label questionable or prediction at least as plausible) | 10 |
| Adjacent intent pair (label definitions overlap) | 7 |
| Lexical overlap (surface cue pulls to wrong intent) | 3 |

* **Gold in top 3:** 22 of the 30 sampled errors, and 78.4% of all 204 errors.
* **Annotation ambiguity:** in 10 of the 30, the prediction fits the text at least as well as the gold label. Some test "errors" are therefore label noise, and test accuracy may understate the model.
* **Lexical overlap:** only 3 errors are explained by a single misleading word.
* **Method:**
  * Codes follow the decision order in `results/error_codebook.md`. The final codes and per-item rationale are in `results/error_codes30_pass2.csv`. The texts are in `results/error_sample30.csv`. `src/error_analysis.py` rebuilds the sample and the tallies.

## Week 5 — Interactive demo
`src/app.py` (Streamlit) implements the mock-up:
* model switch;
* validation-selected review threshold (slider);
* example buttons;
* top-3 intents with uncalibrated scores;
* low-confidence review prompt;
* gloss for the selected intent (`data/glossary.json`, 77 hand-written glosses).

A headless `AppTest` ran both models with no exceptions.
Demo checks used 15 fixed inputs (`results/demo_checks.csv`): 14 hand-written ones, plus the proposal's example "I am still waiting on my card?", which is also a training text. Both models get that one right. Excluding it, top-1 is 11/14 for ModernBERT and 12/14 for TF–IDF.

| Model | Top-1 correct | Gold in top 3 | Review prompts |
|---|---|---|---|
| ModernBERT | 12/15 | 13/15 | 2 |
| TF-IDF | 13/15 | 13/15 | 3 |

Both models miss "Can I add my card to my iPhone wallet?" (`apple_pay_or_google_pay`) and "Tapping my card on the reader doesn't do anything" (`contactless_not_working`). Both are paraphrases that avoid the label's keywords.

## Deviations / notes
* The proposal did not say how to choose the review threshold. We used "≥ 98% selective accuracy on validation".
* The proposal said "early stopping". The code keeps the best of a fixed 3 epochs. All runs chose epoch 3 (see Week 3).
* Two selected hyper-parameters are at the edge of their search ranges: LR 8e-5 and TF–IDF C = 30.
* The demo is hosted as a static Hugging Face Space where both models run in the browser (fp16 ONNX ModernBERT, JS port of TF–IDF), instead of a hosted Streamlit server: free Hugging Face Spaces no longer run Docker apps. The Streamlit app is still in `src/app.py` for local use.
* Only the learning rate was tuned for ModernBERT, so it had 3 configuration checks. Epochs stayed capped at 3.
* Model checkpoints are saved to `/scratch/f008m8h/bank77/models/` because the home quota is too small. The Python venv is at `/scratch/f008m8h/bank77/venv`.
