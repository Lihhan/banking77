# Error-coding codebook (30 random ModernBERT test errors)

The sample is `results/modernbert_errors.csv` with `.sample(30, random_state=42)`. Rebuild it and the tallies with `python src/error_analysis.py`.

Each error gets exactly one code. Apply the rules in this order and stop at the first one that fits:

1. **annotation_ambiguity**: from the text alone, the predicted intent fits at least as well as the gold intent, or the gold label looks wrong.
2. **missing_context**: the query is underspecified, garbled or has an unclear referent, so a careful human could not tell gold from prediction either.
3. **lexical_overlap**: the text has a clear cue for the gold intent, but a specific surface phrase pulls toward the predicted intent.
4. **intent_pair**: the text supports the gold intent, the two label definitions overlap, and no single phrase explains the error.

Pass 1 was drafted before this ordering was written down. Pass 2 applies the ordering. `flag = yes` marks items where the decision is close.
