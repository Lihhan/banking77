"""Rebuild the 30-error sample (seed 42) and tally the error codes from both coding passes."""
import pandas as pd

from common import RESULTS, SEED

err = pd.read_csv(RESULTS / "modernbert_errors.csv")
sample = err.sample(30, random_state=SEED).reset_index(drop=True)
saved = pd.read_csv(RESULTS / "error_sample30.csv")
assert (sample.norm.values == saved.norm.values).all(), "sample does not match error_sample30.csv"
codes = pd.read_csv(RESULTS / "error_codes30_pass2.csv")  # final = pass2; flagged items confirmed by author review
print("gold in top 3:", int((sample.gold_rank <= 3).sum()), "/ 30")
print("author-confirmed flagged items:", int(codes.author_review.notna().sum()))
print("pass agreement:", int((codes.pass1 == codes.pass2).sum()), "/ 30;  flagged:", int((codes.flag == "yes").sum()))
print(pd.DataFrame({"pass1": codes.pass1.value_counts(), "pass2": codes.pass2.value_counts()}).fillna(0).astype(int))
