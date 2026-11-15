"""Demo check: 15 fixed inputs (14 hand-written + the proposal example, which is also a train text); is the gold intent in the displayed top-3?"""
import pandas as pd

from common import RESULTS
from router import THRESHOLDS, BertRouter, TfidfRouter, topk

CASES = [
    ("I am still waiting on my card?", "card_arrival"),  # proposal example (also a train text)
    ("I sent money to a friend yesterday but it still hasn't arrived.", "transfer_not_received_by_recipient"),
    ("My card got eaten by the cash machine at the station", "card_swallowed"),
    ("Why did you charge me a fee for paying with my card in a shop?", "card_payment_fee_charged"),
    ("I think someone has my card number, there are payments I never made", "compromised_card"),
    ("How do I close my account for good?", "terminate_account"),
    ("Tapping my card on the reader doesn't do anything anymore", "contactless_not_working"),
    ("The ATM only gave me 50 but I asked for 100", "wrong_amount_of_cash_received"),
    ("Can I add my card to my iPhone wallet?", "apple_pay_or_google_pay"),
    ("I forgot the passcode for the app", "passcode_forgotten"),
    ("My top-up has been pending for hours", "pending_top_up"),
    ("Is my card a Visa or a Mastercard?", "visa_or_mastercard"),
    ("Why do you need to verify who I am?", "why_verify_identity"),
    ("I lost my phone, what do I do about the app?", "lost_or_stolen_phone"),
    ("Can I hold euros and dollars in my account?", "fiat_currency_support"),
]
rows = []
for name, r in [("ModernBERT-base", BertRouter()), ("TF-IDF + LogReg baseline", TfidfRouter())]:
    for text, gold in CASES:
        top = topk(r.probs(text), 3)
        rows.append({"model": name, "text": text, "gold": gold, "top1": top[0][0], "top1_score": round(top[0][1], 3),
                     "top3": "; ".join(f"{n} ({s:.2f})" for n, s in top), "top1_correct": top[0][0] == gold,
                     "gold_in_top3": gold in [n for n, _ in top], "review_prompt": top[0][1] < THRESHOLDS[name]})
df = pd.DataFrame(rows); df.to_csv(RESULTS / "demo_checks.csv", index=False)
print(df.groupby("model")[["top1_correct", "gold_in_top3", "review_prompt"]].sum())
print(df[~df.top1_correct][["model", "text", "gold", "top3"]].to_string())
