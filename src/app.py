"""Week 5: Streamlit demo.  Run:  streamlit run src/app.py"""
import streamlit as st

from router import GLOSSARY, THRESHOLDS, BertRouter, TfidfRouter, topk

st.set_page_config(page_title="BANKING77 Intent Router", layout="wide")


@st.cache_resource
def load(name):
    return BertRouter() if name == "ModernBERT-base" else TfidfRouter()


EXAMPLES = ["I am still waiting on my card?", "Why was my card declined?", "Is there an exchange fee?",
            "I sent money to a friend yesterday but it still hasn't arrived."]

with st.sidebar:
    st.header("Settings")
    model_name = st.radio("Model", list(THRESHOLDS))
    thr = st.slider("Review threshold", 0.0, 1.0, float(THRESHOLDS[model_name]), 0.01,
                    help="Default is the value selected on validation for this model.")
    st.caption(f"Validation-selected default: {THRESHOLDS[model_name]:.2f}")
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.q = ex
    st.caption("Routing aid only · not financial advice")

st.title("Banking Intent Router")
st.write("Type a banking-support question; the classifier maps it to one of 77 intents.")
q = st.text_input("Question", key="q", label_visibility="collapsed", placeholder="e.g. my transfer is still pending")
if q.strip():
    top = topk(load(model_name).probs(q), 3)
    score = top[0][1]
    if score < thr:
        st.warning(f"Low confidence (top score {score:.2f} < {thr:.2f}). Please review the suggestions below or choose a category.")
    else:
        st.success(f"Routed to **{top[0][0]}** (top score {score:.2f} ≥ {thr:.2f}).")
    st.subheader("Top-3 intents")
    st.caption("Scores are uncalibrated softmax values, used only for ranking — not probabilities.")
    for name, s in top:
        c1, c2 = st.columns([3, 1])
        c1.progress(min(max(s, 0.0), 1.0), text=f"`{name}`")
        c2.write(f"{s:.2f}")
    sel = st.radio("Select an intent to see its gloss", [n for n, _ in top], horizontal=True)
    st.info(f"**Selected:** {sel}\n\n**Gloss:** {GLOSSARY.get(sel, '—')}")
