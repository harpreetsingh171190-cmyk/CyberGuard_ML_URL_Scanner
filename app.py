"""Streamlit web app:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from cyberguard.modeling import explain, load_bundle, predict_urls

st.set_page_config(page_title="CyberGuard URL Scanner", page_icon="🛡️", layout="centered")
st.title("🛡️ CyberGuard URL Scanner")
st.caption("Educational ML project. It reads URL text only. Never rely on it to decide if a link is safe.")


@st.cache_resource
def get_bundle(path="models/cyberguard_model.joblib"):
    return load_bundle(path)


try:
    bundle = get_bundle()
except Exception:
    st.error("No trained model found. Run `python -m cyberguard train --data your.csv` first.")
    st.stop()

if bundle.get("demo"):
    st.warning("This model was trained on the tiny demo set. Results are not meaningful.")

tab1, tab2 = st.tabs(["Single URL", "Batch (CSV)"])
with tab1:
    url = st.text_input("URL", placeholder="http://secure-login.example.xyz/verify")
    if url:
        r = explain(bundle, url)
        st.metric("Malicious probability", f"{r['malicious_probability']:.1%}")
        (st.error if r["verdict"] == "malicious" else st.success)(f"{r['verdict'].upper()} | risk {r['risk_level']}")
        if r["signals"]:
            st.markdown("**Red flags**\n" + "\n".join(f"- {s}" for s in r["signals"]))
        if r["top_features"]:
            st.markdown("**What moved the score** (positive = towards malicious)")
            st.dataframe(pd.DataFrame(r["top_features"]), use_container_width=True, hide_index=True)
with tab2:
    up = st.file_uploader("CSV with a `url` column", type="csv")
    if up:
        df = pd.read_csv(up)
        out = predict_urls(bundle, df["url"].astype(str))
        st.dataframe(out, use_container_width=True)
        st.download_button("Download results", out.to_csv(index=False), "scan_results.csv")
