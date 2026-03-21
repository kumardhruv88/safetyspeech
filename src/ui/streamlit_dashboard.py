"""
streamlit_dashboard.py
-----------------------
GUARDIAN-NLP Full NGO Analyst Dashboard.

Features:
  1. Sidebar: CSV upload for batch analysis + real-time single text check
  2. Confidence threshold slider and platform filter
  3. Color-coded results table (red=violent, orange=hate, yellow=depressive)
  4. Plotly charts: label distribution, platform breakdown, confidence histogram
  5. Word cloud of flagged content
  6. Stats panel: total scanned, flagged %, by category
  7. Export: Download flagged CSV report

Launch: streamlit run src/ui/streamlit_dashboard.py
        → http://localhost:8501
"""

import io
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GUARDIAN-NLP | NGO Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.8rem; font-weight: 900; margin: 0;
    background: linear-gradient(90deg, #11998e, #38ef7d, #2980b9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    border-radius: 16px; padding: 20px; text-align: center;
    border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.metric-value { font-size: 2.4rem; font-weight: 800; color: #38ef7d; }
.metric-label { font-size: 0.9rem; color: #94a3b8; margin-top: 4px; }

.severity-high   { background: #e74c3c; color: white; border-radius: 8px; padding: 2px 10px; font-weight: 700; }
.severity-medium { background: #e67e22; color: white; border-radius: 8px; padding: 2px 10px; font-weight: 700; }
.severity-low    { background: #f1c40f; color: #333; border-radius: 8px; padding: 2px 10px; font-weight: 700; }
.severity-safe   { background: #2ecc71; color: white; border-radius: 8px; padding: 2px 10px; font-weight: 700; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, #11998e, #38ef7d); }
</style>
""", unsafe_allow_html=True)

LABELS = ["normal", "depressive", "hate_speech", "violent"]
LABEL_COLORS = {
    "normal": "#2ecc71",
    "depressive": "#f39c12",
    "hate_speech": "#e74c3c",
    "violent": "#8e0000",
}


# ─────────────────────────────────────────────────────────────────────
# Load Predictor (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🛡️ Loading GUARDIAN-NLP model...")
def load_predictor():
    from src.inference.predictor import Predictor
    return Predictor(
        model_path="models/checkpoints/best_model.pt",
        tokenizer_path="models/checkpoints/tokenizer/",
        device="cpu",
        threshold=0.5,
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def get_row_color(row: pd.Series) -> list:
    """Returns CSS background-color for styled dataframe rows."""
    if row.get("violent", 0) >= row.get("_threshold", 0.5):
        return ["background-color: rgba(231,76,60,0.25)"] * len(row)
    if row.get("hate_speech", 0) >= row.get("_threshold", 0.5):
        return ["background-color: rgba(231,76,60,0.12)"] * len(row)
    if row.get("depressive", 0) >= row.get("_threshold", 0.5):
        return ["background-color: rgba(243,156,18,0.2)"] * len(row)
    return [""] * len(row)


def compute_severity(row: pd.Series, threshold: float) -> str:
    max_toxic = max(row.get(l, 0) for l in LABELS if l != "normal")
    if max_toxic >= 0.85:
        return "HIGH 🔴"
    if max_toxic >= 0.60:
        return "MEDIUM 🟠"
    if max_toxic >= threshold:
        return "LOW 🟡"
    return "SAFE 🟢"


def run_batch_analysis(df: pd.DataFrame, predictor, threshold: float) -> pd.DataFrame:
    """Runs predictor on all rows in a DataFrame."""
    results = []
    progress = st.progress(0, text="Analyzing posts...")
    total = len(df)
    for i, text in enumerate(df["text"].astype(str)):
        result = predictor.predict(text, threshold=threshold)
        results.append({
            "normal": result.get("normal", 0.0),
            "depressive": result.get("depressive", 0.0),
            "hate_speech": result.get("hate_speech", 0.0),
            "violent": result.get("violent", 0.0),
            "top_label": result.get("top_label", "normal"),
            "max_confidence": result.get("max_confidence", 0.0),
        })
        progress.progress((i + 1) / total, text=f"Analyzing... {i+1}/{total}")
    progress.empty()

    results_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    out["severity"] = out.apply(lambda r: compute_severity(r, threshold), axis=1)
    out["is_flagged"] = out.apply(
        lambda r: any(r.get(l, 0) >= threshold for l in LABELS if l != "normal"), axis=1
    )
    return out


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ GUARDIAN-NLP")
    st.markdown("*UN NGO Content Safety Dashboard*")
    st.divider()

    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Detection Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Minimum confidence score to classify a post as toxic",
    )
    platform_filter = st.multiselect(
        "Filter by Platform",
        options=["Twitter/X", "Reddit", "Instagram", "Facebook", "Other"],
        default=[],
    )
    show_safe = st.checkbox("Show SAFE posts in table", value=False)

    st.divider()
    st.markdown("### 📂 Batch Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="CSV must have a 'text' column. Optional: 'platform' column.",
    )

    st.divider()
    st.markdown("### 🔍 Quick Analyze")
    quick_text = st.text_area(
        "Analyze a single post:",
        placeholder="Paste text here...",
        height=100,
    )
    quick_btn = st.button("Analyze", help="Run model on this single text")


# ─────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────
st.markdown('<span class="hero-title">🛡️ GUARDIAN-NLP Dashboard</span>', unsafe_allow_html=True)
st.markdown("**United Nations AI Safety Division** · Multi-label Toxic Content Detection System")
st.divider()

predictor = load_predictor()

# ─────────────────────────────────────────────────────────────────────
# B) Quick single-text analysis
# ─────────────────────────────────────────────────────────────────────
if quick_btn and quick_text.strip():
    result = predictor.predict(quick_text.strip(), threshold=confidence_threshold)
    st.markdown("#### 🔍 Quick Analysis Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Normal", f"{result['normal']:.1%}")
    c2.metric("Depressive", f"{result['depressive']:.1%}")
    c3.metric("Hate Speech", f"{result['hate_speech']:.1%}")
    c4.metric("Violent", f"{result['violent']:.1%}")
    st.info(f"**Severity:** {result['severity']}   |   **Top Label:** {result['top_label'].replace('_', ' ').title()}")
    st.divider()

# ─────────────────────────────────────────────────────────────────────
# C) Batch Analysis
# ─────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if "text" not in raw_df.columns:
        st.error("❌ CSV must contain a 'text' column.")
        st.stop()

    # Add platform if missing
    if "platform" not in raw_df.columns:
        raw_df["platform"] = "Unknown"

    # Apply platform filter
    if platform_filter:
        raw_df = raw_df[raw_df["platform"].isin(platform_filter)]
        if raw_df.empty:
            st.warning("No posts match the selected platform filter.")
            st.stop()

    # Run model
    with st.spinner("Running GUARDIAN-NLP on your dataset..."):
        analyzed_df = run_batch_analysis(raw_df, predictor, confidence_threshold)

    # Filter safe posts if needed
    display_df = analyzed_df if show_safe else analyzed_df[analyzed_df["is_flagged"]]

    # ── Stats Panel ────────────────────────────────────────────────
    total = len(analyzed_df)
    flagged = int(analyzed_df["is_flagged"].sum())
    high_sev = int((analyzed_df["severity"] == "HIGH 🔴").sum())
    pct_flagged = 100 * flagged / max(total, 1)

    st.markdown("### 📊 Scan Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Scanned", total)
    col2.metric("🚩 Flagged Posts", flagged, f"{pct_flagged:.1f}%")
    col3.metric("🔴 High Severity", high_sev)
    col4.metric("✅ Safe Posts", total - flagged)
    st.divider()

    # ── Charts Row ────────────────────────────────────────────────
    st.markdown("### 📈 Analytics")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        label_counts = {
            l.replace("_", " ").title(): int((analyzed_df[l] >= confidence_threshold).sum())
            for l in LABELS
        }
        fig_bar = px.bar(
            x=list(label_counts.keys()),
            y=list(label_counts.values()),
            color=list(label_counts.keys()),
            color_discrete_map={
                "Normal": "#2ecc71", "Depressive": "#f39c12",
                "Hate Speech": "#e74c3c", "Violent": "#8e0000",
            },
            title="Label Distribution",
            labels={"x": "Label", "y": "Post Count"},
            template="plotly_dark",
        )
        fig_bar.update_layout(showlegend=False, title_font_size=16)
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        platform_counts = analyzed_df[analyzed_df["is_flagged"]]["platform"].value_counts()
        if not platform_counts.empty:
            fig_pie = px.pie(
                names=platform_counts.index,
                values=platform_counts.values,
                title="Flagged Posts by Platform",
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_layout(title_font_size=16)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No flagged posts to show platform breakdown.")

    # Confidence Distribution Histogram
    fig_hist = px.histogram(
        analyzed_df,
        x="max_confidence",
        color="top_label",
        nbins=40,
        title="Confidence Score Distribution",
        labels={"max_confidence": "Max Confidence Score", "top_label": "Top Label"},
        template="plotly_dark",
        color_discrete_map={
            "normal": "#2ecc71", "depressive": "#f39c12",
            "hate_speech": "#e74c3c", "violent": "#8e0000",
        },
    )
    fig_hist.update_layout(title_font_size=16)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Word Cloud
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        flagged_texts = " ".join(
            analyzed_df[analyzed_df["is_flagged"]]["text"].astype(str).tolist()
        )
        if flagged_texts.strip():
            wc = WordCloud(
                width=900, height=350, background_color="#0f172a",
                colormap="Reds", max_words=150,
            ).generate(flagged_texts)
            fig_wc, ax = plt.subplots(figsize=(12, 4), facecolor="#0f172a")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.markdown("#### ☁️ Word Cloud (Flagged Content)")
            st.pyplot(fig_wc)
    except ImportError:
        st.info("Install wordcloud for word cloud visualization: pip install wordcloud")

    st.divider()

    # ── Results Table ────────────────────────────────────────────
    st.markdown(f"### 📋 Results ({len(display_df)} posts)")

    # Format confidence as %
    styled_cols = ["text", "platform", "depressive", "hate_speech", "violent", "severity", "top_label"]
    styled_cols = [c for c in styled_cols if c in display_df.columns]
    show_df = display_df[styled_cols].copy()

    for col in ["depressive", "hate_speech", "violent"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(lambda x: f"{x:.1%}")

    st.dataframe(show_df, use_container_width=True, height=400)

    # ── Export ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📥 Export")
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        flagged_export = analyzed_df[analyzed_df["is_flagged"]]
        csv_bytes = flagged_export.to_csv(index=False).encode("utf-8")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📊 Download Flagged CSV",
            data=csv_bytes,
            file_name=f"guardian_flagged_{timestamp}.csv",
            mime="text/csv",
        )

    with export_col2:
        full_csv = analyzed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📋 Download Full Report CSV",
            data=full_csv,
            file_name=f"guardian_full_report_{timestamp}.csv",
            mime="text/csv",
        )

else:
    # Landing state — no file uploaded
    st.markdown("""
    ### 👋 Welcome to the GUARDIAN-NLP Analyst Dashboard

    **Follow these steps to get started:**

    1. **Upload a CSV file** using the sidebar (must have a `text` column)
    2. **Adjust the detection threshold** (default: 0.5)
    3. **Filter by platform** if needed
    4. **Review results** with color-coded severity indicators
    5. **Export** flagged content as a CSV report

    ---

    **Or use the Quick Analyze tool** in the sidebar to test a single post immediately.

    ---
    #### System Info
    """)

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.info("**Model**: BERT-base-uncased")
    info_col2.info("**Labels**: Normal · Depressive · Hate Speech · Violent")
    info_col3.info("**Mode**: Multi-label (overlap allowed)")
