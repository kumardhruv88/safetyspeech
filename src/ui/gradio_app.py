"""
gradio_app.py
-------------
GUARDIAN-NLP Gradio Interface.
Single-text social media content analyzer for UN/NGO use.

Features:
  - Text input with platform selector
  - Multi-label confidence scores with color-coded bar chart
  - Severity indicator (SAFE / LOW / MEDIUM / HIGH)
  - Flag for Review button
  - Example inputs

Launch: python src/ui/gradio_app.py
        → http://localhost:7860
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gradio as gr
from src.inference.predictor import Predictor

# ─────────────────────────────────────────────
# Initialize predictor (load model once at startup)
# ─────────────────────────────────────────────
MODEL_PATH = "models/checkpoints/best_model.pt"
TOKENIZER_PATH = "models/checkpoints/tokenizer/"
THRESHOLD = 0.5

predictor = Predictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    device="cuda" if os.environ.get("USE_GPU") else "cpu",
    threshold=THRESHOLD,
)

LABELS = ["normal", "depressive", "hate_speech", "violent"]
LABEL_DISPLAY = ["Normal", "Depressive", "Hate Speech", "Violent"]
COLORS = {
    "Normal": "#2ecc71",
    "Depressive": "#f39c12",
    "Hate Speech": "#e74c3c",
    "Violent": "#8e0000",
}

SEVERITY_COLORS = {
    "SAFE 🟢": "#2ecc71",
    "LOW 🟡": "#f1c40f",
    "MEDIUM 🟠": "#e67e22",
    "HIGH 🔴": "#e74c3c",
}

FLAGGED_LOG = []


def format_severity_html(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#888")
    return f"""
    <div style="
        background: {color};
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        padding: 16px 24px;
        border-radius: 12px;
        letter-spacing: 1px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    ">
        {severity}
    </div>"""


def format_scores_html(result: dict) -> str:
    """Creates an HTML bar chart of label confidence scores."""
    bars = ""
    for label, display in zip(LABELS, LABEL_DISPLAY):
        score = result.get(label, 0.0)
        pct = int(score * 100)
        color = COLORS.get(display, "#888")
        activated = "⬤" if score >= THRESHOLD else "○"
        bars += f"""
        <div style="margin: 8px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-weight:600; color:{color};">{activated} {display}</span>
                <span style="font-weight:700;">{score:.2%}</span>
            </div>
            <div style="background:#e9ecef; border-radius:8px; height:22px; overflow:hidden;">
                <div style="background:{color}; width:{pct}%; height:100%; border-radius:8px; 
                            transition:width 0.5s ease; display:flex; align-items:center; padding-left:8px;">
                    <span style="color:white; font-size:0.75rem; font-weight:600;">
                        {pct}%
                    </span>
                </div>
            </div>
        </div>"""

    active_labels = [LABEL_DISPLAY[i] for i, l in enumerate(LABELS) if result.get(l, 0) >= THRESHOLD]
    active_str = ", ".join(active_labels) if active_labels else "None"

    return f"""
    <div style="font-family: 'Inter', sans-serif; padding: 8px;">
        <h4 style="margin-bottom:16px; color:#2c3e50;">📊 Confidence Scores</h4>
        {bars}
        <hr style="margin:16px 0; border-color:#dee2e6;">
        <p style="color:#6c757d; font-size:0.9rem;">
            <strong>Active labels (≥{THRESHOLD:.0%}):</strong> {active_str}
        </p>
    </div>"""


def analyze(text: str, platform: str, threshold: float) -> tuple:
    """Main Gradio analysis function."""
    if not text or not text.strip():
        empty_html = "<p style='color:#999; font-style:italic;'>Enter text to analyze.</p>"
        return empty_html, format_severity_html("SAFE 🟢"), gr.update(visible=False)

    result = predictor.predict(text.strip(), threshold=threshold)
    scores_html = format_scores_html(result)
    severity_html = format_severity_html(result["severity"])

    # Show flag button if text is toxic
    is_toxic = any(result.get(l, 0) >= threshold for l in LABELS if l != "normal")
    return scores_html, severity_html, gr.update(visible=is_toxic)


def flag_for_review(text: str, platform: str) -> str:
    """Logs flagged content for analyst review."""
    if text.strip():
        import datetime
        FLAGGED_LOG.append({
            "text": text[:500],
            "platform": platform,
            "timestamp": datetime.datetime.now().isoformat(),
        })
    return f"🚩 Flagged! Total flagged: {len(FLAGGED_LOG)}"


# ─────────────────────────────────────────────
# Gradio UI Layout
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
* { font-family: 'Outfit', sans-serif; }
.gradio-container { background: #0f172a !important; color: #f8fafc !important; }
.gr-panel { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 16px !important; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important; }
.gr-button-primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; border: none !important; font-weight: 600 !important; color: white !important; transition: all 0.2s ease !important; }
.gr-button-primary:hover { transform: translateY(-2px); box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.5) !important; }
.gr-button-stop { background: linear-gradient(135deg, #ef4444, #f97316) !important; border: none !important; color: white !important; font-weight: 600 !important; box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2) !important; }
"""

with gr.Blocks(
    title="SafetySpeech | Toxic Content Detector",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Outfit"),
    ),
    css=CSS,
) as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <h1 style="font-size:2.8rem; font-weight:800; margin:0;
                   background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🛡️ SafetySpeech
        </h1>
        <p style="color:#cbd5e1; font-size:1.1rem; font-weight:500; margin:8px 0 0 0;">
            Social Media Toxic Content Detection
        </p>
        <p style="color:#64748b; font-size:0.9rem; margin:4px 0 0 0;">
            Powered by BERT · Multi-label Detection · Real-time Analysis
        </p>
    </div>
    """)

    with gr.Row():
        # Input Column
        with gr.Column(scale=5):
            with gr.Group():
                text_input = gr.Textbox(
                    label="📝 Social Media Text",
                    placeholder="Paste a tweet, Reddit post, Instagram caption, or any social media text...",
                    lines=6,
                    max_lines=12,
                )
                with gr.Row():
                    platform = gr.Dropdown(
                        choices=["Twitter/X", "Reddit", "Instagram", "Facebook", "TikTok", "YouTube", "Other"],
                        value="Twitter/X",
                        label="📡 Source Platform",
                        scale=2,
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="🎚️ Detection Threshold",
                        scale=3,
                    )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Content", variant="primary", scale=3)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

        # Output Column
        with gr.Column(scale=5):
            scores_output = gr.HTML(
                value="<p style='color:#999; font-style:italic; padding:12px;'>Enter text and click Analyze.</p>",
                label="Detection Results",
            )
            severity_output = gr.HTML(
                value=format_severity_html("SAFE 🟢"),
                label="Severity Level",
            )
            flag_btn = gr.Button("🚩 Flag for Human Review", variant="stop", visible=False)
            flag_status = gr.Textbox(label="", visible=False, interactive=False)

    # Examples
    gr.HTML("<h3 style='color:#94a3b8; margin:16px 0 8px 0;'>💡 Example Inputs</h3>")
    gr.Examples(
        examples=[
            ["I just can't take it anymore, everything feels completely pointless and I'm so tired of existing.", "Reddit", 0.5],
            ["I love spending time with my family on weekends, it really lifts my mood!", "Instagram", 0.5],
            ["People like you don't deserve to exist in this world. You should be eliminated.", "Twitter/X", 0.5],
            ["I'm going to make you regret saying that. Watch your back tonight.", "Twitter/X", 0.5],
            ["These people are ruining our country. We need to get rid of them all.", "Facebook", 0.5],
            ["Had an amazing hike today! The views were breathtaking 🏔️", "Instagram", 0.5],
        ],
        inputs=[text_input, platform, threshold_slider],
        label="Click any example to populate the input:",
    )

    # Stats footer
    gr.HTML("""
    <div style="text-align:center; padding:16px 0 0 0; border-top: 1px solid rgba(255,255,255,0.1); margin-top:16px;">
        <p style="color:#475569; font-size:0.8rem;">
            ⚠️ For research and moderation use only · Not a substitute for human judgment · 
            Report false positives via the Flag button
        </p>
    </div>
    """)

    # Event handlers
    analyze_btn.click(
        fn=analyze,
        inputs=[text_input, platform, threshold_slider],
        outputs=[scores_output, severity_output, flag_btn],
    )
    clear_btn.click(
        fn=lambda: ("", "Twitter/X", 0.5),
        outputs=[text_input, platform, threshold_slider],
    )
    flag_btn.click(
        fn=flag_for_review,
        inputs=[text_input, platform],
        outputs=flag_status,
    ).then(lambda: gr.update(visible=True), outputs=flag_status)


if __name__ == "__main__":
    import yaml
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

    demo.launch(
        server_name="0.0.0.0",
        server_port=config.get("ui", {}).get("gradio_port", 7860),
        share=config.get("ui", {}).get("share", False),
        inbrowser=True,
    )
