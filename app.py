"""
app.py
------
GUARDIAN-NLP: Main Application Launcher.
Launches either the Gradio single-text analyzer or Streamlit dashboard.

Usage:
    python app.py              → launches Gradio (default)
    python app.py --mode gradio
    python app.py --mode streamlit
    python app.py --mode both  → launches Gradio only (run streamlit separately)
"""

import argparse
import os
import subprocess
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP App Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        default="gradio",
        choices=["gradio", "streamlit", "both"],
        help="Which UI to launch (default: gradio)",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def launch_gradio(config: dict) -> None:
    """Launches the Gradio single-text analyzer."""
    print("\n🛡️  Launching GUARDIAN-NLP Gradio Analyzer...")
    print(f"   URL: http://localhost:{config.get('ui', {}).get('gradio_port', 7860)}")
    print("   Press Ctrl+C to stop.\n")
    # Change working directory to project root so relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from src.ui.gradio_app import demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.get("ui", {}).get("gradio_port", 7860),
        share=config.get("ui", {}).get("share", False),
        inbrowser=False,
    )


def launch_streamlit(config: dict) -> None:
    """Launches the Streamlit NGO dashboard via subprocess."""
    port = config.get("ui", {}).get("streamlit_port", 8501)
    print("\n🛡️  Launching GUARDIAN-NLP Streamlit Dashboard...")
    print(f"   URL: http://localhost:{port}")
    print("   Press Ctrl+C to stop.\n")
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "ui", "streamlit_dashboard.py",
    )
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path,
         "--server.port", str(port), "--server.headless", "false"],
        check=True,
    )


def main():
    args = parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        🛡️  SafetySpeech  —  AI Safety System             ║
    ║     Multi-Label Toxic Content Detection for Analysts     ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    if args.mode == "gradio":
        launch_gradio(config)
    elif args.mode == "streamlit":
        launch_streamlit(config)
    elif args.mode == "both":
        print("💡 To run both simultaneously, open two terminals:")
        print("   Terminal 1: python app.py --mode gradio")
        print("   Terminal 2: python app.py --mode streamlit")
        print("\nLaunching Gradio now:")
        launch_gradio(config)


if __name__ == "__main__":
    main()
