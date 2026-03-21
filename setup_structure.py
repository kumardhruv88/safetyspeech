"""
setup_structure.py
------------------
Creates the full GUARDIAN-NLP directory and file structure.
Run once before starting the project:
    python setup_structure.py
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/external",
    "src/collect",
    "src/preprocess",
    "src/models",
    "src/evaluate",
    "src/inference",
    "src/ui",
    "models/checkpoints/tokenizer",
    "notebooks",
    "outputs/reports",
    "outputs/visualizations",
    "runs",
]

INIT_MODULES = [
    "src",
    "src/collect",
    "src/preprocess",
    "src/models",
    "src/evaluate",
    "src/inference",
    "src/ui",
]

PLACEHOLDER_FILES = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "models/checkpoints/.gitkeep",
    "outputs/reports/.gitkeep",
    "outputs/visualizations/.gitkeep",
    "runs/.gitkeep",
]


def create_structure():
    print("🛡️  GUARDIAN-NLP: Setting up project structure...\n")

    # Create directories
    for d in DIRECTORIES:
        full_path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✅ Created directory: {d}")

    # Create __init__.py files
    for module in INIT_MODULES:
        init_path = os.path.join(PROJECT_ROOT, module, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write(f'"""GUARDIAN-NLP: {module} module"""\n')
            print(f"  ✅ Created: {module}/__init__.py")

    # Create placeholder .gitkeep files
    for ph in PLACEHOLDER_FILES:
        full_path = os.path.join(PROJECT_ROOT, ph)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                pass
            print(f"  ✅ Created placeholder: {ph}")

    print("\n✅ Project structure created successfully!")
    print("📌 Next steps:")
    print("   1. Copy .env.example → .env and fill in your API keys")
    print("   2. Download datasets to data/external/")
    print("   3. Run: python -m src.collect.data_merger")
    print("   4. Run: python train.py --config config.yaml")
    print("   5. Run: python app.py")


if __name__ == "__main__":
    create_structure()
