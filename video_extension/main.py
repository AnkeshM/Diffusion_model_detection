import os
from pathlib import Path
import yaml
import click

ROOT_DIR = Path(__file__).resolve().parent

def load_config():
    with open(ROOT_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def init_directories():
    config = load_config()
    paths = [
        ROOT_DIR / config['dataset']['real_dir'],
        ROOT_DIR / config['dataset']['fake_dir'],
        ROOT_DIR / config['paths']['results_dir']
    ]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ensured directory exists: {p}")

if __name__ == "__main__":
    print("Initializing Video Forensics Extension...")
    init_directories()
    print("Please use 'python cli.py --help' for commands.")
