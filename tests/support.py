"""Locate public examples or private numerical regression fixtures."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def config_path(name):
    public = ROOT / 'engine' / 'config' / name
    return public if public.is_file() else ROOT / 'tests' / 'fixtures' / 'config' / name
