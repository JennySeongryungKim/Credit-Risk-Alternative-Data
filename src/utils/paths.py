import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(ROOT, "data")
ARTIFACT_DIR = os.path.join(ROOT, "artifact")
MODELS_DIR = os.path.join(ARTIFACT_DIR, "models")
FIG_DIR = os.path.join(ARTIFACT_DIR, "figures")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)

def artifact_path(name: str) -> str:
    return os.path.join(ARTIFACT_DIR, name)

def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)

def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)
