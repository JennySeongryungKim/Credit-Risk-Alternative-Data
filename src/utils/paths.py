# -*- coding: utf-8 -*-
"""
paths.py
---------
Centralized path management for the project.

Automatically detects and manages all major directories:
    - src, notebooks, docs, data, artifact, EDA_output
Provides helper functions for consistent model, figure, and report saving.
"""

import os
from datetime import datetime

# ----------------------------------------------------------------------------
# 🔹 Base paths
# ----------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SRC_DIR = os.path.join(ROOT_DIR, "src")
NOTEBOOKS_DIR = os.path.join(SRC_DIR, "notebooks")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifact")
EDA_OUTPUT_DIR = os.path.join(ARTIFACT_DIR, "EDA_output")  

# ----------------------------------------------------------------------------
# 🔹 Data files
# ----------------------------------------------------------------------------
PREPROCESSED_SAMPLE = os.path.join(DATA_DIR, "preprocessed_data_sample_1pct.pkl.gz")
PREPROCESSOR_PKL = os.path.join(DATA_DIR, "preprocessor.pkl")

# ----------------------------------------------------------------------------
# 🔹 Artifact files
# ----------------------------------------------------------------------------
MODEL_RESULTS_CSV = os.path.join(ARTIFACT_DIR, "01_Model_results.csv")
MODEL_COMPARISON_PNG = os.path.join(ARTIFACT_DIR, "model_comparison.png")
THIN_FILE_ANALYSIS_PNG = os.path.join(ARTIFACT_DIR, "thin_file_analysis.png")

# ----------------------------------------------------------------------------
# 🔹 Directory verification
# ----------------------------------------------------------------------------
def ensure_dirs():
    """Ensure all required directories exist."""
    dirs = [
        DATA_DIR,
        ARTIFACT_DIR,
        DOCS_DIR,
        NOTEBOOKS_DIR,
        UTILS_DIR,
        EDA_OUTPUT_DIR, 
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Verified all directories exist.")

# ----------------------------------------------------------------------------
# 🔹 Path generators
# ----------------------------------------------------------------------------
def model_path(name: str, ext: str = "pkl") -> str:
    """Return full path for model artifact file."""
    return os.path.join(ARTIFACT_DIR, f"{name}.{ext}")

def figure_path(name: str, ext: str = "png", subdir: str = "EDA_output") -> str:
    """
    Return full path for saving figures.
    By default saves inside artifact/EDA_output.
    Example:
        figure_path("pca_scatter") → artifact/EDA_output/pca_scatter.png
    """
    base_dir = EDA_OUTPUT_DIR if subdir == "EDA_output" else ARTIFACT_DIR
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{name}.{ext}")

def report_path(name: str, ext: str = "md") -> str:
    """Return path for saving documentation or markdown reports."""
    return os.path.join(DOCS_DIR, f"{name}.{ext}")

def timestamped_path(prefix: str, folder: str = ARTIFACT_DIR, ext: str = "csv") -> str:
    """Return a file path with a timestamp suffix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(folder, f"{prefix}_{ts}.{ext}")

# ----------------------------------------------------------------------------
# 🔹 Utility printer
# ----------------------------------------------------------------------------
def print_paths():
    """Print key project paths."""
    print("\n📁 PROJECT PATHS OVERVIEW")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"SRC_DIR: {SRC_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"ARTIFACT_DIR: {ARTIFACT_DIR}")
    print(f"EDA_OUTPUT_DIR: {EDA_OUTPUT_DIR}")
    print(f"DOCS_DIR: {DOCS_DIR}")
    print(f"MODEL_RESULTS_CSV: {MODEL_RESULTS_CSV}")
    print(f"PREPROCESSED_SAMPLE: {PREPROCESSED_SAMPLE}\n")

# ----------------------------------------------------------------------------
# 🔹 Run check
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print_paths()
    ensure_dirs()
