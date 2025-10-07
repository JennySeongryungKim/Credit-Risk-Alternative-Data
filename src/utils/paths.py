# -*- coding: utf-8 -*-
"""
paths.py
---------
Centralized path management for the project.

This module defines unified directory references for:
  - src / notebooks / docs / data / artifact
  - model and figure output files

Usage:
    from src.utils.paths import DATA_DIR, MODEL_RESULTS_PATH
"""

import os

# ----------------------------------------------------------------------------
# 🔹 Base directory (project root)
# ----------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# ----------------------------------------------------------------------------
# 🔹 Main directories
# ----------------------------------------------------------------------------
SRC_DIR = os.path.join(ROOT_DIR, "src")
NOTEBOOKS_DIR = os.path.join(SRC_DIR, "notebooks")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifact")

# ----------------------------------------------------------------------------
# 🔹 Artifact files (for convenience)
# ----------------------------------------------------------------------------
MODEL_RESULTS_PATH = os.path.join(ARTIFACT_DIR, "01_Model_results.csv")
MODEL_COMPARISON_FIG = os.path.join(ARTIFACT_DIR, "model_comparison.png")
THIN_FILE_ANALYSIS_FIG = os.path.join(ARTIFACT_DIR, "thin_file_analysis.png")

# ----------------------------------------------------------------------------
# 🔹 Data files
# ----------------------------------------------------------------------------
PREPROCESSED_DATA = os.path.join(DATA_DIR, "preprocessed_data_sample_1pct.pkl.gz")
FULL_PREPROCESSOR = os.path.join(DATA_DIR, "preprocessor.pkl")

# ----------------------------------------------------------------------------
# 🔹 Utility functions
# ----------------------------------------------------------------------------
def ensure_dirs():
    """Ensure that essential directories exist."""
    for path in [DATA_DIR, ARTIFACT_DIR, DOCS_DIR, NOTEBOOKS_DIR]:
        os.makedirs(path, exist_ok=True)
    print("✅ All key directories verified/created.")

def print_paths():
    """Print overview of key paths."""
    print("📁 PROJECT PATHS OVERVIEW")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"SRC_DIR: {SRC_DIR}")
    print(f"NOTEBOOKS_DIR: {NOTEBOOKS_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"ARTIFACT_DIR: {ARTIFACT_DIR}")
    print(f"DOCS_DIR: {DOCS_DIR}")
    print(f"MODEL_RESULTS_PATH: {MODEL_RESULTS_PATH}")
    print(f"PREPROCESSED_DATA: {PREPROCESSED_DATA}")

# ----------------------------------------------------------------------------
# 🔹 Execute for quick check
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print_paths()
    ensure_dirs()
