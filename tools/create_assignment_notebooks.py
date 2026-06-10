#!/usr/bin/python3
"""Create lightweight assignment notebooks without requiring nbformat."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/usr/bin/python3"


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path, title, description, script_name, result_loader):
    assignment_dir = path.parent
    run_code = """
from pathlib import Path
import subprocess

ASSIGNMENT_DIR = Path.cwd()
if not (ASSIGNMENT_DIR / "{script_name}").exists():
    ASSIGNMENT_DIR = Path(r"{assignment_dir}")

subprocess.run([r"{python}", str(ASSIGNMENT_DIR / "{script_name}")], check=True)
""".format(
        script_name=script_name, assignment_dir=str(assignment_dir), python=PYTHON
    )
    cells = [
        markdown_cell("# {}\n\n{}".format(title, description)),
        markdown_cell(
            "Run the next cell to execute the complete implementation. "
            "It uses `/usr/bin/python3` because the default `python3` on this machine points to a separate runtime without the required data-science packages."
        ),
        code_cell(run_code),
        markdown_cell("## Result Summary"),
        code_cell(result_loader),
    ]
    path.write_text(json.dumps(notebook(cells), indent=2))


def main():
    write_notebook(
        ROOT / "Assignment1" / "Jupyter_Notebook.ipynb",
        "Assignment 1: Exploratory Data Analysis and PCA",
        "Portable implementation for Google Review Ratings. The script cleans malformed rows, computes centered-cosine correlations, mean vector, total variance, covariance matrices, outlier summaries, PCA projection, plots, and an Excel workbook.",
        "assignment1_analysis.py",
        """
import json
import pandas as pd
from pathlib import Path

base = ASSIGNMENT_DIR
with open(base / "results" / "assignment1_results.json") as fh:
    results = json.load(fh)
print(json.dumps(results["correlation_pairs"], indent=2))
display(pd.read_csv(base / "results" / "summary_statistics.csv").head())
""",
    )
    write_notebook(
        ROOT / "Assignment2" / "Assignment2.ipynb",
        "Assignment 2: ModCloth Collaborative Filtering",
        "Item-based collaborative filtering using adjusted cosine similarity, an 85/15 stratified split, five-fold cross-validation, cold-start mean fallbacks, and result infographics.",
        "assignment2_recommender.py",
        """
import json
import pandas as pd
from pathlib import Path

base = ASSIGNMENT_DIR
with open(base / "results" / "assignment2_results.json") as fh:
    results = json.load(fh)
print(json.dumps(results["validation"], indent=2))
print(json.dumps(results["test"], indent=2))
display(pd.read_csv(base / "results" / "cross_validation_results.csv"))
""",
    )
    write_notebook(
        ROOT / "Assignment3" / "Assignment3.ipynb",
        "Assignment 3: Clustering",
        "Density-based, hierarchical, and prototype-based clustering across Breast Cancer Wisconsin, Iris, and Wine datasets. Generates metric tables, plots, and report.pdf.",
        "assignment3_clustering.py",
        """
import json
import pandas as pd
from pathlib import Path

base = ASSIGNMENT_DIR
with open(base / "results" / "assignment3_results.json") as fh:
    results = json.load(fh)
print("Best density results:")
display(pd.DataFrame(results["best_density"]))
print("Best hierarchical results:")
display(pd.DataFrame(results["best_hierarchical"]))
print("Best prototype results:")
display(pd.DataFrame(results["best_prototype"]))
""",
    )
    write_notebook(
        ROOT / "Assignment4" / "Assignment4.ipynb",
        "Assignment 4: Classification",
        "Decision Tree, Random Forest, XGBoost, and AdaBoost on cervical cancer, fetal health, and banking datasets. Generates metrics, ROC curves, decision boundaries, readme.pdf, and PPT.",
        "assignment4_classification.py",
        """
import json
import pandas as pd
from pathlib import Path

base = ASSIGNMENT_DIR
with open(base / "results" / "assignment4_results.json") as fh:
    results = json.load(fh)
print("XGBoost available:", results["xgboost_available"])
metrics = pd.read_csv(base / "results" / "classification_metrics.csv")
display(metrics[metrics["split"] == "test"].sort_values(["dataset", "accuracy"], ascending=[True, False]))
""",
    )

    # Keep a conventionally named Assignment1 notebook alongside the original name.
    source = ROOT / "Assignment1" / "Jupyter_Notebook.ipynb"
    target = ROOT / "Assignment1" / "Assignment1.ipynb"
    target.write_text(source.read_text())


if __name__ == "__main__":
    main()
