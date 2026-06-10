# Data Mining Assignments

This repository contains implementations for Data Mining Assignments 1-4. Each assignment folder includes the original assignment PDF, a Python implementation, a notebook wrapper, generated result files, and visual outputs.

## Setup

Use a Python environment with the packages listed in [requirements.txt](requirements.txt).

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code was developed against the package versions pinned in `requirements.txt`. If newer versions are used, minor API differences may require small adjustments.

## Run Everything

From the repository root:

```bash
python Assignment1/assignment1_analysis.py
python Assignment2/assignment2_recommender.py
python Assignment3/assignment3_clustering.py
python Assignment4/assignment4_classification.py
```

The notebooks are lightweight wrappers around the same scripts and display saved result summaries after execution.

## Repository Structure

```text
Assignment1/
  Assignment1.pdf
  Assignment1.ipynb
  Jupyter_Notebook.ipynb
  assignment1_analysis.py
  assignment1_analysis.xlsx
  google_review_ratings_original.csv
  google_review_ratings_cleaned.csv
  images/
  results/

Assignment2/
  Assignment2.pdf
  Assignment2.ipynb
  assignment2_recommender.py
  data/
  images/
  results/

Assignment3/
  Assignment3.pdf
  Assignment3.ipynb
  assignment3_clustering.py
  report.pdf
  report.tex
  report.md
  images/
  results/

Assignment4/
  Assignment4.pdf
  Assignment4.ipynb
  assignment4_classification.py
  readme.pdf
  readme.tex
  readme.md
  Assignment4_classification.pptx
  data/
  images/
  results/

tools/
  create_assignment_notebooks.py
```

## Assignment 1: EDA and Dimensionality Reduction

Dataset: Google Review Ratings / Travel Review Ratings from UCI.

Source: <https://archive.ics.uci.edu/dataset/485/tarvel+review+ratings>

Implementation: [Assignment1/assignment1_analysis.py](Assignment1/assignment1_analysis.py)

Main deliverables:

- [Assignment1/Assignment1.ipynb](Assignment1/Assignment1.ipynb)
- [Assignment1/Jupyter_Notebook.ipynb](Assignment1/Jupyter_Notebook.ipynb)
- [Assignment1/assignment1_analysis.xlsx](Assignment1/assignment1_analysis.xlsx)
- [Assignment1/google_review_ratings_cleaned.csv](Assignment1/google_review_ratings_cleaned.csv)
- [Assignment1/results/](Assignment1/results/)
- [Assignment1/images/](Assignment1/images/)

What it does:

- Cleans malformed CSV rows and repairs numeric rating fields.
- Computes summary statistics, missing-value counts, and IQR outlier counts.
- Generates histograms, box plots, scatter plots, and a correlation heatmap.
- Computes centered-cosine correlation matrix.
- Computes mean vector, total variance, and sample covariance matrix.
- Verifies covariance through both inner-product and outer-product formulations.
- Applies PCA and writes the 2D projection.
- Writes an Excel workbook with key tables.

Key generated results:

- Rows after cleaning: 5,456
- Total variance: `39.5503`
- Most correlated pair: `parks` vs `theatres`, correlation `0.6269`
- Most anti-correlated pair: `malls` vs `view_points`, correlation `-0.3603`
- Least correlated pair: `beaches` vs `cafes`, correlation `0.0011`
- PCA explained variance ratios: PC1 `0.1966`, PC2 `0.1456`

## Assignment 2: Collaborative Filtering

Dataset: ModCloth ratings dataset from the `marketBias` repository.

Source: <https://github.com/MengtingWan/marketBias>

Direct CSV used by the script: <https://raw.githubusercontent.com/MengtingWan/marketBias/master/data/df_modcloth.csv>

Implementation: [Assignment2/assignment2_recommender.py](Assignment2/assignment2_recommender.py)

Main deliverables:

- [Assignment2/Assignment2.ipynb](Assignment2/Assignment2.ipynb)
- [Assignment2/results/cross_validation_results.csv](Assignment2/results/cross_validation_results.csv)
- [Assignment2/results/test_predictions.csv](Assignment2/results/test_predictions.csv)
- [Assignment2/results/assignment2_results.json](Assignment2/results/assignment2_results.json)
- [Assignment2/images/](Assignment2/images/)

What it does:

- Downloads the ModCloth CSV when missing.
- Cleans invalid user/item/rating rows.
- Builds an 85/15 stratified train/test split by rating.
- Runs 5-fold cross-validation on the training portion.
- Implements item-based collaborative filtering with adjusted cosine similarity.
- Uses top-30 item neighbors.
- Uses user/item/global mean fallback for cold-start and zero-similarity cases.
- Reports MAE and RMSE.
- Generates rating-distribution, dataset-scale, validation/test-error, actual-vs-predicted, and baseline-comparison plots.

Key generated results:

- Rows after cleaning: 99,892
- Users: 44,783
- Items: 1,020
- Matrix sparsity: `0.9978`
- CV MAE: `0.8724 +/- 0.0084`
- CV RMSE: `1.1754 +/- 0.0096`
- Test MAE: `0.8798`
- Test RMSE: `1.1899`
- Test rows: 14,984

## Assignment 3: Clustering

Datasets:

- Breast Cancer Wisconsin Diagnostic: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
- Iris: <https://archive.ics.uci.edu/dataset/53/iris>
- Wine Recognition: <https://archive.ics.uci.edu/dataset/109/wine>

Implementation: [Assignment3/assignment3_clustering.py](Assignment3/assignment3_clustering.py)

Main deliverables:

- [Assignment3/Assignment3.ipynb](Assignment3/Assignment3.ipynb)
- [Assignment3/report.pdf](Assignment3/report.pdf)
- [Assignment3/report.tex](Assignment3/report.tex)
- [Assignment3/results/](Assignment3/results/)
- [Assignment3/images/](Assignment3/images/)

What it does:

- Standardizes all datasets.
- Evaluates OPTICS as the density-based algorithm.
- Tunes OPTICS parameters on the density-selected dataset.
- Evaluates agglomerative clustering with single, complete, average, and Ward/minimum-variance linkage.
- Evaluates KMeans baseline, custom KMedoids, and PCA+KMeans for prototype-based clustering.
- Reports intrinsic metrics: silhouette and Davies-Bouldin.
- Reports extrinsic metrics: normalized mutual information, adjusted Rand index, and purity.
- Compares observed clustering against random Gaussian baselines.
- Generates PCA views, best-cluster visualizations, tuning plots, linkage comparison, prototype comparison, and Iris dendrogram.
- Generates a PDF report.

Key generated results:

- Density family: complete OPTICS results are saved in [Assignment3/results/density_results.csv](Assignment3/results/density_results.csv).
- Hierarchical family: best Wine result used Ward/minimum-variance linkage with purity `0.9270` and adjusted Rand `0.7899`.
- Prototype family: Wine KMeans baseline achieved purity `0.9663`, NMI `0.8759`, adjusted Rand `0.8975`.

The generated [Assignment3/report.pdf](Assignment3/report.pdf) includes the dataset-selection hypotheses and states where the results support or weaken those hypotheses.

## Assignment 4: Classification

Datasets:

- Cervical Cancer Risk Factors: <https://christophm.github.io/interpretable-ml-book/cervical.html>
- Fetal Health Classification: <https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification>
- Banking Marketing CSV: <https://github.com/AndrzejSzymanski/TDS/blob/master/banking.csv>

Implementation: [Assignment4/assignment4_classification.py](Assignment4/assignment4_classification.py)

Main deliverables:

- [Assignment4/Assignment4.ipynb](Assignment4/Assignment4.ipynb)
- [Assignment4/readme.pdf](Assignment4/readme.pdf)
- [Assignment4/Assignment4_classification.pptx](Assignment4/Assignment4_classification.pptx)
- [Assignment4/results/classification_metrics.csv](Assignment4/results/classification_metrics.csv)
- [Assignment4/results/dataset_profiles.json](Assignment4/results/dataset_profiles.json)
- [Assignment4/images/](Assignment4/images/)

What it does:

- Downloads datasets when missing.
- Handles inconsistencies and missing values.
- Clips numeric outliers using the 1.5 IQR rule.
- Uses stratified 80/20 train/test splits.
- Performs 5-fold cross-validation on the training split.
- Trains Decision Tree, Random Forest, XGBoost, and AdaBoost classifiers.
- Reports precision, recall, F1, accuracy, and AUC-ROC.
- Generates ROC curves.
- Generates PCA-based decision-boundary visualizations.
- Generates `readme.pdf` and a PPT.

Generated held-out test metrics:

| Dataset | Model | Accuracy | AUC-ROC |
|---|---:|---:|---:|
| cervical | Decision Tree | 0.8140 | 0.5536 |
| cervical | Random Forest | 0.9302 | 0.5641 |
| cervical | XGBoost | 0.9302 | 0.5776 |
| cervical | AdaBoost | 0.9128 | 0.5127 |
| fetal_health | Decision Tree | 0.9014 | 0.9327 |
| fetal_health | Random Forest | 0.9343 | 0.9835 |
| fetal_health | XGBoost | 0.9460 | 0.9799 |
| fetal_health | AdaBoost | 0.8967 | 0.9434 |
| banking | Decision Tree | 0.8813 | 0.7989 |
| banking | Random Forest | 0.9087 | 0.9444 |
| banking | XGBoost | 0.9145 | 0.9471 |
| banking | AdaBoost | 0.9116 | 0.9460 |

## Regenerating Notebooks

The notebooks are generated by:

```bash
python tools/create_assignment_notebooks.py
```

This keeps the notebooks consistent with the scripts. The notebooks intentionally stay small; the assignment logic lives in the Python scripts for easier reruns and debugging.

## Notes and Constraints

- Existing assignment PDFs are preserved.
- Existing Assignment1 source dataset is preserved.
- Generated datasets under `Assignment2/data/` and `Assignment4/data/` are reproducibility caches.
- `Assignment1/google_review_ratings_work.xlsx` is the original existing workbook.
- `Assignment1/assignment1_analysis.xlsx` is the generated workbook.
- `Assignment3/report.tex` and `Assignment4/readme.tex` are retained because they are source files for the generated PDFs.
- LaTeX auxiliary files are not retained.
- If remote CSV downloads fail, rerun when network access is available or keep the already downloaded CSVs in the `data/` folders.
