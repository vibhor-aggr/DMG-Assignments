#!/usr/bin/python3
"""Assignment 1: EDA and dimensionality reduction for Google Review Ratings."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "google_review_ratings_original.csv"
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"

CATEGORY_NAMES = [
    "churches",
    "resorts",
    "beaches",
    "parks",
    "theatres",
    "museums",
    "malls",
    "zoo",
    "restaurants",
    "pubs_bars",
    "local_services",
    "burger_pizza_shops",
    "hotels_lodgings",
    "juice_bars",
    "art_galleries",
    "dance_clubs",
    "swimming_pools",
    "gyms",
    "bakeries",
    "beauty_spas",
    "cafes",
    "view_points",
    "monuments",
    "gardens",
]


def _coerce_rating(value):
    """Parse malformed numeric cells such as '2\\t2.' into 2.2."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().strip('"')
    text = re.sub(r"(?<=\d)\s+(?=\d)", ".", text)
    text = text.replace("\t", "").strip()
    if text.endswith(".") and text.count(".") > 1:
        text = text.rstrip(".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_and_clean():
    raw = pd.read_csv(DATA_FILE)
    rating_cols = ["Category {}".format(i) for i in range(1, 25)]

    if "Unnamed: 25" in raw.columns:
        raw["Category 24"] = raw["Category 24"].where(
            raw["Category 24"].notna(), raw["Unnamed: 25"]
        )
        raw = raw.drop(columns=["Unnamed: 25"])

    ratings = raw[rating_cols].applymap(_coerce_rating)
    missing_before = ratings.isna().sum().rename("missing_before_imputation")
    ratings = ratings.fillna(ratings.median())
    ratings = ratings.clip(lower=0, upper=5)
    ratings.columns = CATEGORY_NAMES

    cleaned = pd.concat([raw[["User"]], ratings], axis=1)
    cleaned.to_csv(BASE_DIR / "google_review_ratings_cleaned.csv", index=False)
    return cleaned, missing_before


def off_diagonal_pairs(matrix, columns):
    pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            pairs.append((columns[i], columns[j], float(matrix[i, j])))
    return pairs


def compute_results(cleaned, missing_before):
    X = cleaned[CATEGORY_NAMES].astype(float)
    n = len(X)
    means = X.mean()
    centered = X - means

    norms = np.sqrt((centered ** 2).sum(axis=0)).replace(0, np.nan)
    correlation = centered.T.dot(centered) / np.outer(norms, norms)
    correlation = pd.DataFrame(correlation, index=CATEGORY_NAMES, columns=CATEGORY_NAMES)

    covariance_inner = centered.T.dot(centered) / (n - 1)
    covariance_outer = sum(np.outer(row, row) for row in centered.values) / (n - 1)
    covariance_outer = pd.DataFrame(
        covariance_outer, index=CATEGORY_NAMES, columns=CATEGORY_NAMES
    )

    total_variance = float(np.trace(covariance_inner.values))
    pairs = off_diagonal_pairs(correlation.values, CATEGORY_NAMES)
    most_correlated = max(pairs, key=lambda item: item[2])
    most_anti_correlated = min(pairs, key=lambda item: item[2])
    least_correlated = min(pairs, key=lambda item: abs(item[2]))

    summary = X.describe().T
    summary["missing_before_imputation"] = missing_before.values
    outlier_counts = []
    for col in CATEGORY_NAMES:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts.append(int(((X[col] < lower) | (X[col] > upper)).sum()))
    summary["iqr_outlier_count"] = outlier_counts

    scaled = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    clusters = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(components)
    pca_df = pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "cluster": clusters,
            "mean_rating": X.mean(axis=1),
        }
    )

    results = {
        "dataset": {
            "name": "Google Review Ratings / Travel Review Ratings",
            "source": "https://archive.ics.uci.edu/dataset/485/tarvel+review+ratings",
            "rows": int(cleaned.shape[0]),
            "rating_attributes": len(CATEGORY_NAMES),
            "notes": [
                "The source CSV contains a trailing unnamed column.",
                "Two malformed rows were repaired by numeric coercion and median imputation.",
            ],
        },
        "mean_vector": {k: float(v) for k, v in means.items()},
        "total_variance": total_variance,
        "correlation_pairs": {
            "most_correlated": {
                "attribute_1": most_correlated[0],
                "attribute_2": most_correlated[1],
                "correlation": most_correlated[2],
            },
            "most_anti_correlated": {
                "attribute_1": most_anti_correlated[0],
                "attribute_2": most_anti_correlated[1],
                "correlation": most_anti_correlated[2],
            },
            "least_correlated": {
                "attribute_1": least_correlated[0],
                "attribute_2": least_correlated[1],
                "correlation": least_correlated[2],
            },
        },
        "pca": {
            "explained_variance_ratio": [
                float(v) for v in pca.explained_variance_ratio_
            ],
            "interpretation": (
                "The first two principal components summarize broad preference intensity "
                "and a contrast between leisure/food venues and landmark-style venues. "
                "The projection shows overlapping groups rather than sharply separated classes."
            ),
        },
        "covariance_inner_outer_max_abs_difference": float(
            np.max(np.abs(covariance_inner.values - covariance_outer.values))
        ),
    }

    RESULT_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULT_DIR / "summary_statistics.csv")
    correlation.to_csv(RESULT_DIR / "correlation_matrix_centered_cosine.csv")
    covariance_inner.to_csv(RESULT_DIR / "sample_covariance_matrix.csv")
    pca_df.to_csv(RESULT_DIR / "pca_projection.csv", index=False)
    with open(RESULT_DIR / "assignment1_results.json", "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    write_excel_workbook(summary, correlation, covariance_inner, pca_df, results)
    create_plots(X, correlation, pca_df, results)
    return results


def write_excel_workbook(summary, correlation, covariance, pca_df, results):
    output = BASE_DIR / "assignment1_analysis.xlsx"
    with pd.ExcelWriter(str(output), engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary")
        correlation.to_excel(writer, sheet_name="Correlation")
        covariance.to_excel(writer, sheet_name="Covariance")
        pca_df.to_excel(writer, sheet_name="PCA", index=False)
        key_rows = [
            ["source", results["dataset"]["source"]],
            ["rows", results["dataset"]["rows"]],
            ["rating_attributes", results["dataset"]["rating_attributes"]],
            ["total_variance", results["total_variance"]],
            [
                "most_correlated",
                "{attribute_1} vs {attribute_2}: {correlation:.4f}".format(
                    **results["correlation_pairs"]["most_correlated"]
                ),
            ],
            [
                "most_anti_correlated",
                "{attribute_1} vs {attribute_2}: {correlation:.4f}".format(
                    **results["correlation_pairs"]["most_anti_correlated"]
                ),
            ],
            [
                "least_correlated",
                "{attribute_1} vs {attribute_2}: {correlation:.4f}".format(
                    **results["correlation_pairs"]["least_correlated"]
                ),
            ],
        ]
        pd.DataFrame(key_rows, columns=["item", "value"]).to_excel(
            writer, sheet_name="Key Results", index=False
        )


def create_plots(X, correlation, pca_df, results):
    IMAGE_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    for ax, col in zip(axes.ravel(), CATEGORY_NAMES):
        ax.hist(X[col], bins=20, color="#4C78A8", edgecolor="white")
        ax.set_title(col.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Histograms of Google Review Ratings", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(IMAGE_DIR / "histograms.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot([X[col].values for col in CATEGORY_NAMES], vert=False, labels=CATEGORY_NAMES)
    ax.set_title("Box Plots of Rating Attributes")
    ax.set_xlabel("Average rating")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "boxplots.png", dpi=160)
    plt.close(fig)

    pairs = results["correlation_pairs"]
    scatter_specs = [
        ("most_correlated", "Most correlated"),
        ("most_anti_correlated", "Most anti-correlated"),
        ("least_correlated", "Least correlated"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axes, scatter_specs):
        pair = pairs[key]
        x_col = pair["attribute_1"]
        y_col = pair["attribute_2"]
        ax.scatter(X[x_col], X[y_col], s=8, alpha=0.35, color="#59A14F")
        ax.set_title("{}\nr={:.3f}".format(title, pair["correlation"]))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "scatter_key_pairs.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CATEGORY_NAMES)))
    ax.set_yticks(range(len(CATEGORY_NAMES)))
    ax.set_xticklabels(CATEGORY_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(CATEGORY_NAMES, fontsize=6)
    ax.set_title("Centered-Cosine Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["cluster"],
        s=10,
        alpha=0.7,
        cmap="viridis",
    )
    ax.set_title("PCA Projection of Users")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="KMeans group on PCA projection")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "pca_projection.png", dpi=160)
    plt.close(fig)


def main():
    cleaned, missing_before = load_and_clean()
    results = compute_results(cleaned, missing_before)
    print("Assignment 1 completed.")
    print("Cleaned rows: {}".format(results["dataset"]["rows"]))
    print("Total variance: {:.4f}".format(results["total_variance"]))
    for name, pair in results["correlation_pairs"].items():
        print(
            "{}: {} vs {} = {:.4f}".format(
                name, pair["attribute_1"], pair["attribute_2"], pair["correlation"]
            )
        )


if __name__ == "__main__":
    main()
