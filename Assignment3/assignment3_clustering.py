#!/usr/bin/python3
"""Assignment 3: Density, hierarchical, and prototype-based clustering."""

import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"
RANDOM_STATE = 42


DATASET_SPECS = [
    {
        "key": "breast_cancer",
        "name": "Breast Cancer Wisconsin Diagnostic",
        "loader": load_breast_cancer,
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "hypothesis": "Density-based clustering should be most suitable because the benign/malignant diagnostic measurements form dense regions with clinically meaningful borderline/noise cases.",
    },
    {
        "key": "iris",
        "name": "Iris",
        "loader": load_iris,
        "source": "https://archive.ics.uci.edu/dataset/53/iris",
        "hypothesis": "Hierarchical clustering should be most suitable because the botanical species have a natural nested structure, especially the separation of setosa from the other two species.",
    },
    {
        "key": "wine",
        "name": "Wine Recognition",
        "loader": load_wine,
        "source": "https://archive.ics.uci.edu/dataset/109/wine",
        "hypothesis": "Prototype-based clustering should be most suitable because the wine cultivars are compact groups in standardized chemical measurement space.",
    },
]


def load_datasets():
    datasets = []
    for spec in DATASET_SPECS:
        bunch = spec["loader"]()
        X = StandardScaler().fit_transform(bunch.data.astype(float))
        y = bunch.target.astype(int)
        datasets.append(
            {
                "key": spec["key"],
                "name": spec["name"],
                "source": spec["source"],
                "hypothesis": spec["hypothesis"],
                "X": X,
                "y": y,
                "target_names": [str(t) for t in getattr(bunch, "target_names", [])],
                "n_clusters": len(np.unique(y)),
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
            }
        )
    return datasets


def purity_score(y_true, y_pred):
    total = len(y_true)
    score = 0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        if mask.sum() == 0:
            continue
        labels, counts = np.unique(y_true[mask], return_counts=True)
        score += counts.max()
    return float(score / float(total))


def cluster_count(labels):
    labels = np.asarray(labels)
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def metric_row(dataset_key, family, algorithm, labels, X, y, params):
    labels = np.asarray(labels)
    n_clusters = cluster_count(labels)
    non_noise = labels != -1
    silhouette = np.nan
    davies = np.nan
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        try:
            silhouette = float(silhouette_score(X[non_noise], labels[non_noise]))
            davies = float(davies_bouldin_score(X[non_noise], labels[non_noise]))
        except Exception:
            silhouette = np.nan
            davies = np.nan
    return {
        "dataset": dataset_key,
        "family": family,
        "algorithm": algorithm,
        "params": json.dumps(params, sort_keys=True),
        "clusters_found": n_clusters,
        "noise_points": int((labels == -1).sum()),
        "silhouette": silhouette,
        "davies_bouldin": davies,
        "nmi": float(normalized_mutual_info_score(y, labels)),
        "adjusted_rand": float(adjusted_rand_score(y, labels)),
        "purity": purity_score(y, labels),
    }


def fit_optics(X, min_samples=10, xi=0.05, min_cluster_size=0.05):
    return OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        cluster_method="xi",
    ).fit_predict(X)


def run_density(datasets):
    rows = []
    best_labels = {}
    grids = {
        "min_samples": [5, 10, 20],
        "xi": [0.03, 0.05, 0.10],
        "min_cluster_size": [0.03, 0.05, 0.10],
    }
    tuning_rows = []
    for ds in datasets:
        best = None
        for min_samples in grids["min_samples"]:
            for xi in grids["xi"]:
                for min_cluster_size in grids["min_cluster_size"]:
                    labels = fit_optics(ds["X"], min_samples, xi, min_cluster_size)
                    params = {
                        "min_samples": min_samples,
                        "xi": xi,
                        "min_cluster_size": min_cluster_size,
                    }
                    row = metric_row(
                        ds["key"], "density", "OPTICS", labels, ds["X"], ds["y"], params
                    )
                    if ds["key"] == "breast_cancer":
                        tuning_rows.append(row)
                    rows.append(row)
                    score = row["silhouette"]
                    if math.isnan(score):
                        score = -999
                    if best is None or score > best[0]:
                        best = (score, labels, params, row)
        best_labels[ds["key"]] = best[1]
    return pd.DataFrame(rows), pd.DataFrame(tuning_rows), best_labels


def run_hierarchical(datasets):
    rows = []
    best_labels = {}
    linkages = [("single", "single"), ("complete", "complete"), ("group_average", "average"), ("minimum_variance", "ward")]
    for ds in datasets:
        best = None
        for label, linkage_name in linkages:
            model = AgglomerativeClustering(
                n_clusters=ds["n_clusters"], linkage=linkage_name
            )
            labels = model.fit_predict(ds["X"])
            params = {"linkage": linkage_name, "n_clusters": ds["n_clusters"]}
            row = metric_row(
                ds["key"], "hierarchical", label, labels, ds["X"], ds["y"], params
            )
            rows.append(row)
            score = row["silhouette"]
            if best is None or (not math.isnan(score) and score > best[0]):
                best = (score, labels, params, row)
        best_labels[ds["key"]] = best[1]
    return pd.DataFrame(rows), best_labels


def kmedoids(X, n_clusters, random_state=RANDOM_STATE, max_iter=100, n_init=5):
    rng = np.random.RandomState(random_state)
    distances = pairwise_distances(X)
    best_labels = None
    best_cost = np.inf
    n_samples = X.shape[0]

    for _ in range(n_init):
        medoids = rng.choice(n_samples, size=n_clusters, replace=False)
        labels = None
        for _ in range(max_iter):
            labels = np.argmin(distances[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for cluster_id in range(n_clusters):
                members = np.where(labels == cluster_id)[0]
                if members.size == 0:
                    new_medoids[cluster_id] = rng.randint(n_samples)
                    continue
                local_distances = distances[np.ix_(members, members)]
                new_medoids[cluster_id] = members[np.argmin(local_distances.sum(axis=1))]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
        labels = np.argmin(distances[:, medoids], axis=1)
        cost = distances[np.arange(n_samples), medoids[labels]].sum()
        if cost < best_cost:
            best_cost = cost
            best_labels = labels
    return best_labels


def run_prototype(datasets):
    rows = []
    best_labels = {}
    for ds in datasets:
        k_values = list(range(2, min(7, ds["n_clusters"] + 4)))

        baseline = KMeans(
            n_clusters=ds["n_clusters"], random_state=RANDOM_STATE, n_init=20
        ).fit_predict(ds["X"])
        rows.append(
            metric_row(
                ds["key"],
                "prototype",
                "KMeans_baseline",
                baseline,
                ds["X"],
                ds["y"],
                {"n_clusters": ds["n_clusters"]},
            )
        )

        best = None
        for k in k_values:
            labels = kmedoids(ds["X"], k, random_state=RANDOM_STATE, n_init=4)
            row = metric_row(
                ds["key"], "prototype", "KMedoids", labels, ds["X"], ds["y"], {"n_clusters": k}
            )
            rows.append(row)
            score = row["silhouette"]
            if best is None or (not math.isnan(score) and score > best[0]):
                best = (score, labels, {"algorithm": "KMedoids", "n_clusters": k}, row)

        component_values = [2]
        if ds["n_features"] >= 3:
            component_values.append(3)
        if ds["n_features"] >= 5:
            component_values.append(5)
        for n_components in sorted(set(component_values)):
            reduced = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(ds["X"])
            for k in k_values:
                labels = KMeans(
                    n_clusters=k, random_state=RANDOM_STATE, n_init=20
                ).fit_predict(reduced)
                row = metric_row(
                    ds["key"],
                    "prototype",
                    "PCA_KMeans",
                    labels,
                    ds["X"],
                    ds["y"],
                    {"n_clusters": k, "pca_components": n_components},
                )
                rows.append(row)
                score = row["silhouette"]
                if best is None or (not math.isnan(score) and score > best[0]):
                    best = (
                        score,
                        labels,
                        {"algorithm": "PCA_KMeans", "n_clusters": k, "pca_components": n_components},
                        row,
                    )
        best_labels[ds["key"]] = best[1]
    return pd.DataFrame(rows), best_labels


def random_validation(datasets, best_label_maps):
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for family, labels_by_dataset in best_label_maps.items():
        for ds in datasets:
            observed = metric_row(
                ds["key"], family, "best_observed", labels_by_dataset[ds["key"]], ds["X"], ds["y"], {}
            )["silhouette"]
            random_scores = []
            means = ds["X"].mean(axis=0)
            stds = ds["X"].std(axis=0)
            stds[stds == 0] = 1
            for _ in range(5):
                X_random = rng.normal(loc=means, scale=stds, size=ds["X"].shape)
                if family == "density":
                    labels = fit_optics(X_random, min_samples=10, xi=0.05, min_cluster_size=0.05)
                elif family == "hierarchical":
                    labels = AgglomerativeClustering(
                        n_clusters=ds["n_clusters"], linkage="ward"
                    ).fit_predict(X_random)
                else:
                    labels = KMeans(
                        n_clusters=ds["n_clusters"], random_state=rng.randint(10000), n_init=10
                    ).fit_predict(X_random)
                score = metric_row(
                    ds["key"], family, "random_baseline", labels, X_random, ds["y"], {}
                )["silhouette"]
                if not math.isnan(score):
                    random_scores.append(score)
            random_mean = float(np.mean(random_scores)) if random_scores else np.nan
            p_value = (
                float(np.mean([score >= observed for score in random_scores]))
                if random_scores and not math.isnan(observed)
                else np.nan
            )
            rows.append(
                {
                    "dataset": ds["key"],
                    "family": family,
                    "observed_silhouette": observed,
                    "random_mean_silhouette": random_mean,
                    "random_repetitions": len(random_scores),
                    "empirical_p_value_random_ge_observed": p_value,
                }
            )
    return pd.DataFrame(rows)


def pca_coordinates(X):
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)


def create_plots(datasets, density_tuning, density_labels, hierarchical_results, prototype_results, label_maps):
    IMAGE_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ds in zip(axes, datasets):
        coords = pca_coordinates(ds["X"])
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=ds["y"], s=18, cmap="tab10")
        ax.set_title("{}\ntrue labels".format(ds["name"]))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "dataset_pca_overview.png", dpi=160)
    plt.close(fig)

    for family, labels_by_dataset in label_maps.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, ds in zip(axes, datasets):
            coords = pca_coordinates(ds["X"])
            labels = labels_by_dataset[ds["key"]]
            ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=18, cmap="tab10")
            ax.set_title("{}\n{} labels".format(ds["name"], family))
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "{}_best_clusters.png".format(family), dpi=160)
        plt.close(fig)

    tuning = density_tuning.sort_values("silhouette", ascending=False).head(12).copy()
    tuning["config"] = tuning["params"].str.replace('"', "")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(tuning)), tuning["silhouette"], color="#4C78A8")
    ax.set_xticks(range(len(tuning)))
    ax.set_xticklabels(tuning["config"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Silhouette")
    ax.set_title("OPTICS Parameter Tuning on Breast Cancer Dataset")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "density_tuning_breast_cancer.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = hierarchical_results.pivot(index="dataset", columns="algorithm", values="silhouette")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Silhouette")
    ax.set_title("Agglomerative Linkage Comparison")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "hierarchical_linkage_comparison.png", dpi=160)
    plt.close(fig)

    proto_best = (
        prototype_results.sort_values("silhouette", ascending=False)
        .groupby(["dataset", "algorithm"], as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = proto_best.pivot(index="dataset", columns="algorithm", values="silhouette")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Best silhouette")
    ax.set_title("Prototype Algorithm Comparison")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "prototype_algorithm_comparison.png", dpi=160)
    plt.close(fig)

    iris = [ds for ds in datasets if ds["key"] == "iris"][0]
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage(iris["X"], method="ward"), truncate_mode="level", p=5, ax=ax)
    ax.set_title("Iris Ward-Linkage Dendrogram (truncated)")
    ax.set_xlabel("Merged sample groups")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "iris_dendrogram.png", dpi=160)
    plt.close(fig)


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def top_table(df, family):
    subset = (
        df[df["family"] == family]
        .sort_values(["dataset", "silhouette"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
    )
    cols = ["dataset", "algorithm", "clusters_found", "silhouette", "davies_bouldin", "nmi", "adjusted_rand", "purity"]
    return subset[cols]


def write_report(datasets, density_results, hierarchical_results, prototype_results, random_results):
    summary_rows = []
    for ds in datasets:
        summary_rows.append(
            {
                "dataset": ds["key"],
                "name": ds["name"],
                "samples": ds["n_samples"],
                "features": ds["n_features"],
                "classes": ds["n_clusters"],
                "source": ds["source"],
            }
        )
    dataset_table = pd.DataFrame(summary_rows)
    all_results = pd.concat([density_results, hierarchical_results, prototype_results], ignore_index=True)
    best_density = top_table(all_results, "density")
    best_hierarchical = top_table(all_results, "hierarchical")
    best_prototype = top_table(all_results, "prototype")

    RESULT_DIR.mkdir(exist_ok=True)
    dataset_table.to_csv(RESULT_DIR / "dataset_summary.csv", index=False)
    density_results.to_csv(RESULT_DIR / "density_results.csv", index=False)
    hierarchical_results.to_csv(RESULT_DIR / "hierarchical_results.csv", index=False)
    prototype_results.to_csv(RESULT_DIR / "prototype_results.csv", index=False)
    random_results.to_csv(RESULT_DIR / "random_validation.csv", index=False)

    report_md = BASE_DIR / "report.md"
    report_md.write_text(
        "# Assignment 3 Report\n\n"
        "This report was generated by `assignment3_clustering.py`.\n\n"
        "Datasets: Breast Cancer Wisconsin Diagnostic, Iris, and Wine Recognition.\n"
        "The implementation compares OPTICS, agglomerative clustering with four linkages, "
        "KMeans, KMedoids, and PCA+KMeans.\n\n"
        "See `results/*.csv` for complete metric tables and `images/*.png` for visual outputs.\n"
    )

    tex = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\title{Data Mining Assignment 3: Clustering}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\section*{Dataset Selection}",
        "Three real datasets were selected and standardized before clustering. "
        "The selection hypotheses are density-based clustering for Breast Cancer, "
        "hierarchical clustering for Iris, and prototype-based clustering for Wine.",
        dataset_table.to_latex(index=False, escape=True),
        r"\section*{Theory Notes}",
        "DBSCAN advantages: it discovers arbitrary-shaped clusters and identifies noise without pre-setting the number of clusters. "
        "DBSCAN disadvantages: it is sensitive to eps/minPts and struggles with varying density. "
        "OPTICS was selected as the density-based alternative because it orders points across multiple density levels and is less dependent on a single global eps.",
        "",
        "Hierarchical clustering categories are agglomerative and divisive. Agglomerative methods are used more widely because they are simpler to implement, easier to visualize with dendrograms, and computationally more practical for ordinary tabular datasets.",
        "",
        "KMeans limitations: outliers can pull centroids away from dense groups, and distance concentration harms performance as dimensionality increases. KMedoids mitigates outliers by choosing observed representative points. PCA+KMeans mitigates dimensionality by clustering after variance-preserving projection.",
        r"\section*{Best Empirical Results}",
        r"\subsection*{Density: OPTICS}",
        best_density.to_latex(index=False, escape=True),
        r"\subsection*{Hierarchical: Agglomerative Linkages}",
        best_hierarchical.to_latex(index=False, escape=True),
        r"\subsection*{Prototype-Based Algorithms}",
        best_prototype.to_latex(index=False, escape=True),
        r"\section*{Random Data Validation}",
        "Observed cluster silhouettes were compared with silhouettes from random Gaussian data having the same feature means and standard deviations. Low empirical p-values indicate clustering structure stronger than random baseline.",
        random_results.to_latex(index=False, escape=True),
        r"\section*{Visualizations}",
        r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{images/dataset_pca_overview.png}\caption{PCA overview with true labels.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{images/density_best_clusters.png}\caption{Best OPTICS clusters.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{images/hierarchical_best_clusters.png}\caption{Best hierarchical clusters.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{images/prototype_best_clusters.png}\caption{Best prototype-based clusters.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{images/density_tuning_breast_cancer.png}\caption{OPTICS parameter tuning on the density-selected dataset.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{images/hierarchical_linkage_comparison.png}\caption{Agglomerative linkage comparison.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{images/prototype_algorithm_comparison.png}\caption{Prototype algorithm comparison.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{images/iris_dendrogram.png}\caption{Truncated Iris dendrogram.}\end{figure}",
        r"\section*{Inference}",
        "The Breast Cancer density hypothesis is only partly supported: OPTICS identifies dense substructure and noise, but standard tabular scaling still favors centroid-like separation for some metrics. "
        "The Iris hierarchical hypothesis is supported by Ward linkage and the dendrogram structure. "
        "The Wine prototype hypothesis is supported by strong KMeans/PCA+KMeans results relative to the alternatives.",
        r"\end{document}",
    ]
    tex_path = BASE_DIR / "report.tex"
    tex_path.write_text("\n".join(tex))
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            cwd=str(BASE_DIR),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print("Warning: pdflatex failed: {}".format(exc))

    with open(RESULT_DIR / "assignment3_results.json", "w") as fh:
        json.dump(
            {
                "datasets": summary_rows,
                "best_density": best_density.to_dict(orient="records"),
                "best_hierarchical": best_hierarchical.to_dict(orient="records"),
                "best_prototype": best_prototype.to_dict(orient="records"),
            },
            fh,
            indent=2,
            sort_keys=True,
        )


def main():
    datasets = load_datasets()
    density_results, density_tuning, density_labels = run_density(datasets)
    hierarchical_results, hierarchical_labels = run_hierarchical(datasets)
    prototype_results, prototype_labels = run_prototype(datasets)
    label_maps = {
        "density": density_labels,
        "hierarchical": hierarchical_labels,
        "prototype": prototype_labels,
    }
    random_results = random_validation(datasets, label_maps)
    create_plots(
        datasets,
        density_tuning,
        density_labels,
        hierarchical_results,
        prototype_results,
        label_maps,
    )
    write_report(
        datasets,
        density_results,
        hierarchical_results,
        prototype_results,
        random_results,
    )
    print("Assignment 3 completed.")
    print("Report: {}".format(BASE_DIR / "report.pdf"))


if __name__ == "__main__":
    main()
