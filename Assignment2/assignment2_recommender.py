#!/usr/bin/python3
"""Assignment 2: Collaborative filtering on the ModCloth ratings dataset."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"
DATA_URL = "https://raw.githubusercontent.com/MengtingWan/marketBias/master/data/df_modcloth.csv"
DATA_FILE = DATA_DIR / "df_modcloth.csv"
RANDOM_STATE = 42


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_FILE.exists():
        return DATA_FILE
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    DATA_FILE.write_bytes(response.content)
    return DATA_FILE


def load_dataset():
    download_dataset()
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["user_id", "item_id", "rating"]).copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    return df


class ItemBasedCollaborativeFilter:
    """Adjusted-cosine item-based collaborative filtering with mean fallbacks."""

    def __init__(self, top_k=30):
        self.top_k = top_k

    def fit(self, ratings):
        grouped = (
            ratings.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
        )
        self.global_mean = float(grouped["rating"].mean())
        self.user_means = grouped.groupby("user_id")["rating"].mean().to_dict()
        self.item_means = grouped.groupby("item_id")["rating"].mean().to_dict()

        self.users = sorted(grouped["user_id"].unique())
        self.items = sorted(grouped["item_id"].unique())
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.items)}
        self.user_mean_array = np.array(
            [self.user_means[user] for user in self.users], dtype=float
        )

        rows = grouped["user_id"].map(self.user_to_idx).values
        cols = grouped["item_id"].map(self.item_to_idx).values
        values = grouped["rating"].values.astype(float)
        self.rating_matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(len(self.users), len(self.items))
        )

        centered_values = values - self.user_mean_array[rows]
        centered = sparse.csr_matrix(
            (centered_values, (rows, cols)), shape=self.rating_matrix.shape
        )
        self.item_similarity = cosine_similarity(centered.T, dense_output=True)
        np.fill_diagonal(self.item_similarity, 0.0)
        return self

    def _fallback(self, user_id, item_id):
        if user_id in self.user_means and item_id in self.item_means:
            return 0.5 * self.user_means[user_id] + 0.5 * self.item_means[item_id]
        if user_id in self.user_means:
            return self.user_means[user_id]
        if item_id in self.item_means:
            return self.item_means[item_id]
        return self.global_mean

    def predict_one(self, user_id, item_id):
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return float(np.clip(self._fallback(user_id, item_id), 1, 5))

        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        row = self.rating_matrix.getrow(user_idx)
        rated_items = row.indices
        rated_values = row.data

        if rated_items.size == 0:
            return float(np.clip(self._fallback(user_id, item_id), 1, 5))

        similarities = self.item_similarity[item_idx, rated_items]
        mask = np.abs(similarities) > 1e-12
        if not np.any(mask):
            return float(np.clip(self._fallback(user_id, item_id), 1, 5))

        similarities = similarities[mask]
        rated_values = rated_values[mask]
        if similarities.size > self.top_k:
            keep = np.argsort(np.abs(similarities))[-self.top_k :]
            similarities = similarities[keep]
            rated_values = rated_values[keep]

        baseline = self.user_mean_array[user_idx]
        numerator = np.dot(similarities, rated_values - baseline)
        denominator = np.sum(np.abs(similarities))
        if denominator == 0:
            prediction = self._fallback(user_id, item_id)
        else:
            prediction = baseline + numerator / denominator
        return float(np.clip(prediction, 1, 5))

    def predict(self, ratings):
        return np.array(
            [
                self.predict_one(user_id, item_id)
                for user_id, item_id in zip(ratings["user_id"], ratings["item_id"])
            ]
        )


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cold_start_summary(train_df, eval_df):
    train_users = set(train_df["user_id"])
    train_items = set(train_df["item_id"])
    cold_users = int((~eval_df["user_id"].isin(train_users)).sum())
    cold_items = int((~eval_df["item_id"].isin(train_items)).sum())
    return {
        "rows": int(len(eval_df)),
        "cold_user_rows": cold_users,
        "cold_item_rows": cold_items,
        "cold_user_pct": float(cold_users / float(len(eval_df)) if len(eval_df) else 0),
        "cold_item_pct": float(cold_items / float(len(eval_df)) if len(eval_df) else 0),
    }


def run_experiment(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["rating"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    fold_rows = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_idx, val_idx) in enumerate(kfold.split(train_df), start=1):
        fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        model = ItemBasedCollaborativeFilter(top_k=30).fit(fit_df)
        preds = model.predict(val_df)
        metrics = evaluate_predictions(val_df["rating"].values, preds)
        metrics.update({"fold": fold})
        metrics.update(cold_start_summary(fit_df, val_df))
        fold_rows.append(metrics)
        print("Fold {} MAE={:.4f} RMSE={:.4f}".format(fold, metrics["mae"], metrics["rmse"]))

    final_model = ItemBasedCollaborativeFilter(top_k=30).fit(train_df)
    test_predictions = final_model.predict(test_df)
    test_metrics = evaluate_predictions(test_df["rating"].values, test_predictions)
    test_metrics.update(cold_start_summary(train_df, test_df))

    baseline_pred = np.repeat(train_df["rating"].mean(), len(test_df))
    baseline_metrics = evaluate_predictions(test_df["rating"].values, baseline_pred)

    predictions = test_df[["user_id", "item_id", "rating"]].copy()
    predictions["prediction"] = test_predictions
    predictions["absolute_error"] = np.abs(predictions["rating"] - predictions["prediction"])

    fold_results = pd.DataFrame(fold_rows)
    return train_df, test_df, fold_results, test_metrics, baseline_metrics, predictions


def create_plots(df, fold_results, test_metrics, baseline_metrics, predictions):
    IMAGE_DIR.mkdir(exist_ok=True)

    rating_counts = df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rating_counts.index.astype(str), rating_counts.values, color="#4C78A8")
    ax.set_title("ModCloth Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Interactions")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "rating_distribution.png", dpi=160)
    plt.close(fig)

    stats = pd.Series(
        {
            "interactions": len(df),
            "users": df["user_id"].nunique(),
            "items": df["item_id"].nunique(),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(stats.index, stats.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Dataset Scale")
    ax.set_ylabel("Count")
    for idx, value in enumerate(stats.values):
        ax.text(idx, value, "{:,}".format(int(value)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "dataset_scale.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fold_results["fold"], fold_results["mae"], marker="o", label="CV MAE")
    ax.plot(fold_results["fold"], fold_results["rmse"], marker="o", label="CV RMSE")
    ax.axhline(test_metrics["mae"], color="#E45756", linestyle="--", label="Test MAE")
    ax.axhline(test_metrics["rmse"], color="#72B7B2", linestyle="--", label="Test RMSE")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Error")
    ax.set_title("5-Fold Validation and Test Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "validation_test_errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    sample = predictions.sample(min(len(predictions), 5000), random_state=RANDOM_STATE)
    ax.scatter(sample["rating"], sample["prediction"], s=8, alpha=0.25, color="#4C78A8")
    ax.plot([1, 5], [1, 5], color="#E45756", linewidth=2)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlabel("Actual rating")
    ax.set_ylabel("Predicted rating")
    ax.set_title("Actual vs Predicted Ratings on Test Set")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["CF Test MAE", "Mean Baseline MAE", "CF Test RMSE", "Mean Baseline RMSE"]
    values = [
        test_metrics["mae"],
        baseline_metrics["mae"],
        test_metrics["rmse"],
        baseline_metrics["rmse"],
    ]
    ax.bar(labels, values, color=["#4C78A8", "#BAB0AC", "#F58518", "#BAB0AC"])
    ax.set_title("Collaborative Filtering vs Mean Baseline")
    ax.set_ylabel("Error")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "baseline_comparison.png", dpi=160)
    plt.close(fig)


def save_results(df, train_df, test_df, fold_results, test_metrics, baseline_metrics, predictions):
    RESULT_DIR.mkdir(exist_ok=True)
    fold_results.to_csv(RESULT_DIR / "cross_validation_results.csv", index=False)
    predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    summary = {
        "dataset": {
            "source": DATA_URL,
            "rows_after_cleaning": int(len(df)),
            "users": int(df["user_id"].nunique()),
            "items": int(df["item_id"].nunique()),
            "sparsity": float(
                1.0
                - len(df)
                / float(df["user_id"].nunique() * df["item_id"].nunique())
            ),
        },
        "split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_pct": float(len(train_df) / float(len(df))),
            "test_pct": float(len(test_df) / float(len(df))),
            "strategy": "85/15 stratified split by rating, with explicit cold-start fallbacks",
        },
        "model": {
            "name": "Item-based collaborative filtering",
            "similarity": "Adjusted cosine similarity over user-centered item vectors",
            "top_k_neighbors": 30,
            "fallback": "user/item/global mean fallback for cold-start or zero-similarity rows",
        },
        "validation": {
            "mae_mean": float(fold_results["mae"].mean()),
            "mae_std": float(fold_results["mae"].std()),
            "rmse_mean": float(fold_results["rmse"].mean()),
            "rmse_std": float(fold_results["rmse"].std()),
        },
        "test": test_metrics,
        "mean_baseline_test": baseline_metrics,
    }
    with open(RESULT_DIR / "assignment2_results.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def main():
    df = load_dataset()
    train_df, test_df, fold_results, test_metrics, baseline_metrics, predictions = run_experiment(df)
    create_plots(df, fold_results, test_metrics, baseline_metrics, predictions)
    summary = save_results(
        df, train_df, test_df, fold_results, test_metrics, baseline_metrics, predictions
    )
    print("Assignment 2 completed.")
    print(
        "CV MAE={:.4f} (+/- {:.4f}), CV RMSE={:.4f} (+/- {:.4f})".format(
            summary["validation"]["mae_mean"],
            summary["validation"]["mae_std"],
            summary["validation"]["rmse_mean"],
            summary["validation"]["rmse_std"],
        )
    )
    print(
        "Test MAE={:.4f}, Test RMSE={:.4f}".format(
            summary["test"]["mae"], summary["test"]["rmse"]
        )
    )


if __name__ == "__main__":
    main()
