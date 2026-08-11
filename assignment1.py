"""COMP662 Assignment 1: synthetic bean data with a conditional VAE."""
from pathlib import Path
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


SOURCE_ROOT = Path(__file__).resolve().parent
KAGGLE_ROOT = Path("/kaggle")
ROOT = KAGGLE_ROOT / "working" if KAGGLE_ROOT.is_dir() else SOURCE_ROOT
DATA = KAGGLE_ROOT / "input" / "comp662-a1-bean-data" / "train.csv" if KAGGLE_ROOT.is_dir() else SOURCE_ROOT / "data" / "train.csv"
OUTPUT_DATA = ROOT / "data"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
SEED = 42


class ConditionalVAE(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, 4)
        self.encoder = nn.Sequential(nn.Linear(n_features + 4, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mean = nn.Linear(32, 8)
        self.log_var = nn.Linear(32, 8)
        self.decoder = nn.Sequential(nn.Linear(8 + 4, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, n_features))

    def forward(self, x, labels):
        conditioned = torch.cat([x, self.embedding(labels)], dim=1)
        hidden = self.encoder(conditioned)
        mean, log_var = self.mean(hidden), self.log_var(hidden)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        return self.decoder(torch.cat([latent, self.embedding(labels)], dim=1)), mean, log_var

    def sample(self, labels):
        latent = torch.randn(len(labels), 8, device=labels.device)
        return self.decoder(torch.cat([latent, self.embedding(labels)], dim=1))


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def save_eda(data, features):
    counts = data["Class"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts.plot.bar(ax=axes[0], title="Class counts", xlabel="Class", ylabel="Beans")
    (counts / len(data) * 100).plot.bar(ax=axes[1], title="Class percentages", xlabel="Class", ylabel="Percent")
    fig.tight_layout(); fig.savefig(FIGURES / "class_distribution.png", dpi=160); plt.close(fig)
    chosen = ["Area", "Perimeter", "MajorAxisLength", "Eccentricity"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for feature, ax in zip(chosen, axes.ravel()):
        for label, group in data.groupby("Class"):
            ax.hist(group[feature], bins=30, alpha=.35, label=str(label), density=True)
        ax.set_title(feature)
    axes[0, 0].legend(title="Class"); fig.tight_layout(); fig.savefig(FIGURES / "feature_distributions.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(data[features].corr(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=80, fontsize=7); ax.set_yticks(range(len(features)), features, fontsize=7)
    fig.colorbar(image, ax=ax, label="Correlation"); fig.tight_layout(); fig.savefig(FIGURES / "real_correlation.png", dpi=160); plt.close(fig)
    return counts


def classifier():
    return RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=SEED)


def evaluate(model, x_test, y_test, label):
    prediction = model.predict(x_test)
    return {"model": label, "macro_f1": f1_score(y_test, prediction, average="macro"), "accuracy": accuracy_score(y_test, prediction), "prediction": prediction}


def train_vae(x_train, y_train, n_classes):
    scaler = StandardScaler().fit(x_train)
    x_scaled = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
    labels = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    model = ConditionalVAE(x_train.shape[1], n_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(100):
        order = torch.randperm(len(x_scaled))
        for batch in order.split(256):
            reconstruction, mean, log_var = model(x_scaled[batch], labels[batch])
            reconstruction_loss = F.mse_loss(reconstruction, x_scaled[batch])
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            optimiser.zero_grad(); (reconstruction_loss + .02 * kl_loss).backward(); optimiser.step()
    return model.eval(), scaler


def generate_balanced(model, scaler, x_train, y_train):
    target = y_train.value_counts().max()
    generated_labels = np.concatenate([np.full(target - (y_train == label).sum(), label) for label in sorted(y_train.unique())])
    labels = torch.tensor(generated_labels, dtype=torch.long)
    with torch.no_grad():
        synthetic = scaler.inverse_transform(model.sample(labels).numpy())
    return pd.DataFrame(synthetic, columns=x_train.columns), pd.Series(generated_labels, name="Class")


def save_quality_figures(real, synthetic, features):
    chosen = ["Area", "Perimeter", "MajorAxisLength", "Eccentricity"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for feature, ax in zip(chosen, axes.ravel()):
        ax.hist(real[feature], bins=35, density=True, alpha=.55, label="real")
        ax.hist(synthetic[feature], bins=35, density=True, alpha=.55, label="synthetic")
        ax.set_title(feature)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(FIGURES / "real_vs_synthetic.png", dpi=160); plt.close(fig)
    correlation_gap = (real[features].corr() - synthetic[features].corr()).abs().to_numpy().mean()
    distances = NearestNeighbors(n_neighbors=2).fit(real[features]).kneighbors(real[features])[0][:, 1]
    synth_distances = NearestNeighbors(n_neighbors=1).fit(real[features]).kneighbors(synthetic[features])[0][:, 0]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(distances, bins=40, alpha=.6, label="real nearest neighbour"); ax.hist(synth_distances, bins=40, alpha=.6, label="synthetic to real")
    ax.legend(); ax.set_title("Nearest-neighbour distances"); fig.tight_layout(); fig.savefig(FIGURES / "nearest_neighbour_distances.png", dpi=160); plt.close(fig)
    return correlation_gap, float(np.median(distances)), float(np.median(synth_distances))


def main():
    set_seed(); OUTPUT_DATA.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    data = pd.read_csv(DATA)
    features = [column for column in data.columns if column != "Class"]
    assert data.shape == (12111, 11) and data["Class"].between(0, 4).all(), "Unexpected COMP662 dataset"
    counts = save_eda(data, features)
    x_train, x_test, y_train, y_test = train_test_split(data[features], data["Class"], test_size=.2, stratify=data["Class"], random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline = classifier()
    cv_f1 = cross_val_score(baseline, x_train, y_train, scoring="f1_macro", cv=cv, n_jobs=1)
    baseline.fit(x_train, y_train)
    baseline_result = evaluate(baseline, x_test, y_test, "Baseline random forest")
    vae, scaler = train_vae(x_train, y_train, len(counts))
    synthetic_x, synthetic_y = generate_balanced(vae, scaler, x_train, y_train)
    synthetic = synthetic_x.assign(Class=synthetic_y)
    synthetic.to_csv(OUTPUT_DATA / "synthetic_train.csv", index=False)
    correlation_gap, real_nn, synthetic_nn = save_quality_figures(x_train, synthetic_x, features)
    augmented = classifier()
    augmented.fit(pd.concat([x_train, synthetic_x], ignore_index=True), pd.concat([y_train, synthetic_y], ignore_index=True))
    augmented_result = evaluate(augmented, x_test, y_test, "Augmented random forest")
    results = pd.DataFrame([{key: value for key, value in row.items() if key != "prediction"} for row in [baseline_result, augmented_result]])
    results.to_csv(OUTPUT_DATA / "performance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); results.set_index("model")[["macro_f1", "accuracy"]].plot.bar(ax=ax, ylim=(0, 1), rot=0); fig.tight_layout(); fig.savefig(FIGURES / "performance_comparison.png", dpi=160); plt.close(fig)
    winner = augmented_result if augmented_result["macro_f1"] > baseline_result["macro_f1"] else baseline_result
    final_model = classifier()
    if winner is augmented_result:
        final_model.fit(pd.concat([data[features], synthetic_x], ignore_index=True), pd.concat([data["Class"], synthetic_y], ignore_index=True))
    else:
        final_model.fit(data[features], data["Class"])
    joblib.dump({"model": final_model, "features": features, "selected": winner["model"]}, MODELS / "1173808_Assignment1_final.joblib")
    report = f"""# COMP662 Assignment 1 results

## Task 1 — EDA

The dataset has {len(data):,} observations, 10 numerical predictors and five target classes. Class counts are {counts.to_dict()}. The class chart and four representative-feature distributions are in `figures/`. Correlation is assessed in `real_correlation.png`. Class imbalance makes accuracy alone misleading, so macro-F1 is primary and stratified splitting/CV preserve class proportions.

## Task 2 — Baseline

The baseline is a class-weighted random forest. It handles non-linear feature relationships and does not require feature scaling; scaling is fitted only for the VAE. A stratified 80/20 held-out test set is reserved before modelling, and five-fold stratified CV on the training set estimates development performance. Mean CV macro-F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}. Held-out macro-F1: {baseline_result['macro_f1']:.4f}; accuracy: {baseline_result['accuracy']:.4f}.

## Task 3 — Generative model

A conditional VAE is used because it models continuous tabular rows while conditioning generation on the five class labels. The encoder maps standardised features plus a 4-dimensional class embedding to an 8-dimensional Gaussian latent distribution; the decoder reconstructs features from a latent sample plus the embedding. Standardisation is fitted only on the training split. Risks include unrealistic feature combinations, mode collapse/limited diversity, and the VAE smoothing rare patterns.

## Task 4 — Quality assurance

The VAE trains for 100 epochs with reconstruction plus KL loss. It generates only enough minority-class rows to match the largest training class ({len(synthetic):,} rows). `real_vs_synthetic.png` compares feature distributions; the mean absolute correlation-matrix gap is {correlation_gap:.4f}. Median nearest-neighbour distance is {real_nn:.4f} for real rows and {synthetic_nn:.4f} from synthetic to real rows. These checks support inspection but do not prove privacy or absence of memorisation.

## Task 5 — Augmentation

The augmented classifier reuses exactly the baseline algorithm and hyperparameters, with no further tuning. Held-out macro-F1: {augmented_result['macro_f1']:.4f}; accuracy: {augmented_result['accuracy']:.4f}. The selected final approach is **{winner['model']}**, based on held-out macro-F1. Class-level reports can be regenerated by running `assignment1.py`.

## Task 6 — Hidden test

`models/1173808_Assignment1_final.joblib` stores the selected classifier and ordered feature names. Use `predict.py` to produce one `Class` prediction per input row.
"""
    (ROOT / "1173808_Assignment1_Report.md").write_text(report)
    (OUTPUT_DATA / "baseline_classification_report.txt").write_text(classification_report(y_test, baseline_result["prediction"]))
    (OUTPUT_DATA / "augmented_classification_report.txt").write_text(classification_report(y_test, augmented_result["prediction"]))
    print(results.to_string(index=False)); print(f"Saved {winner['model']}")


if __name__ == "__main__":
    main()
