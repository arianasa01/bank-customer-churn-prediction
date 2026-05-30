"""
Task 2: Model training and evaluation for customer churn prediction.

This script compares a small set of classification models and selects a compact
Random Forest model with feature selection. The feature selection step helps
reduce noise in a relatively small simulated dataset.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "model_ready_customer_churn_dataset.csv"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for folder in [PLOTS_DIR, MODELS_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numerical and categorical variables."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def evaluate_at_threshold(y_true, probabilities, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions),
        "f1_score": f1_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def choose_threshold_by_validation_f1(y_val, val_probabilities) -> float:
    """Choose a probability threshold using the validation set only."""
    precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_index])


def get_selected_feature_names(model: Pipeline) -> list[str]:
    """Return readable feature names after preprocessing and feature selection."""
    all_feature_names = model.named_steps["preprocess"].get_feature_names_out()
    all_feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in all_feature_names
    ]
    selector = model.named_steps.get("feature_selection")
    if selector is None:
        return all_feature_names
    selected_mask = selector.get_support()
    return [name for name, selected in zip(all_feature_names, selected_mask) if selected]


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=["CustomerID", "ChurnStatus"])
    y = data["ChurnStatus"].astype(int)

    preprocessor = make_preprocessor(X)

    # 60/20/20 split: train, validation, test. The test set is not used for threshold tuning.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                class_weight="balanced", C=0.1, max_iter=1000, random_state=42
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=75,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=20,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Random Forest + Feature Selection": Pipeline([
            ("preprocess", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    comparison_rows = []
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, pipeline in models.items():
        cv_scores = cross_val_score(
            pipeline, X_temp, y_temp, cv=cv, scoring="roc_auc", n_jobs=1
        )
        pipeline.fit(X_train, y_train)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        default_metrics = evaluate_at_threshold(y_test, test_probabilities, 0.5)
        comparison_rows.append({
            "model": model_name,
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
            "test_roc_auc": default_metrics["roc_auc"],
            "test_average_precision": default_metrics["average_precision"],
            "accuracy_at_0_50": default_metrics["accuracy"],
            "precision_at_0_50": default_metrics["precision"],
            "recall_at_0_50": default_metrics["recall"],
            "f1_at_0_50": default_metrics["f1_score"],
        })
        trained_models[model_name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "test_roc_auc", ascending=False
    )
    comparison.to_csv(OUTPUTS_DIR / "model_comparison_metrics.csv", index=False)

    selected_model_name = "Random Forest + Feature Selection"
    selected_model = trained_models[selected_model_name]

    validation_probabilities = selected_model.predict_proba(X_val)[:, 1]
    selected_threshold = choose_threshold_by_validation_f1(y_val, validation_probabilities)

    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    default_metrics = evaluate_at_threshold(y_test, test_probabilities, 0.5)
    selected_metrics = evaluate_at_threshold(y_test, test_probabilities, selected_threshold)

    metrics_table = pd.DataFrame([
        {"model": selected_model_name, "threshold_type": "default", **default_metrics},
        {"model": selected_model_name, "threshold_type": "validation_selected", **selected_metrics},
    ])
    metrics_table.to_csv(OUTPUTS_DIR / "selected_model_test_metrics.csv", index=False)

    predictions = data.loc[X_test.index, ["CustomerID", "ChurnStatus"]].copy()
    predictions["predicted_churn_probability"] = test_probabilities
    predictions["predicted_churn"] = (test_probabilities >= selected_threshold).astype(int)
    predictions = predictions.sort_values("predicted_churn_probability", ascending=False)
    predictions.to_csv(OUTPUTS_DIR / "test_set_churn_predictions.csv", index=False)

    predictions["risk_tier"] = pd.qcut(
        predictions["predicted_churn_probability"].rank(method="first", ascending=False),
        q=4,
        labels=["Highest risk", "High risk", "Medium risk", "Lowest risk"],
    )
    risk_summary = predictions.groupby("risk_tier", observed=False).agg(
        customers=("CustomerID", "count"),
        churned_customers=("ChurnStatus", "sum"),
        churn_rate=("ChurnStatus", "mean"),
        average_predicted_probability=("predicted_churn_probability", "mean"),
    ).reset_index()
    risk_summary.to_csv(OUTPUTS_DIR / "risk_tier_summary.csv", index=False)

    selected_feature_names = get_selected_feature_names(selected_model)
    importances = selected_model.named_steps["model"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": selected_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    feature_importance.to_csv(
        OUTPUTS_DIR / "selected_model_feature_importance.csv", index=False
    )

    joblib.dump(
        {
            "pipeline": selected_model,
            "threshold": selected_threshold,
            "selected_model_name": selected_model_name,
            "features": X.columns.tolist(),
        },
        MODELS_DIR / "smartbank_churn_random_forest_feature_selected.pkl",
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {selected_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - selected model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve_selected_model.png", dpi=200)
    plt.close()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {selected_metrics['average_precision']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve - selected model")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve_selected_model.png", dpi=200)
    plt.close()

    # Confusion matrix at the selected threshold
    selected_predictions = (test_probabilities >= selected_threshold).astype(int)
    cm = confusion_matrix(y_test, selected_predictions)
    display = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    display.plot(values_format="d")
    plt.title("Confusion matrix - selected threshold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_selected_model.png", dpi=200)
    plt.close()

    # Feature importance
    top_features = feature_importance.head(12).sort_values("importance")
    top_features.plot(kind="barh", x="feature", y="importance", legend=False, figsize=(8, 6))
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top feature importance values")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_selected_model.png", dpi=200)
    plt.close()

    # Model comparison chart
    comparison.sort_values("test_roc_auc").plot(
        kind="barh", x="model", y="test_roc_auc", legend=False, figsize=(8, 5)
    )
    plt.xlabel("Hold-out test ROC-AUC")
    plt.ylabel("Model")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison_roc_auc.png", dpi=200)
    plt.close()

    # Risk tier chart
    risk_summary.plot(kind="bar", x="risk_tier", y="churn_rate", legend=False, figsize=(7, 4))
    plt.ylabel("Observed churn rate")
    plt.xlabel("Risk tier")
    plt.title("Observed churn rate by model risk tier")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "risk_tier_churn_rate.png", dpi=200)
    plt.close()

    print("Task 2 outputs saved.")
    print(metrics_table)


if __name__ == "__main__":
    main()
