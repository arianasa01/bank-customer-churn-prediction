"""
Task 1: Data gathering, exploratory analysis and preprocessing.

This script builds a customer-level churn dataset from the original multi-sheet
Excel workbook. It also creates simple EDA charts used in the report.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Customer_Churn_Data_Large.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "plots"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_source_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Load all source sheets from the churn workbook."""
    return {
        "demographics": pd.read_excel(path, sheet_name="Customer_Demographics"),
        "transactions": pd.read_excel(path, sheet_name="Transaction_History"),
        "service": pd.read_excel(path, sheet_name="Customer_Service"),
        "online": pd.read_excel(path, sheet_name="Online_Activity"),
        "churn": pd.read_excel(path, sheet_name="Churn_Status"),
    }


def build_customer_level_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate behavioural tables and merge them into one row per customer."""
    demographics = tables["demographics"].copy()
    transactions = tables["transactions"].copy()
    service = tables["service"].copy()
    online = tables["online"].copy()
    churn = tables["churn"].copy()

    # Transaction behaviour
    max_transaction_date = transactions["TransactionDate"].max()
    transactions["transaction_weekday"] = transactions["TransactionDate"].dt.weekday

    transaction_agg = transactions.groupby("CustomerID").agg(
        transaction_count=("TransactionID", "count"),
        total_spent=("AmountSpent", "sum"),
        avg_spent=("AmountSpent", "mean"),
        median_spent=("AmountSpent", "median"),
        max_spent=("AmountSpent", "max"),
        min_spent=("AmountSpent", "min"),
        spend_std=("AmountSpent", "std"),
        first_transaction_date=("TransactionDate", "min"),
        last_transaction_date=("TransactionDate", "max"),
        unique_product_categories=("ProductCategory", "nunique"),
        weekend_transaction_rate=("transaction_weekday", lambda x: np.mean(x >= 5)),
    ).reset_index()

    transaction_agg["spend_std"] = transaction_agg["spend_std"].fillna(0)
    transaction_agg["days_since_last_transaction"] = (
        max_transaction_date - transaction_agg["last_transaction_date"]
    ).dt.days
    transaction_agg["transaction_history_days"] = (
        transaction_agg["last_transaction_date"] - transaction_agg["first_transaction_date"]
    ).dt.days
    transaction_agg["transactions_per_active_day"] = (
        transaction_agg["transaction_count"] / (transaction_agg["transaction_history_days"] + 1)
    )
    transaction_agg = transaction_agg.drop(
        columns=["first_transaction_date", "last_transaction_date"]
    )

    transaction_category_counts = (
        pd.crosstab(transactions["CustomerID"], transactions["ProductCategory"])
        .add_prefix("tx_category_count_")
        .reset_index()
    )

    # Customer service behaviour
    service["is_unresolved"] = (service["ResolutionStatus"] == "Unresolved").astype(int)
    service["is_complaint"] = (service["InteractionType"] == "Complaint").astype(int)
    max_service_date = service["InteractionDate"].max()

    service_agg = service.groupby("CustomerID").agg(
        service_interaction_count=("InteractionID", "count"),
        unresolved_interaction_count=("is_unresolved", "sum"),
        complaint_count=("is_complaint", "sum"),
        first_service_interaction=("InteractionDate", "min"),
        last_service_interaction=("InteractionDate", "max"),
    ).reset_index()

    service_agg["unresolved_rate"] = (
        service_agg["unresolved_interaction_count"]
        / service_agg["service_interaction_count"]
    )
    service_agg["days_since_last_service_interaction"] = (
        max_service_date - service_agg["last_service_interaction"]
    ).dt.days
    service_agg["service_history_days"] = (
        service_agg["last_service_interaction"] - service_agg["first_service_interaction"]
    ).dt.days
    service_agg = service_agg.drop(
        columns=["first_service_interaction", "last_service_interaction"]
    )

    service_type_counts = (
        pd.crosstab(service["CustomerID"], service["InteractionType"])
        .add_prefix("service_type_count_")
        .reset_index()
    )
    service_resolution_counts = (
        pd.crosstab(service["CustomerID"], service["ResolutionStatus"])
        .add_prefix("service_resolution_count_")
        .reset_index()
    )

    # Online behaviour
    max_login_date = online["LastLoginDate"].max()
    online["days_since_last_login"] = (max_login_date - online["LastLoginDate"]).dt.days
    online = online.drop(columns=["LastLoginDate"])

    model_data = (
        demographics
        .merge(transaction_agg, on="CustomerID", how="left")
        .merge(transaction_category_counts, on="CustomerID", how="left")
        .merge(service_agg, on="CustomerID", how="left")
        .merge(service_type_counts, on="CustomerID", how="left")
        .merge(service_resolution_counts, on="CustomerID", how="left")
        .merge(online, on="CustomerID", how="left")
        .merge(churn, on="CustomerID", how="left")
    )

    # Missing values occur after aggregation when a customer has no record in a table.
    numeric_cols = model_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["CustomerID", "ChurnStatus", "Age", "LoginFrequency"]:
            model_data[col] = model_data[col].fillna(0)

    # Behavioural ratios are useful because they make activity levels comparable.
    model_data["avg_spend_per_login"] = (
        model_data["total_spent"] / model_data["LoginFrequency"].replace(0, np.nan)
    )
    model_data["avg_spend_per_login"] = model_data["avg_spend_per_login"].fillna(
        model_data["total_spent"]
    )
    model_data["service_interactions_per_transaction"] = (
        model_data["service_interaction_count"]
        / model_data["transaction_count"].replace(0, np.nan)
    ).fillna(0)
    model_data["complaints_per_transaction"] = (
        model_data["complaint_count"] / model_data["transaction_count"].replace(0, np.nan)
    ).fillna(0)
    model_data["low_login_frequency_flag"] = (model_data["LoginFrequency"] <= 9).astype(int)

    return model_data


def save_eda_outputs(df: pd.DataFrame) -> None:
    """Create a small set of clear EDA charts."""
    # Churn distribution
    churn_counts = df["ChurnStatus"].value_counts().sort_index()
    churn_counts.plot(kind="bar", figsize=(6, 4))
    plt.xlabel("Churn status")
    plt.ylabel("Customer count")
    plt.title("Churn distribution")
    plt.xticks([0, 1], ["Stayed", "Churned"], rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task1_churn_distribution.png", dpi=200)
    plt.close()

    # Churn by income level
    df.groupby("IncomeLevel")["ChurnStatus"].mean().sort_values().plot(
        kind="bar", figsize=(6, 4)
    )
    plt.ylabel("Churn rate")
    plt.xlabel("Income level")
    plt.title("Churn rate by income level")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task1_churn_by_income.png", dpi=200)
    plt.close()

    # Login frequency band
    login_band = pd.cut(
        df["LoginFrequency"],
        bins=[-1, 9, 19, 29, 39, 49],
        labels=["0-9", "10-19", "20-29", "30-39", "40-49"],
    )
    df.assign(login_band=login_band).groupby("login_band", observed=False)[
        "ChurnStatus"
    ].mean().plot(kind="bar", figsize=(7, 4))
    plt.ylabel("Churn rate")
    plt.xlabel("Login frequency band")
    plt.title("Churn rate by login frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task1_churn_by_login_band.png", dpi=200)
    plt.close()

    # Mean login frequency by churn
    df.groupby("ChurnStatus")["LoginFrequency"].mean().plot(kind="bar", figsize=(6, 4))
    plt.xlabel("Churn status")
    plt.ylabel("Mean login frequency")
    plt.title("Average login frequency by churn status")
    plt.xticks([0, 1], ["Stayed", "Churned"], rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task1_mean_login_by_churn.png", dpi=200)
    plt.close()


def main() -> None:
    tables = load_source_tables(RAW_FILE)
    model_data = build_customer_level_dataset(tables)
    output_path = PROCESSED_DIR / "model_ready_customer_churn_dataset.csv"
    model_data.to_csv(output_path, index=False)
    save_eda_outputs(model_data)
    print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    main()
