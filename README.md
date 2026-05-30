# Customer Churn Prediction - Lloyds Banking Group Forage Simulation

This project was completed as part of the Lloyds Banking Group Data Science Job Simulation on Forage. The task was to explore customer behaviour and build a machine learning model that can help identify customers who may be at risk of leaving.

The project follows a full data science workflow: data gathering, exploratory analysis, preprocessing, model training, model evaluation, and business recommendations.

## Project objective

The main goal was to support customer retention. Instead of only looking at past churn, the model ranks customers by their predicted churn risk. This could help a bank focus retention activity on customers who may need earlier support.

## What I did

- Combined customer demographics, transaction history, customer service data, online activity, and churn status into one customer-level dataset.
- Created behavioural features such as transaction count, total spend, login frequency, unresolved interaction rate, and recency measures.
- Explored churn patterns using charts and summary statistics.
- Built and compared classification models, including Logistic Regression, Gradient Boosting, Random Forest, and a Random Forest with feature selection.
- Evaluated the model using ROC-AUC, average precision, precision, recall, F1-score, and a confusion matrix.
- Converted the model output into risk tiers so the result is easier to use in a business context.

## Key results

The best hold-out test result came from a Random Forest model with feature selection.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.632 |
| Average precision | 0.299 |
| Precision at selected threshold | 0.276 |
| Recall at selected threshold | 0.829 |
| F1-score at selected threshold | 0.415 |

The ROC-AUC is modest, but it is still better than a random classifier. In a churn setting, this model is more useful as an early risk-ranking tool than as a final automated decision system.

At the selected threshold, the model identified around 83% of churned customers in the test set, but with a clear trade-off: it also produced false positives. In a real bank, this would be acceptable only if the retention action is low-cost, such as a personalised check-in, product education, or service follow-up.

## Business interpretation

The model suggests that churn risk is linked more strongly to customer behaviour than to demographics alone. Important variables included transaction history, service interaction history, login frequency, spend patterns, and recency measures.

A practical use case would be to score customers each month and group them into risk tiers. The bank could then review the highest-risk group first and choose retention actions based on customer value, complaint history, and product usage.

## Project structure

```text
lloyds-customer-churn-prediction/
├── data/
│   ├── raw/
│   │   └── Customer_Churn_Data_Large.xlsx
│   └── processed/
│       └── model_ready_customer_churn_dataset.csv
├── models/
│   └── smartbank_churn_random_forest_feature_selected.pkl
├── outputs/
│   ├── model_comparison_metrics.csv
│   ├── selected_model_test_metrics.csv
│   ├── selected_model_feature_importance.csv
│   ├── risk_tier_summary.csv
│   └── test_set_churn_predictions.csv
├── plots/
│   ├── model_comparison_roc_auc.png
│   ├── roc_curve_selected_model.png
│   ├── precision_recall_curve_selected_model.png
│   ├── confusion_matrix_selected_model.png
│   ├── feature_importance_selected_model.png
│   └── risk_tier_churn_rate.png
├── reports/
│   ├── customer_churn_task1_eda_report.docx
│   ├── customer_churn_task2_model_report.docx
│   └── customer_churn_task2_model_report.pdf
├── src/
│   ├── task1_eda_preprocessing.py
│   └── task2_model_training.py
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

## How to run the project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the preprocessing and EDA script:

```bash
python src/task1_eda_preprocessing.py
```

Run the model training and evaluation script:

```bash
python src/task2_model_training.py
```

The scripts will recreate the processed dataset, charts, metrics, predictions, feature importance file, and trained model.

## Important note

This is a fictional simulation dataset. The model should not be treated as a production banking model. A real version would need more customer history, stronger validation, fairness checks, monitoring, and approval from risk and compliance teams before use.
