# Model Card - Customer Churn Prediction

## Model name

SmartBank churn risk model - Random Forest with feature selection

## Intended use

The model is designed to rank customers by their likelihood of churn. The output can support customer retention planning, especially when the action is low-risk and customer-friendly, such as a follow-up message, product support, or service review.

## Not intended for

The model should not be used to make automated decisions that could negatively affect customers. It is not suitable for credit decisions, pricing decisions, or any action that requires a regulated decision process.

## Data used

The model uses a simulated customer churn dataset containing:

- customer demographics
- transaction behaviour
- customer service interactions
- online activity
- churn status

## Selected model

Random Forest with feature selection.

This model was chosen because it can capture non-linear patterns, handle mixed customer behaviour variables, and provide feature importance values that are easier to explain than many black-box models.

## Main test metrics

| Metric | Result |
|---|---:|
| ROC-AUC | 0.632 |
| Average precision | 0.299 |
| Precision | 0.276 |
| Recall | 0.829 |
| F1-score | 0.415 |

The selected threshold was chosen using the validation set to improve the balance between precision and recall. Recall was prioritised because missing a customer who is likely to churn can be costly for a retention project.

## Key limitations

- The dataset is simulated and relatively small.
- The model signal is moderate rather than strong.
- The model may produce false positives, so retention actions should be low-cost and low-risk.
- The model has not been tested for fairness across demographic groups.
- The model would need monitoring over time because customer behaviour can change.

## Recommended next steps

- Add richer behaviour data, such as monthly product usage, balance trends, complaint text, and campaign history.
- Test more models such as XGBoost or LightGBM with careful validation.
- Add calibration so the predicted probabilities are easier to interpret.
- Monitor performance after deployment and retrain when the data distribution changes.
- Review fairness across age, gender, and income groups before any real business use.
