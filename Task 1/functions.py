import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    classification_report,
)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y,
    c_report: bool = False,
    print_metrics: bool = False,
):
    """Produces dictionary of the model metrics that we will be using for
    comparison. Also allows classification report to be printed and metrics
    printed as well.

    Arguments:
        model: model to evaluate
        X (pd.DataFrame): dataframe to create predictions
        y : used to compare prediction against these true values
        c_report (bool): boolean if to print classification report
        print_metrics (bool): boolean if to print metrics

    Returns:
        metrics (dict): dictionary of model metrics
    """

    # Calculates Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    if c_report:
        print(" --- Classification Report --- ")
        print(classification_report(y, y_pred))

    # Calculates model metrics: roc, accuracy, precision, recall
    metrics = {
        "roc": roc_auc_score(
            y, y_pred_proba, average="macro", multi_class="ovr"
        ),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="macro"),
        "recall": recall_score(y, y_pred, average="macro"),
    }

    if print_metrics:
        print("\n --- MACRO METRICS --- ")
        for k, v in metrics.items():
            print(f"{k.upper()}: {round(v, 5)}")
    else:
        return metrics
