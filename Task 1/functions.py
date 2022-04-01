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

    Args:
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
        "ROC": roc_auc_score(
            y, y_pred_proba, average="macro", multi_class="ovr"
        ),
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average="macro"),
        "Recall": recall_score(y, y_pred, average="macro"),
    }

    if print_metrics:
        print("\n --- MACRO METRICS --- ")
        for k, v in metrics.items():
            print(f"{k.upper()}: {round(v, 5)}")
    else:
        return metrics


def evaluate_model_on_ttv(model, datasets: dict, eval_val: bool = False):
    """Evaluates model for each dataset and produces a dataframe displaying
    the metrics

    Args:
        model: model to evaluate
        datasets (dict): dictionary of the split datasets
        eval_val (bool): flags if we also want validation tested.
            Defaults to False.

    Returns:
        _type_: _description_
    """

    train_data = datasets["train"]
    test_data = datasets["test"]
    val_data = datasets["val"]

    model_metrics = {
        "Train": evaluate_model(model, **train_data),
        "Test": evaluate_model(model, **test_data),
    }

    if eval_val:
        model_metrics["Validation"] = (evaluate_model(model, **val_data),)

    print("--- Model Metrics for Datasets --- ")

    # Return metrics in a dataframe
    return pd.DataFrame(model_metrics).T
