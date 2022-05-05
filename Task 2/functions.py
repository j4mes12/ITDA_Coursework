import pandas as pd
from sklearn_crfsuite.metrics import (
    flat_f1_score,
    flat_accuracy_score,
    flat_precision_score,
    flat_recall_score,
)


def calculate_metrics(labels, args):

    opt = {
        "F1 Score": flat_f1_score(**args, average="weighted", labels=labels),
        "Accuracy": flat_accuracy_score(**args),
        "Precision": flat_precision_score(
            **args, average="weighted", labels=labels
        ),
        "Recall": flat_recall_score(**args, average="weighted", labels=labels),
    }

    return opt


def evaluate_model(
    model, X_train, y_train, X_val, y_val, X_test=None, y_test=None
):

    labels = list(model.classes_)
    labels.remove("O")

    train_args = {
        "y_true": y_train,
        "y_pred": model.predict(X_train),
    }
    val_args = {
        "y_true": y_val,
        "y_pred": model.predict(X_val),
    }

    metrics = {
        "Train": calculate_metrics(labels, train_args),
        "Validation": calculate_metrics(labels, val_args),
    }

    if (X_test is not None) & (y_test is not None):

        test_args = {
            "y_true": y_test,
            "y_pred": model.predict(X_test),
        }

        metrics["Test"] = calculate_metrics(labels, test_args)

    return pd.DataFrame(metrics).T
