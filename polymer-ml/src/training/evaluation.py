import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "RMSE": rmse}


def evaluate_classification(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {"report": report, "cm": cm}


def save_confusion_matrix(cm, labels, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix (PolymerClass)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
