from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    output_folder: Path,
    threshold: int,
) -> None:
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=[f"< {threshold}", f">= {threshold}"]).plot(ax=ax)
    ax.set_title(f"Confusion matrix (threshold={threshold} mmHg)")
    fig.tight_layout()
    fig.savefig(output_folder / "figures" / f"confusion_matrix_{threshold}.png")
    plt.close(fig)


def plot_figures(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    df: pd.DataFrame,
    output_folder: Path,
    split: str,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4)
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("True HVPG (mmHg)")
    ax.set_ylabel("Predicted HVPG (mmHg)")
    ax.set_title(f"Predicted vs True HVPG — {split}")
    fig.tight_layout()
    fig.savefig(output_folder / "figures" / f"scatter_{split}.png")
    plt.close(fig)
