import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import bootstrap, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

seed = 42  # bootsrapping has no random state => seed set here
random.seed(seed)
np.random.seed(seed)


def sensitivity_score(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)
    except ValueError:
        # if the bootstrapping outputs a group of samples with all the same ground_truth value,
        # the confusion matrix will be ValueError.
        return np.nan


def find_optimal_threshold(y_true_labels, y_pred_continuous, threshold, split, output_folder):
    """
    Finds the threshold that maximizes the Youden Index.
    y_true_labels: binary ground truth (e.g., hvpg >= 10)
    y_pred_continuous: continuous predictions from your model
    """

    fpr, tpr, thresholds = roc_curve(y_true_labels, y_pred_continuous)
    roc_auc = roc_auc_score(y_true_labels, y_pred_continuous)

    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve - AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.axvline(x=fpr[best_idx], color="r", linestyle="--", label=f"Threshold {thresholds[best_idx]:.2f}")
    plt.axhline(y=tpr[best_idx], color="r", linestyle="--", label=f"Youden Index {j_scores[best_idx]:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for threshold {threshold} and split {split}")
    plt.legend()
    output_file_path = output_folder / "figures" / f"roc_curve_{threshold}_{split}.png"
    os.makedirs(output_file_path.parent, exist_ok=True)
    plt.savefig(output_file_path)

    return thresholds[best_idx]


def specificity_score(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return np.nan

    return tn / (tn + fp)


def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def correlation_score(y_true, y_pred):
    result = spearmanr(y_true, y_pred)
    return result.correlation if not np.isnan(result.correlation) else 0.0


def icc_score(y_true, y_pred):
    """
    2. Compute ICC(2, 1)
     Type: 'ICC2' corresponds to the Two-way random effects model.
     Agreement: 'absolute' forces the calculation to use absolute agreement.
     We assume single raters (k=1, since we have one true measure and one predicted measure per subject)
    """

    # Assuming each row in y_true/y_pred corresponds to a unique subject (patient)
    num_subjects = len(y_true)
    subjects = np.arange(num_subjects)

    # Create the long-form DataFrame
    data = pd.DataFrame(
        {
            "Subject": np.concatenate([subjects, subjects]),
            "Rater": ["True_HVPG"] * num_subjects + ["Predicted_vHVPG"] * num_subjects,
            "Score": np.concatenate([y_true, y_pred]),
        }
    )

    icc_results = pg.intraclass_corr(data=data, targets="Subject", raters="Rater", ratings="Score")

    # 3. Extract the ICC(2, 1) value and its 95% CI
    icc_val = icc_results.loc[icc_results["Type"] == "ICC2", "ICC"].iloc[0]
    # icc_ci = icc_results.loc[icc_results['Type'] == 'ICC2', 'CI95%'].iloc[0]

    return icc_val


def compute_metric_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str, confidence_level: float = 0.95, axis: int = 1
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for various metrics using scipy's bootstrap.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or scores
        metric: Metric to compute the bootstrap on
        confidence_level: Confidence level for the interval

    Returns:
        Tuple containing (mean_metric, (lower_ci, upper_ci))
    """
    data = (y_true, y_pred)

    if metric == "auc":
        statistic = roc_auc_score
    elif metric == "balanced_accuracy":
        statistic = balanced_accuracy_score
    elif metric == "accuracy":
        statistic = accuracy_score
    elif metric == "sensitivity":
        statistic = sensitivity_score
    elif metric == "specificity":
        statistic = specificity_score
    elif metric == "mae":
        statistic = mae_score
    elif metric == "rmse":
        statistic = rmse_score
    elif metric == "correlation":
        statistic = correlation_score
    elif metric == "icc":
        statistic = icc_score
    else:
        raise ValueError(f"Metric {metric} not supported")
    # Calculate bootstrap confidence interval
    bootstrap_result = bootstrap(
        data=data,
        statistic=statistic,
        n_resamples=1000,
        confidence_level=confidence_level,
        method="percentile",
        paired=True,
        rng=np.random.default_rng(seed=seed),
    )
    # The sampling procedure within the bootstraping process may select a group of samples
    # that have all the same ground_truth value. It creates nan in the bootstrap_distribution
    # and confidence_interval. => We remove them and recompute the distribution and CIs.
    if np.isnan(bootstrap_result.bootstrap_distribution).sum() > 0:
        print(
            f"Warning: There are {np.isnan(bootstrap_result.bootstrap_distribution).sum()} nan",
            "in the bootstrap_distribution. Removing them and recomputing",
        )
        bootstrap_result.bootstrap_distribution = bootstrap_result.bootstrap_distribution[
            ~np.isnan(bootstrap_result.bootstrap_distribution)
        ]
        # recompute the confidence_interval as done in the boostrap function with the
        # default values : method="percentile", alternative="two-sided"
        alpha = (1 - confidence_level) / 2
        interval = alpha, 1 - alpha
        ci_l = np.percentile(a=bootstrap_result.bootstrap_distribution, q=interval[0] * 100, axis=-1)
        ci_u = np.percentile(a=bootstrap_result.bootstrap_distribution, q=interval[1] * 100, axis=-1)
        bootstrap_result.confidence_interval = (ci_l, ci_u)

    mean_metric = float(np.mean(bootstrap_result.bootstrap_distribution))
    ci_lower, ci_upper = bootstrap_result.confidence_interval
    assert not np.isnan(mean_metric), "Mean metric is nan"
    assert not np.isnan(ci_lower), "CI lower is nan"
    assert not np.isnan(ci_upper), "CI upper is nan"

    return mean_metric, (float(ci_lower), float(ci_upper))
