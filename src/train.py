import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils.metrics import (  # icc_score,
    compute_metric_bootstrap,
    find_optimal_threshold,
    sensitivity_score,
    specificity_score,
)
from utils.vizualization import plot_confusion_matrix, plot_figures
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

seed = 42  # bootsrapping has no random state => seed set here
random.seed(seed)
np.random.seed(seed)

COLUMNS_TO_DROP = ["hvpg", "csph", "split", "sample_uuid", "patient_uuid"]


def compute_single_metric_bootstrap(
    y_true_binary: np.ndarray, y_pred_binary: np.ndarray, metric: str
) -> Tuple[str, float, float, float]:
    bootstrap_metric, (metric_ci_lower, metric_ci_upper) = compute_metric_bootstrap(
        y_true_binary, y_pred_binary, metric
    )
    if metric in ["rmse", "mae", "correlation"]:
        return metric, bootstrap_metric, metric_ci_lower, metric_ci_upper
    elif metric in ["balanced_accuracy", "accuracy", "sensitivity", "specificity"]:
        return metric, bootstrap_metric * 100, metric_ci_lower * 100, metric_ci_upper * 100
    else:
        raise ValueError(f"Metric {metric} not supported")


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    output_folder: Path,
    opt_threshold_dict: dict,
):
    """
    Compute evaluation metrics for a given model.

    Args:
        y_pred_proba: Predicted probabilities
        y_pred: Predicted labels
        y: Target values
        opt_threshold_dict: Dictionary containing the optimal thresholds for each real threshold

    Returns:
        dict: Dictionary containing computed metrics
    """
    all_metrics = {}

    # Computing metrics in original space
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    corr = spearmanr(y_true, y_pred).correlation
    corr_p_value = spearmanr(y_true, y_pred).pvalue
    # icc = icc_score(y_true, y_pred)  # icc make the whole pipeline break

    # Bootstrap for regression metrics
    results = Parallel(n_jobs=-1)(
        delayed(compute_single_metric_bootstrap)(y_true, y_pred, metric)
        for metric in ["rmse", "mae", "correlation"]  # icc is skipped
    )
    for metric, bootstrap_metric, metric_ci_lower, metric_ci_upper in results:
        all_metrics[f"bootstrap_{metric}"] = bootstrap_metric
        all_metrics[f"bootstrap_{metric}_ci_lower"] = metric_ci_lower
        all_metrics[f"bootstrap_{metric}_ci_upper"] = metric_ci_upper

    # bootstrap_icc, (icc_ci_lower, icc_ci_upper) = compute_metric_bootstrap(y_true, y_pred, "icc")

    all_metrics["rmse"] = rmse
    all_metrics["mae"] = mae
    all_metrics["correlation"] = corr
    all_metrics["correlation_p_value"] = corr_p_value

    # all_metrics["icc"] = icc
    # all_metrics["bootstrap_icc"] = bootstrap_icc
    # all_metrics["bootstrap_icc_ci_lower"] = icc_ci_lower
    # all_metrics["bootstrap_icc_ci_upper"] = icc_ci_upper

    for threshold in [10, 16]:

        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= opt_threshold_dict[threshold]).astype(int)

        # plot the confusion matrix
        plot_confusion_matrix(y_true_binary, y_pred_binary, output_folder, threshold)

        auc = roc_auc_score(y_true_binary, y_pred)
        bootstrap_auc, (auc_ci_lower, auc_ci_upper) = compute_metric_bootstrap(y_true_binary, y_pred, "auc")

        results = Parallel(n_jobs=-1)(
            delayed(compute_single_metric_bootstrap)(y_true_binary, y_pred_binary, metric)
            for metric in ["balanced_accuracy", "accuracy", "sensitivity", "specificity"]
        )

        for metric, bootstrap_metric, metric_ci_lower, metric_ci_upper in results:
            all_metrics[f"bootstrap_{metric}_{threshold}"] = bootstrap_metric
            all_metrics[f"bootstrap_{metric}_{threshold}_ci_lower"] = metric_ci_lower
            all_metrics[f"bootstrap_{metric}_{threshold}_ci_upper"] = metric_ci_upper

        balanced_accuracy = balanced_accuracy_score(y_true_binary, y_pred_binary)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        sensitivity = sensitivity_score(y_true_binary, y_pred_binary)
        specificity = specificity_score(y_true_binary, y_pred_binary)

        f1 = f1_score(y_true_binary, y_pred_binary)

        all_metrics[f"optimal_threshold_for_{threshold}"] = opt_threshold_dict[threshold]
        all_metrics[f"auc_{threshold}"] = auc * 100
        all_metrics[f"bootstrap_auc_{threshold}"] = bootstrap_auc * 100
        all_metrics[f"bootstrap_auc_{threshold}_ci_lower"] = auc_ci_lower * 100
        all_metrics[f"bootstrap_auc_{threshold}_ci_upper"] = auc_ci_upper * 100
        all_metrics[f"accuracy_{threshold}"] = accuracy * 100
        all_metrics[f"balanced_accuracy_{threshold}"] = balanced_accuracy * 100
        all_metrics[f"sensitivity_{threshold}"] = sensitivity * 100
        all_metrics[f"specificity_{threshold}"] = specificity * 100
        all_metrics[f"f1_{threshold}"] = f1 * 100
    return all_metrics


def train_models(
    df_internal: pd.DataFrame,
    df_external: pd.DataFrame,
    output_folder: Path,
    features: List[str],
    penalty: Optional[str],
):
    """
    Train models using cross-validation to predict HVPG/CSPH.
    Test the different models on the test set and on the external dataset.
    """
    (output_folder / "figures").mkdir(parents=True, exist_ok=True)

    df_train = df_internal[df_internal["split"] == "train"]
    df_val = df_internal[df_internal["split"] == "val"]
    df_train_val = df_internal[df_internal["split"].isin(["train", "val"])]
    df_test = df_internal[df_internal["split"] == "test"]

    groups = df_train_val["patient_uuid"]

    y_train = np.array(df_train["hvpg"])
    y_val = np.array(df_val["hvpg"])
    y_train_val = np.array(df_train_val["hvpg"])
    y_test = np.array(df_test["hvpg"])

    # external dataset
    y_external = df_external["hvpg"]
    y_external = np.array(y_external)

    scaler = StandardScaler()
    scaler.fit(df_train[features])
    X_train_val = scaler.transform(df_train_val[features])
    X_train = scaler.transform(df_train[features])
    X_val = scaler.transform(df_val[features])
    X_test = scaler.transform(df_test[features])
    X_external = scaler.transform(df_external[features])

    # let's dump the scaled features
    for X_to_dump, name, original_df in [
        (X_train_val, "X_train_val", df_train_val),
        (X_train, "X_train", df_train),
        (df_val, "X_val", df_val),
        (df_test, "X_test", df_test),
        (df_external, "X_external", df_external),
    ]:
        df_to_dump = pd.DataFrame(X_to_dump, columns=features)
        df_to_dump["patient_uuid"] = original_df["patient_uuid"]
        save_path = output_folder / "scaled_features" / f"{name}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_dump.to_csv(save_path)

    regularization_param_list = [0.05, 0.1, 0.5, 1]

    # TODO : Bin the target variable to be able to stratify ?
    cv_splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)

    if penalty == "l2":
        model = Ridge(random_state=seed, solver="saga", max_iter=1000)
        param_grid = {"alpha": regularization_param_list}
        # alpha = 0 => ordinary least squares

    elif penalty == "l1":
        model = Lasso(random_state=seed, max_iter=1000)
        param_grid = {"alpha": regularization_param_list}
        # alpha = 0 => ordinary least squares

    scoring = "neg_mean_squared_error"

    grid_search = GridSearchCV(model, param_grid, cv=cv_splitter, scoring=scoring, return_train_score=True, refit=True)
    grid_search.fit(X_train_val, y_train_val, groups=groups)
    # let's dumps the patient id contained in each split
    split_to_patient_uuids = {}
    with open(str(output_folder / "splitted_patient_uuids.json"), "w") as f:
        for idx, (train_index, val_index) in enumerate(cv_splitter.split(X_train_val, y_train_val, groups)):
            train_patient_uuids = df_train_val["patient_uuid"].iloc[train_index].tolist()
            val_patient_uuids = df_train_val["patient_uuid"].iloc[val_index].tolist()
            split_to_patient_uuids[idx] = {
                "train_patient_uuids": sorted(train_patient_uuids),
                "val_patient_uuids": sorted(val_patient_uuids),
            }
            assert len(set(train_patient_uuids).intersection(set(val_patient_uuids))) == 0
        json.dump(split_to_patient_uuids, f, indent=4)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv(str(output_folder / "cv_results.csv"))

    best_estimator = grid_search.best_estimator_

    # saving the best model and its parameters
    with open(str(output_folder / "hyperparameters.json"), "w") as f:
        parameter_gs = {
            "best_params": grid_search.best_params_,
            "param_grid": param_grid,
            "best_estimator": str(best_estimator),
            "scoring": scoring,
            "scaler": str(scaler),
            "splitter": str(cv_splitter),
        }
        json.dump(parameter_gs, f, indent=4)

    results = {}

    X_to_use = X_train_val
    y_to_use = y_train_val
    split = "train_val_internal"
    predicted_y = best_estimator.predict(X_to_use)
    opt_threshold_dict = {}
    for threshold in [10, 16]:
        csph_or_sph = (y_to_use >= threshold).astype(int)
        opt_threshold = find_optimal_threshold(csph_or_sph, predicted_y, threshold, split, output_folder)
        opt_threshold_dict[threshold] = opt_threshold

    for df, my_x, my_y, split in [
        (df_train, X_train, y_train, "train_internal"),
        (df_val, X_val, y_val, "val_internal"),
        (df_test, X_test, y_test, "test_internal"),
        (df_external, X_external, y_external, "test_external"),
    ]:

        if (
            (split == "test_external" and "apri" in features)
            or (split == "test_external" and "fib" in features)
            or (split == "test_external" and "gamma_gt_n" in features)
        ):
            # in the case of FIB-4 or APRI, we don't have a test set for the external dataset
            results[split] = None
            continue

        predicted_y = best_estimator.predict(my_x)

        # compute the metrics
        results[split] = compute_metrics(
            predicted_y,
            my_y,
            output_folder=output_folder,
            opt_threshold_dict=opt_threshold_dict,
        )

        # save the predictions
        base_df = {
            "patient_uuid": df["patient_uuid"],
            "sample_uuid": df["sample_uuid"],
            "pred": predicted_y.squeeze(),
            "y": my_y.squeeze(),
        }

        if "slice" in df.columns:
            base_df["slice"] = df["slice"]

        df = pd.DataFrame(base_df)
        df.to_csv(output_folder / f"predictions_{split}.csv")

        plot_figures(my_y, predicted_y, df, output_folder, split)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_folder / "metrics.csv")


def compile_results(output_folder: Path, list_combination_names: List[str]):
    results = {}
    for combination_name in list_combination_names:
        file = output_folder / combination_name / "metrics.csv"
        df = pd.read_csv(file, index_col=0)  # Set first column as index
        results[combination_name] = {
            "train_internal_mae": df.loc["mae", "train_internal"],
            "val_internal_mae": df.loc["mae", "val_internal"],
            "test_internal_mae": df.loc["mae", "test_internal"],
            "test_external_mae": df.loc["mae", "test_external"],
            "train_internal_auc_10": df.loc["auc_10", "train_internal"],
            "val_internal_auc_10": df.loc["auc_10", "val_internal"],
            "test_internal_auc_10": df.loc["auc_10", "test_internal"],
            "test_external_auc_10": df.loc["auc_10", "test_external"],
            "train_internal_auc_16": df.loc["auc_16", "train_internal"],
            "val_internal_auc_16": df.loc["auc_16", "val_internal"],
            "test_internal_auc_16": df.loc["auc_16", "test_internal"],
            "test_external_auc_16": df.loc["auc_16", "test_external"],
        }

    results = pd.DataFrame(results).transpose()
    results.to_csv(output_folder / "compiled_results.csv")


def handle_missing_data(df_internal: pd.DataFrame, df_external: pd.DataFrame, features: List[str]):
    """
    Handling of missing data in the internal and external datasets. 
    If it is in train : dropping, 
    If it is in test : filling with the mean of the train and val samples.

    Internal dataset misses
    - 2 patients with splenectomy 
    - 7 patients with bmi missing 
    
    External dataset misses
    - 6 patients with bmi missing
    """
    initial_df_internal = df_internal.copy()
    initial_df_external = df_external.copy()

    for serum_feature in ["gamma_gt_n", "bilirubine", "platelets", "INR"]:
        if serum_feature in features:
            # A column with no missing value is read as an integer dtype, which cannot hold the
            # mean we impute below, so make the dtype explicit before filling.
            df_internal[serum_feature] = df_internal[serum_feature].astype(float)
            df_external[serum_feature] = df_external[serum_feature].astype(float)

            # Fill nan values with the mean of the serum feature columns for the test sample only
            # Drop the train and val samples that has na values for serum feature
            # the mean value must be computed on the train and val samples only
            mean_value = df_internal[df_internal["split"].isin(["train", "val"])][serum_feature].mean()
            df_internal.loc[(df_internal["split"] == "test") & (df_internal[serum_feature].isna()), serum_feature] = mean_value
            df_internal = df_internal[df_internal[serum_feature].notna()]  # dropping the rest of the patients
            df_external.loc[df_external[serum_feature].isna(), serum_feature] = mean_value

    if "bmi" in features:
        df_internal["bmi"] = df_internal["bmi"].astype(float)
        df_external["bmi"] = df_external["bmi"].astype(float)

        # For internal dataset:
        # Fill nan values with the mean of the bmi column for the test sample only
        # Drop the train and val samples that has na values for bmi
        mean_value = df_internal[df_internal["split"].isin(["train", "val"])]["bmi"].mean()
        df_internal.loc[df_internal["patient_uuid"] == "fc32ba0ea8", "bmi"] = mean_value
        df_internal = df_internal[df_internal["bmi"].notna()]  # dropping the rest of the patients
        df_external.loc[df_external["bmi"].isna(), "bmi"] = mean_value

    if "volume_spleen" in features or "lsvr" in features:
        # we drop the 2 patients that have splenectomy
        df_internal = df_internal[df_internal["patient_uuid"] != "2fa475c415"]
        df_internal = df_internal[df_internal["patient_uuid"] != "2caf6bb0b0"]

    print(f"Dropping {len(initial_df_internal) - len(df_internal)} rows with NaN values in internal dataset")
    print(f"Dropping {len(initial_df_external) - len(df_external)} rows with NaN values in external dataset")
    return df_internal, df_external


def _train_single_combination(
    combination: dict,
    df: pd.DataFrame,
    df_external: pd.DataFrame,
    output_folder: Path,
    penalty: str,
) -> str:
    """Train models for a single feature combination. Returns the combination name."""
    combination_name = combination["name"]
    features_names = combination["features"]
    df_internal, df_external = handle_missing_data(df, df_external, features_names)

    train_models(
        df_internal,
        df_external,
        output_folder / combination_name,
        features_names,
        penalty,
    )
    return combination_name


def main(
    df: pd.DataFrame,
    df_external: pd.DataFrame,
    output_folder: Path,
    method: str,
    penalty: str,
    biomarker_combinations: Path,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    if method == "biomarkers":

        # Load combinations from JSON file
        with open(biomarker_combinations, "r") as f:
            combinations_data = json.load(f)
            list_feature_combinations = combinations_data["combinations"]

        with open(output_folder / "biomarkers_composition_combinations.json", "w") as f:
            json.dump(combinations_data, f, indent=4)

        list_combination_names = Parallel(n_jobs=20)(
            delayed(_train_single_combination)(
                combination,
                df,
                df_external,
                output_folder,
                penalty,
            )
            for combination in tqdm(list_feature_combinations, total=len(list_feature_combinations))
        )

        compile_results(output_folder, list_combination_names)

    elif method == "fm":
        features = [col for col in df.columns if col not in COLUMNS_TO_DROP + ["uuid", "slice"]]

        train_models(df, df_external, output_folder, features, penalty)

    elif method == "radiomics":
        features = [col for col in df.columns if col not in COLUMNS_TO_DROP + ["study_type"]]

        train_models(df, df_external, output_folder, features, penalty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_features_internal", type=Path, required=True)
    parser.add_argument("--df_features_external", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, default="results")
    parser.add_argument("--method", type=str, choices=["biomarkers", "radiomics", "fm"], required=True)
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        required=True,
    )
    # Only used by --method biomarkers: the feature combinations to train, one model per combination.
    parser.add_argument(
        "--biomarker_combinations",
        type=Path,
        default=Path(__file__).parent / "biomarkers_composition_combinations.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.df_features_internal)
    df_external = pd.read_csv(args.df_features_external)
    print("Training models for the following method: ", args.method, "with the following penalty: ", args.penalty)
    main(df, df_external, args.output_folder, args.method, args.penalty, args.biomarker_combinations)
