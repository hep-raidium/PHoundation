import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
FEATURES_NAMES_MAPPING = {
    "volume_pv": "PVV",
    "pv_volume_to_liver": "PVV/LV",
    "diameter_pv_to_volume_liver": "PVD/LV",
    "diameter_pv_to_volume_pv": "PVD/PVV",
    "diameter_pv": "PVD",
    "volume_spleen": "SV",
    "volume_liver": "LV",
    "lsvr": "LSVR",
    "couinaud_ratio": "L2R",
    "right_to_liver": "R/LV",
    "left_to_liver": "L/LV",
    "posterior_to_right": "P/R",
    "IV_to_left": "S4/L",
    "segment_1_to_liver": "S1/LV",
    "segment_2_to_liver": "S2/LV",
    "segment_3_to_liver": "S3/LV",
    "segment_4_to_liver": "S4/LV",
    "segment_5_to_liver": "S5/LV",
    "segment_6_to_liver": "S6/LV",
    "segment_7_to_liver": "S7/LV",
    "segment_8_to_liver": "S8/LV",
    "segment_1": "S1",
    "segment_2": "S2",
    "segment_3": "S3",
    "segment_4": "S4",
    "segment_5": "S5",
    "segment_6": "S6",
    "segment_7": "S7",
    "segment_8": "S8",
    "lsn": "LSN",
    "rarm": "Curia",
    "apri": "APRI",
    "radiomics": "Radiomics",
    "biomedclip": "BioMedCLIP",
    "medimageinsight": "MedimageInsight",
}


def shap_values_summary(model, X_train, X_test, features, output_folder, dataset):
    # Compute SHAP values
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.to_csv(str(output_folder / "shap_values.csv"))

    # Create and save SHAP summary plot
    plt.figure(figsize=(10, 6))
    features_names = [FEATURES_NAMES_MAPPING.get(feature, feature) for feature in features]
    shap.summary_plot(shap_values, X_test, feature_names=features_names, show=False, max_display=20)
    # shap.plots.waterfall(explainer(X_test)[0], max_display=len(features))
    plt.tight_layout()
    plt.savefig(output_folder / "figures" / f"shap_summary_{dataset}.png")
    plt.close()


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

    features_importances = []

    coefficients = best_estimator.coef_

    for feature_name, coef in zip(features, coefficients):
        features_importances.append({"feature_name": feature_name, "coef": coef})

    df_features_importances = pd.DataFrame(features_importances, columns=["feature_name", "coef"])
    df_features_importances.to_csv(str(output_folder / "features_importances.csv"))

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

        predicted_probas = None  # if classif : best_estimator.predict_proba(my_x)[:, 1]
        predicted_y = best_estimator.predict(my_x)

        # compute the metrics
        results[split] = compute_metrics(
            predicted_y,
            my_y,
            output_folder=output_folder,
            opt_threshold_dict=opt_threshold_dict,
        )

        # save the probas
        base_df = {
            "patient_uuid": df["patient_uuid"],
            "sample_uuid": df["sample_uuid"],
            "probas": predicted_probas,
            "pred": predicted_y.squeeze(),
            "y": my_y.squeeze(),
        }

        if "slice" in df.columns:
            base_df["slice"] = df["slice"]

        df = pd.DataFrame(base_df)
        df.to_csv(output_folder / f"probas_{split}.csv")

        plot_figures(my_y, predicted_y, df, output_folder, split)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_folder / "metrics.csv")

    shap_values_summary(best_estimator, X_train, X_test, features, output_folder, "internal_test")
    shap_values_summary(best_estimator, X_train, X_external, features, output_folder, "external_test")


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


def post_process_fm_results(output_folder: Path):

    # Until here, the optimal_threshold was computed at the slice level. Let's compute it at the patient level
    opt_threshold_dict_patient_level = {}
    df_val = pd.read_csv(output_folder / "probas_val_internal.csv")
    df_train = pd.read_csv(output_folder / "probas_train_internal.csv")
    df_train_val = pd.concat([df_train, df_val])
    df_train_val = df_train_val.drop(columns=["sample_uuid", "slice"])
    agg_method = "median"
    agg_dict = {
        column: "first" if column == "color" else agg_method
        for column in df_train_val.columns
        if column != "patient_uuid"
    }
    df_train_val = df_train_val.groupby("patient_uuid", as_index=False).agg(agg_dict)
    y_to_use = df_train_val["y"]
    predicted_y = df_train_val["pred"]
    for threshold in [10, 16]:
        csph_or_sph = (y_to_use >= threshold).astype(int)
        opt_threshold = find_optimal_threshold(csph_or_sph, predicted_y, threshold, "train_val_internal_pp", output_folder)
        opt_threshold_dict_patient_level[threshold] = opt_threshold
    # opt_threshold_dict_patient_level DONE

    post_processed_metrics = {}
    for file_name in ["train_internal", "val_internal", "test_internal", "test_external"]:
        file_path = output_folder / f"probas_{file_name}.csv"

        df = pd.read_csv(file_path)
        df = df.drop(columns=["sample_uuid", "slice"])  # drop the sample_uuid and slice columns

        agg_method = "median"
        agg_dict = {
            column: "first" if column == "color" else agg_method for column in df.columns if column != "patient_uuid"
        }
        df = df.groupby("patient_uuid", as_index=False).agg(agg_dict)
        # save the post processed dataframe
        df.to_csv(output_folder / f"probas_{file_name}_post_processed.csv")

        plot_figures(df["y"], df["pred"], df, output_folder, f"{file_name}_pp_{agg_method}")
        # let's compute the optimal thrshold with the post processed data

        post_processed_metrics[file_name] = compute_metrics(
            df["pred"].to_numpy(), df["y"].to_numpy(), output_folder, opt_threshold_dict_patient_level
        )

    results_df = pd.DataFrame(post_processed_metrics)
    save_filename = output_folder / f"metrics_post_processed_{agg_method}.csv"
    results_df.to_csv(save_filename)


def handle_missing_data(df_internal: pd.DataFrame, df_external: pd.DataFrame, features: List[str]):
    """
    internal dataset misses
    - 2 patients with splenectomy => cannot be used if volume slpeen and lsnvr are considered
    - 6 patients with apri or fib missing => cannot be used if apri or fib are considered
             (Dropping 5 patients and filling the test patient with the mean of the test set)
    - 7 patients with bmi missing => cannot be used if bmi is considered
    external dataset misses
    - 6 patients with bmi missing => cannot be used if bmi is considered
    """
    initial_df_internal = df_internal.copy()
    initial_df_external = df_external.copy()

    if "apri" in features or "fib" in features:
        # Fill nan values with the mean of the apri and fib columns for the test sample only
        # Drop the train and val samples that has na values for apri and fib
        df_internal.loc[df_internal["patient_uuid"] == "fc32ba0ea8", "apri"] = df_internal[df_internal["split"].isin(["train", "val"])][
            "apri"
        ].mean()
        df_internal.loc[df_internal["patient_uuid"] == "fc32ba0ea8", "fib"] = df_internal[df_internal["split"].isin(["train", "val"])][
            "fib"
        ].mean()
        df_internal = df_internal[df_internal["apri"].notna() & df_internal["fib"].notna()]  # dropping the rest of the patients

    for serum_feature in ["gamma_gt_n", "bilirubine", "platelets", "INR"]:
        if serum_feature in features:
            # Fill nan values with the mean of the serum feature columns for the test sample only
            # Drop the train and val samples that has na values for serum feature
            # the mean value must be computed on the train and val samples only
            mean_value = df_internal[df_internal["split"].isin(["train", "val"])][serum_feature].mean()
            df_internal.loc[(df_internal["split"] == "test") & (df_internal[serum_feature].isna()), serum_feature] = mean_value
            df_internal = df_internal[df_internal[serum_feature].notna()]  # dropping the rest of the patients
            df_external.loc[df_external[serum_feature].isna(), serum_feature] = mean_value

    if "bmi" in features:
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


def main(df: pd.DataFrame, df_external: pd.DataFrame, output_folder: Path, method: str, penalty: str):
    output_folder.mkdir(parents=True, exist_ok=True)

    if method in ["morphology", "composite"]:

        # Load combinations from JSON file
        with open(f"src/raidium/rd/{method.upper()}/biomarkers_composition_combinations.json", "r") as f:
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
        post_process_fm_results(output_folder)

    elif method == "radiomics":
        features = [col for col in df.columns if col not in COLUMNS_TO_DROP + ["study_type"]]

        train_models(df, df_external, output_folder, features, penalty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_features_internal", type=Path, required=True)
    parser.add_argument("--df_features_external", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, default="results")
    parser.add_argument("--method", type=str, choices=["morphology", "radiomics", "fm"], required=True)
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        required=True,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.df_features_internal)
    df_external = pd.read_csv(args.df_features_external)
    print("Training models for the following method: ", args.method, "with the following penalty: ", args.penalty)
    main(df, df_external, args.output_folder, args.method, args.penalty)
