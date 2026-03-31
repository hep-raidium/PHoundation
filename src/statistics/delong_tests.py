import argparse
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from MLstatkit.stats import Delong_test


RESULTS_PATHS = {
    "curia": {
        "internal": Path("foundation_model_curia/probas_test_internal.csv"),
        "external": Path("foundation_model_curia/probas_test_external.csv"),
    },
    "medimage": {
        "internal": Path("foundation_model_medimageinsight/probas_test_internal_post_processed.csv"),
        "external": Path("foundation_model_medimageinsight/probas_test_external_post_processed.csv"),
    },
    "biomed": {
        "internal": Path("foundation_model_biomedclip/probas_test_internal_post_processed.csv"),
        "external": Path("foundation_model_biomedclip/probas_test_external_post_processed.csv"),
    },
    "morpho": {
        "internal": Path("morphology/probas_test_internal.csv"),
        "external": Path("morphology/probas_test_external.csv"),
    },
    "radiomics": {
        "internal": Path("radiomics/probas_test_internal.csv"),
        "external": Path("radiomics/probas_test_external.csv"),
    },
    "lsn": {
        "internal": Path("radiologist_assisted/probas_test_internal.csv"),
        "external": Path("radiologist_assisted/probas_test_external.csv"),
    },
    "serum_clinical": {
        "internal": Path("serum/probas_test_internal.csv"),
        "external": Path("serum/probas_test_external.csv"),
    },
}


def prepare_df(model_name: str, base_path: Path, threshold: float):
    df_internal = pd.read_csv(base_path / RESULTS_PATHS[model_name]["internal"], index_col=0)
    df_external = pd.read_csv(base_path / RESULTS_PATHS[model_name]["external"], index_col=0)

    for df in [df_internal, df_external]:
        df["csph"] = df["y"] >= threshold

    # let's order the df according to the patient_uuid
    df_internal = df_internal.sort_values(by="patient_uuid")  # Make sure to keep this line !!
    df_external = df_external.sort_values(by="patient_uuid")  # Make sure to keep this line !!

    # use patient_uuid as index
    if df_internal.index.name != "patient_uuid":
        df_internal = df_internal.set_index("patient_uuid")
    if df_external.index.name != "patient_uuid":
        df_external = df_external.set_index("patient_uuid")
    return df_internal, df_external


def main(base_path: Path, output_dir: Path, significance_level: float, threshold: float):

    curia_internal, curia_external = prepare_df("curia", base_path, threshold)
    medimage_internal, medimage_external = prepare_df("medimage", base_path, threshold)
    biomed_internal, biomed_external = prepare_df("biomed", base_path, threshold)
    morpho_internal, morpho_external = prepare_df("morpho", base_path, threshold)
    radiomics_internal, radiomics_external = prepare_df("radiomics", base_path, threshold)
    lsn_internal, lsn_external = prepare_df("lsn", base_path, threshold)
    serum_clinical_internal, serum_clinical_external = prepare_df("serum_clinical", base_path, threshold)

    # Pairwise comparison and plot the results
    list_of_combinations_internal = list(
        permutations(
            [
                ("curia", curia_internal),
                ("biomed", biomed_internal),
                ("medimage", medimage_internal),
                ("morpho", morpho_internal),
                ("radiomics", radiomics_internal),
                ("lsn", lsn_internal),
                ("serum_clinical", serum_clinical_internal),
            ],
            2,
        )
    )

    list_of_combinations_external = list(
        permutations(
            [
                ("curia", curia_external),
                ("biomed", biomed_external),
                ("medimage", medimage_external),
                ("morpho", morpho_external),
                ("radiomics", radiomics_external),
                ("lsn", lsn_external),
                ("serum_clinical", serum_clinical_external),
            ],
            2,
        )
    )

    for list_of_combinations, test_dataset in [(list_of_combinations_internal, "internal"), (list_of_combinations_external, "external")]:

        delong_test_results = pd.DataFrame(columns=["model1", "model2", "z_score", "p_value", "significant"])
        for combination in list_of_combinations:
            (model1_name, model1_df), (model2_name, model2_df) = combination

            assert model1_df.index.all() == model2_df.index.all()

            z_score, p_value = Delong_test(
                list(model1_df["csph"].astype(int)),
                list(model1_df["pred"]),
                list(model2_df["pred"]),
                return_ci=False,
                return_auc=False,
            )

            delong_test_results = pd.concat(
                [
                    delong_test_results,
                    pd.DataFrame(
                        {
                            "model1": [model1_name],
                            "model2": [model2_name],
                            "z_score": [z_score],
                            "p_value": [p_value],
                            "significant": [p_value < significance_level],
                        },
                    ),
                ]
            )

        # save the results
        print(
            "Delong test results are saved in ",
            output_dir / f"delong_test_results_{test_dataset}_{significance_level}.csv",
        )
        delong_test_results.to_csv(
            output_dir / f"delong_test_results_{test_dataset}_{significance_level}.csv", index=False
        )
        # create a double entry table for the results, and plot a heatmap of the results
        delong_test_results_pivot = delong_test_results.pivot(index="model1", columns="model2", values="significant")
        delong_test_results_pivot = delong_test_results_pivot.astype(bool)
        plt.figure(figsize=(10, 10))
        plt.title(f"Delong test results for {test_dataset} dataset and {significance_level} significance level")
        sns.heatmap(delong_test_results_pivot, annot=True, cmap="coolwarm")
        plt.savefig(output_dir / f"delong_test_results_heatmap_{test_dataset}_{significance_level}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default="src/statistics/regression")
    parser.add_argument("--output_dir", type=Path, default="Delong_test_results")
    parser.add_argument("--significance_level", type=float, default=0.05)
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()
    args.output_dir = args.output_dir / str(args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args.base_path, args.output_dir, args.significance_level, args.threshold)
