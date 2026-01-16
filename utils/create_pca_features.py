import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def main(args):
    for n_components in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

        (args.output_path).mkdir(parents=True, exist_ok=True)

        features = pd.read_csv(args.input_path_internal)
        features_train = features[features["split"] == "train"]
        features_external = pd.read_csv(args.input_path_external)
        pca = PCA(n_components=n_components)
        # making sure the columns hvpg and csph are dropped
        if "hvpg" not in args.columns_to_drop:
            args.columns_to_drop.append("hvpg")
        if "csph" not in args.columns_to_drop:
            args.columns_to_drop.append("csph")

        pca.fit(features_train.drop(columns=args.columns_to_drop))

        # 10 columns that contribute the most to the PC1 (in absolute value)
        input_columns = features.drop(columns=args.columns_to_drop).columns
        loadings = pd.DataFrame(
            pca.components_.T,         
            columns=[f'PC{i+1}' for i in range(n_components)], 
            index=input_columns
        )
        top_features_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
        print("10 columns that contribute the most to the PC1 (in absolute value)")
        print(top_features_pc1)

        pca_features = pca.transform(features.drop(columns=args.columns_to_drop))
        pca_features_external = pca.transform(features_external.drop(columns=args.columns_to_drop))

        # plot the points
        plt.figure(figsize=(10, 5))
        plt.scatter(pca_features[:, 0], pca_features[:, 1], c=features["csph"], alpha=0.5, cmap="coolwarm", label="internal")
        plt.colorbar(label="internal")
        plt.scatter(
            pca_features_external[:, 0],
            pca_features_external[:, 1],
            c=features_external["csph"],
            alpha=0.5,
            cmap="summer",
            label="external",
        )
        plt.colorbar(label="external")

        plt.xlabel("PCA component 1")
        plt.ylabel("PCA component 2")
        plt.title("PCA of CSPH")
        plt.savefig(args.output_path / f"pca_{n_components}.png")

        pca_features = pd.DataFrame(pca_features)
        pca_features_external = pd.DataFrame(pca_features_external)
        for col in args.columns_to_drop:
            pca_features[col] = features[col]
            pca_features_external[col] = features_external[col]

        # save the df as csv
        pca_features.to_csv(args.output_path / f"internal_{n_components}.csv", index=False)
        pca_features_external.to_csv(args.output_path / f"external_{n_components}.csv", index=False)

        # plot the PCA
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel("Number of components")
        plt.ylabel("Explained variance ratio")
        plt.title("PCA explained variance ratio")
        plt.savefig(args.output_path / f"explained_variance_{n_components}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_internal", type=Path)
    parser.add_argument("--input_path_external", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--columns_to_drop", nargs="+")
    # enter default=["sample_uuid", "uuid", "split", "patient_uuid", "hvpg", "csph", "slice"] for FM
    # enter ["sample_uuid", "split", "patient_uuid", "hvpg", "csph", "study_type"] for radiomics
    args = parser.parse_args()
    main(args)
