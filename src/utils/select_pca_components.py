"""Automatically select the number of PCA components using the BIC criterion.

scikit-learn's ``PCA`` is a Gaussian Probabilistic PCA model, so ``pca.score(X)`` gives the
average per-sample log-likelihood under that model. For each candidate dimensionality ``k`` we
compute the total log-likelihood on the (train-only, standardized) features, count the model's
free parameters, and derive the BIC. The selected number of components is the ``argmin`` of the
BIC.

Features are every column except those in ``--columns_to_drop`` (the id / split / metadata columns);
the endpoints ``hvpg`` / ``csph`` are always dropped as well, so they are never fed to PCA.
Everything is fit on the train split only.
"""

import argparse
import math
from pathlib import Path

import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT_PATH = Path("results/pca-component-selection")

# Colour of the BIC-selected marker line, shared across the BIC curve and the scree plot so a
# reader immediately connects the two.
BIC_MARKER_COLOR = "C2"


def ppca_n_params(k: int, n_features: int) -> int:
    """Number of free parameters of a Probabilistic PCA model with ``k`` latent dimensions.

    Following the standard PPCA parameterization (Minka, 2000): the loading matrix up to
    rotation (``D*k - k*(k-1)/2``), the isotropic noise variance (``+1``) and the mean (``+D``).
    """
    return n_features * k - k * (k - 1) // 2 + 1 + n_features


def plot_scree(explained_variance_ratio, k_selected, criterion_name, output_path):
    """Bar scree plot of the per-component explained variance ratio.

    Individual variance is drawn as bars, the cumulative curve on a second axis, and the
    number of components selected by ``criterion_name`` is marked with a vertical line.
    """
    components = range(1, len(explained_variance_ratio) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(components, explained_variance_ratio, color="C0", label="Individual")
    ax.axvline(
        k_selected, color=BIC_MARKER_COLOR, linestyle="--", alpha=0.8, label=f"{criterion_name} (k={k_selected})"
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PCA scree plot ({criterion_name}-selected components)")

    ax_cum = ax.twinx()
    ax_cum.plot(components, explained_variance_ratio.cumsum(), color="C1", marker="o", markersize=3, label="Cumulative")
    ax_cum.set_ylabel("Cumulative explained variance ratio")

    lines, labels = ax.get_legend_handles_labels()
    lines_cum, labels_cum = ax_cum.get_legend_handles_labels()
    ax.legend(lines + lines_cum, labels + labels_cum, loc="center right")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_reduced(source_df, feature_cols, scaler, pca, output_path):
    """Reduce ``source_df`` with the (train-fitted) ``scaler`` + ``pca`` and save it as a CSV.

    The reduced components are named ``"0".."k-1"`` and the metadata / endpoint columns (everything
    that is not a feature) are re-attached, so the file is directly consumable by ``train.py``
    (``--method fm`` or ``--method radiomics``).
    """
    reduced = pca.transform(scaler.transform(source_df[feature_cols].to_numpy()))
    reduced_df = pd.DataFrame(reduced, columns=[str(i) for i in range(reduced.shape[1])])
    metadata_cols = [c for c in source_df.columns if c not in feature_cols]
    reduced_df = pd.concat([reduced_df, source_df[metadata_cols].reset_index(drop=True)], axis=1)
    reduced_df.to_csv(output_path, index=False)
    print(f"Saved reduced features ({reduced.shape[1]} components, {len(reduced_df)} samples) to {output_path}")


def main(args):
    args.output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_path)

    # Features are every column except the metadata columns; the endpoints (hvpg, csph) are always
    # dropped too, so they can never leak into PCA regardless of the passed --columns_to_drop.
    columns_to_drop = list(args.columns_to_drop)
    for endpoint in ["hvpg", "csph"]:
        if endpoint not in columns_to_drop:
            columns_to_drop.append(endpoint)
    feature_cols = [c for c in df.columns if c not in columns_to_drop]
    assert "hvpg" not in feature_cols and "csph" not in feature_cols, "endpoint leaked into features"
    assert feature_cols, "no feature columns found after dropping metadata"

    # Fit everything on the train split only.
    df_train = df[df["split"] == "train"]
    n_train = len(df_train)
    n_features = len(feature_cols)
    print(f"{n_train} train samples, {n_features} feature columns")

    # Standardize using train statistics only.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[feature_cols].to_numpy())

    # Cap k below the p >> n degeneracy: after mean-centering the rank is <= n-1, so the
    # isotropic noise variance estimated from the discarded eigenvalues collapses to ~0 near
    # the top of the range.
    k_max = min(n_train - 2, n_features - 1)

    records = []
    for k in tqdm(range(1, k_max + 1)):
        pca = PCA(n_components=k)
        pca.fit(x_train)
        if pca.noise_variance_ <= 1e-12:
            print(f"stopping at k={k}: noise variance collapsed ({pca.noise_variance_:.2e})")
            break
        log_likelihood = pca.score(x_train) * n_train
        n_params = ppca_n_params(k, n_features)
        bic = -2 * log_likelihood + n_params * math.log(n_train)
        records.append(
            {
                "k": k,
                "log_likelihood": log_likelihood,
                "n_params": n_params,
                "bic": bic,
                "cumulative_explained_variance": pca.explained_variance_ratio_.sum(),
            }
        )

    results = pd.DataFrame(records)
    results.to_csv(args.output_path / "pca_bic.csv", index=False)

    k_bic = int(results.loc[results["bic"].idxmin(), "k"])
    k_last = int(results["k"].iloc[-1])

    # Independent cross-check: scikit-learn's Minka MLE for the dimensionality. It is only
    # defined when n_samples >= n_features, so it is unavailable in this p >> n setting.
    if n_train >= n_features:
        k_mle = str(PCA(n_components="mle").fit(x_train).n_components_)
    else:
        k_mle = "n/a (requires n_samples >= n_features)"

    def annotate(k_sel: int) -> str:
        # A minimum at the top of the feasible range is not a genuine interior optimum:
        # the criterion is still decreasing and would keep going if more components were allowed.
        return " (grid boundary: no interior minimum, criterion still decreasing)" if k_sel == k_last else ""

    print(f"Selected components (BIC) = {k_bic}{annotate(k_bic)}")
    print(f"Reference (Minka MLE)     = {k_mle}")
    print(
        "With D >> n, BIC penalizes parameters heavily and favors a small k; inspect "
        "pca_bic.csv and pca_bic.png in the output folder to justify the choice."
    )

    # Plot the BIC curve with a marker at its minimum.
    plt.figure(figsize=(10, 5))
    plt.plot(results["k"], results["bic"], label="BIC")
    plt.axvline(k_bic, color=BIC_MARKER_COLOR, linestyle="--", alpha=0.6, label=f"BIC min (k={k_bic})")
    plt.xlabel("Number of PCA components")
    plt.ylabel("BIC (lower is better)")
    plt.title("PCA component selection via BIC")
    plt.legend()
    plt.savefig(args.output_path / "pca_bic.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Scree plot (bar) from a single full-spectrum fit, independent of the BIC grid step,
    # marking the BIC-selected number of components.
    full_pca = PCA(n_components=k_max).fit(x_train)
    explained_variance_ratio = full_pca.explained_variance_ratio_
    plot_scree(explained_variance_ratio, k_bic, "BIC", args.output_path / "pca_scree_plot_bic.png")

    # Save the PCA-reduced features at the BIC-selected number of components. Scaler and PCA are
    # fit on the train split only (no leakage); the internal and (optionally) external frames are
    # reduced with the SAME train-fitted transforms, ready to be fed to train.py.
    pca_bic = PCA(n_components=k_bic).fit(x_train)
    save_reduced(df, feature_cols, scaler, pca_bic, args.output_path / f"internal_bic_{k_bic}.csv")

    if args.input_path_external is not None:
        df_external = pd.read_csv(args.input_path_external)
        missing = [c for c in feature_cols if c not in df_external.columns]
        assert not missing, f"external features missing {len(missing)} feature columns, e.g. {missing[:5]}"
        save_reduced(df_external, feature_cols, scaler, pca_bic, args.output_path / f"external_bic_{k_bic}.csv")

    # Record the chosen count so a caller (e.g. the launcher) can reference the exact filenames
    # without parsing stdout.
    (args.output_path / "bic_n_components.txt").write_text(f"{k_bic}\n")
    print(f"BIC-selected number of components: {k_bic}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--input_path_external", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--columns_to_drop", nargs="+", default=["sample_uuid", "split", "patient_uuid", "hvpg", "csph"]
    )
    args = parser.parse_args()
    main(args)
