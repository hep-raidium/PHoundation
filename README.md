# PHoundation

This is the repository associated with the paper "Estimating clinically significant portal hypertension: foundation models versus automated non-invasive tests". The paper benchmarks several non-invasive tests, including foundation models, to estimate the severity of portal hypertension (PH).
This repository enables training a cross-validated regression to predict Hepatic Venous Pressure Gradient (HVPG) using a set of numerical features.

# How to install?

Run `uv sync` in your terminal to create the environment and install the dependencies.
If you do not have `uv`, install it with `pipx install uv` (or follow https://docs.astral.sh/uv/getting-started/installation/).
Make sure you have Python >= 3.12 installed.

Prefix commands with `uv run` to execute them inside that environment.

# How to use?

Your features must be input as CSV files (one for the internal dataset, one for the external test) in the following format :
- each line is a patient
- each column is a feature except for 2 columns :
    - one column must be named "hvpg" (if you do regression) or "csph" (if you do classification).
    - one column must be named "split" and contain either "train", "val", or "test".

The rest of the columns must be features. The identifier and metadata columns listed in `COLUMNS_TO_DROP` in `src/train.py` (`hvpg`, `csph`, `split`, `sample_uuid`, `patient_uuid`) are dropped automatically.

To train the regression, launch the following command line :

```bash
uv run python src/train.py \
    --df_features_internal path/to/features/for/internal/dataset.csv \
    --df_features_external path/to/features/for/external/dataset.csv \
    --output_folder path/to/your/desired/output/folder \
    --method radiomics \
    --penalty l2
```

The options for the `--method` argument are "biomarkers", "radiomics" and "fm"; the options for the `--penalty` argument are "l1" and "l2". Both are required.

Use "biomarkers" to train one model per feature combination listed in `src/biomarkers_composition_combinations.json` (liver surface nodularity, clinical, serum, and all the morphological features); point `--biomarker_combinations` at your own file to change that list.

# Miscellaneous

## Selecting the number of PCA components

Features can be fed to `train.py` as they are, or reduced by PCA beforehand. The number of components is chosen automatically with the Bayesian Information Criterion (BIC) rather than swept over a fixed grid.

scikit-learn's `PCA` is a Gaussian probabilistic PCA model, so for each candidate dimensionality the total log-likelihood on the standardised training features is combined with the model's free-parameter count to give a BIC; the selected number of components is the argmin. The scaler and the PCA are fit on the **train split only**, and the endpoints (`hvpg`, `csph`) are always excluded, so nothing leaks into the reduction.

```bash
uv run python utils/select_pca_components.py \
    --input_path path/to/features/for/internal/dataset.csv \
    --input_path_external path/to/features/for/external/dataset.csv \
    --output_path path/to/your/desired/output/folder \
    --columns_to_drop sample_uuid split patient_uuid hvpg csph
```

It writes the reduced feature CSVs (`internal_bic_<k>.csv`, `external_bic_<k>.csv`) that you can input to `train.py`, alongside the BIC curve, a scree plot and `bic_n_components.txt`.

The results published in `src/statistics/regression` correspond to these BIC-selected component counts — for instance the Curia model uses 21 components and the radiomics model 23.

## Compute the portal vein's largest diameter

An example of how to compute the portal vein's largest diameter is displayed in `src/portal_vein/compute_diameter_pv.py`.

## Extract radiomics features

An example of how to extract radiomics features is displayed in `src/radiomics/extract_radiomics_features.py`.

PyRadiomics ships no wheels for recent Python versions, so it is declared as an optional dependency and is **not** installed by `uv sync`. Run this script from a separate Python 3.9 environment:

```bash
uv run --python 3.9 --with pyradiomics --with SimpleITK python src/radiomics/extract_radiomics_features.py \
    --dataset_path path/to/dataset
```

## Re-run statistical analysis

The predictions of all the models of the article are stored in `src/statistics/regression` and can be used as is to re-run DeLong's tests with the command line :
```bash
uv run python src/statistics/delong_tests.py
```

Use `--threshold 10` (default) or `--threshold 16` to select the HVPG cut-off. The pairwise test covers the five models compared in the paper; BiomedCLIP and MedImageInsight are reported separately and are commented out in `RESULTS_PATHS` usage — uncomment them in `main()` to include them.
