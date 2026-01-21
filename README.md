# PHoundation

This is the repository associated with the paper "Estimating clinically significant portal hypertension: foundation models versus automated non-invasive tests". The paper benchmarks several non-invasive tests, including foundation models, to estimate the severity of portal hypertension (PH). 
This repository enables training a cross-validated regression to predict Hepatic Venous Pressure Gradient (HVPG) using a set of numerical features.

# How to install?

Run `poetry install` in your terminal to install the environment. 
If you do not have `poetry`, install it with pipx (`pipx install poetry`)
Make sure you have Python >= 3.9 installed. 

# How to use? 

Your features must be input as CSV files (one for the internal dataset, one for the external test) in the following format : 
- each line is a patient 
- each column is a feature except for 2 columns : 
    - one column must be named "hvpg" (if you do regression) or "csph" (if you do classification). 
    - one column must be named "split" and contain either "train", "val", or "test". 

The rest of the column must be features. To drop columns that are not features (column that is a patient's identifiers for instance), use the argument --column_to_drop 

To train the regression, launch the following command line :

```bash
poetry run python train.py \
    --df_features_internal path/to/features/for/internal/dataset.csv \
    --df_features_external path/to/features/for/external/dataset.csv \
    --output_folder path/to/your/desired/output/folder \
    --column_to_drop patient_id \
    --penalty l2
```

The options for the penalty argument are "l1", "l2".

# Miscellaneous 

## Using PCA features 

You can feed your features as they are, or apply a PCA on it beforehand. To do so, launch the following command : 

```bash
poetry run python utils/create_pca_features.py \
    --input_path_internal path/to/features/for/internal/dataset.csv \
    --input_path_external path/to/features/for/external/dataset.csv \
    --columns_to_drop "sample_uuid" "split" "patient_uuid" "hvpg" "csph" "study_type"
```

It will output new CSV files that you can input to train.py. 

## Compute the portal vein's largest diameter 

An example of how to extract radiomics features is displayed in src/radiomics/extract_radiomics_features.py.

## Extract radiomics features 

An example of how to compute the portal vein's largest diameter is displayed in src/portal_vein/compute_diameter_pv.py

## Re-run statistical analysis 

The predictions of all the models of the article are stored in `src/statistics/regression_results` and can be used as is to re-run Delong's tests with the command line : 
```bash
poetry run python src/statistics/delong_tests.py   
```

