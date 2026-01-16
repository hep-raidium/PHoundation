import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import radiomics
from joblib import Parallel, delayed
from radiomics import featureextractor
from radiomics_utils import load_nifti_to_sitk
from tqdm import tqdm

radiomics.setVerbosity(logging.INFO)


def create_parameters_for_radiomics(output_path):
    """
    dump a json file with the parameters for the radiomics
    """
    output_path.mkdir(parents=True, exist_ok=True)
    parameters = {
        "setting": {
            "minimumROIDimensions": 2,
            "minimumROISize": None,
            "normalize": False,  # False is recommended for CT images
            "removeOutliers": None,
            "resampledPixelSpacing": (0.75, 0.75, 1.25),  # changed from defaults to 0.75mm (closest to our own data)
            "interpolator": "sitkBSpline",
            "preCrop": False,
            "padDistance": 5,
            "binWidth": 15,
            "binCount": None,
            "distances": [1],
            "force2D": False,
            "force2Ddimension": 0,
            "resegmentRange": None,
            "label": 1,
            "additionalInfo": True,
        },
        "imageType": {
            "Original": {},
            "Wavelet": {},
            "LoG": {
                "sigma": [1.0, 3.0, 5.0],
            },
        },
        "featureClass": {"shape": [], "firstorder": [], "glcm": [], "glrlm": [], "glszm": [], "gldm": [], "ngtdm": []},
    }

    with open(output_path / "parameters.json", "w") as f:
        json.dump(parameters, f)
    return parameters


def keep_only_features(features):
    """
    Getting rid of the "diagnostic" features that are cumbersome and
    not easy to handle because of their shape (dict of dict, list of list, etc.)
    """
    cleaned_features = {}
    for k, v in features.items():
        if "diagnostics" in k:
            continue
        cleaned_features[k] = v
    return cleaned_features


def process_single_sample(sample, dataset_path, extractor, metadata, organ_of_interest, output_path):
    """
    Process a single sample and return its features
    """

    image_path = dataset_path / sample["sample_uuid"] / "ct.nii.gz"
    all_organs_features = {}
    if organ_of_interest == "all":
        for organ_of_interest in ["spleen", "liver"]:
            mask_path = dataset_path / sample["sample_uuid"] / "labels" / f"{organ_of_interest}.nii.gz"
            image_sitk, mask_sitk = load_nifti_to_sitk(image_path, mask_path)
            features = extractor.execute(image_sitk, mask_sitk)
            features = keep_only_features(features)
            renamed_features = {f"{organ_of_interest}_{k}": v for k, v in features.items()}
            all_organs_features.update(renamed_features)

    else:  # either spleen or liver
        mask_path = dataset_path / sample["sample_uuid"] / "labels" / f"{organ_of_interest}.nii.gz"
        image_sitk, mask_sitk = load_nifti_to_sitk(image_path, mask_path)
        features = extractor.execute(image_sitk, mask_sitk)
        features = keep_only_features(features)
        renamed_features = {f"{organ_of_interest}_{k}": v for k, v in features.items()}
        all_organs_features.update(renamed_features)

    hvpg = metadata[metadata["patient_uuid"] == sample["patient_uuid"]]["Gradient (hvpg)"].values[0]
    csph = hvpg >= 10

    row = pd.DataFrame(
        [
            {
                "sample_uuid": sample["sample_uuid"],
                "patient_uuid": sample["patient_uuid"],
                "split": sample["split"],
                "study_type": sample["study_type"],
                "hvpg": hvpg,
                "csph": csph,
                **all_organs_features,
            }
        ]
    )
    return row


def main(dataset_path, output_path, organ_of_interest):
    """
    Process all samples in parallel using joblib
    """
    with open(dataset_path / "samples.json", "r") as f:
        samples = json.load(f)

    metadata = pd.read_csv(dataset_path / "metadata.csv")

    create_parameters_for_radiomics(output_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(str(output_path / "parameters.json"))

    # Process samples in parallel
    results = Parallel(n_jobs=10)(
        delayed(process_single_sample)(sample, dataset_path, extractor, metadata, organ_of_interest, output_path)
        for sample in tqdm(samples, total=len(samples))
    )

    # Combine all results
    all_features = pd.concat(results, ignore_index=True)
    dataset = "internal" if "internal" in str(dataset_path).lower() else "external"
    all_features.to_csv(output_path / f"radiomics_features_{dataset}_{organ_of_interest}.csv", index=False)

    print(
        "The radiomics features have been extracted for dataset",
        dataset_path,
        "and organ of interest",
        organ_of_interest,
    )
    print("There are", all_features.shape, "features in the dataframe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default="src/raidium/rd/results/features/radiomics_features")
    parser.add_argument("--organ_of_interest", type=str, default="spleen")
    args = parser.parse_args()
    main(args.dataset_path, args.output_path, args.organ_of_interest)
