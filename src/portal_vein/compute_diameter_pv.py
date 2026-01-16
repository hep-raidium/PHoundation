import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import nibabel as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def compute_tubular_diameter(segmentation: np.ndarray, resolution: List[float]) -> Dict:
    """
    Compute the maximum diameter of the PV in a given segmentation

    Segmentation is a 3D numpy array of shape (H, W, num_slices)
    Resolution is a list of 3 floats (res_x, res_y, res_z) representing the voxel size in mm
    """
    # Compute Euclidian distance of each voxel to the nearest background voxel
    radiuses = distance_transform_edt(segmentation, sampling=resolution)  # edt = euclidian distance transforms

    # Ensure no NaN values in radiuses array
    if np.isnan(radiuses).any():
        raise ValueError("NaN values found in distance transform results")

    # Calculate statistics
    max_diameter = np.max(radiuses) * 2

    metrics = {
        "max_diameter": max_diameter,
        "radiuses": radiuses,
    }

    return metrics


def process_single_sample(sample, input_path, output_path, save_radiuses=False):

    segmentation = nb.load(  # type: ignore
        input_path / sample["sample_uuid"] / "labels" / "portal_vein_and_splenic_vein.nii.gz"
    )

    metrics = compute_tubular_diameter(
        segmentation.get_fdata(),  # type: ignore
        resolution=sample["resolution"],
    )

    if save_radiuses:
        output_dir = output_path / sample["sample_uuid"]
        output_dir.mkdir(parents=True, exist_ok=True)
        nb.save(  # type: ignore
            nb.Nifti1Image(metrics["radiuses"], segmentation.affine),  # type: ignore
            output_dir / "radiuses.nii.gz",
        )

    return (
        sample["patient_uuid"],
        sample["sample_uuid"],
        sample["study_type"],
        metrics["max_diameter"],
    )


def main(args):
    with open(args.input_path / "samples.json") as f:
        samples = json.load(f)

    # Process samples in parallel
    results = Parallel(n_jobs=20)(
        delayed(process_single_sample)(sample, args.input_path, args.output_path, save_radiuses=(i < 5))
        for i, sample in tqdm(enumerate(samples), total=len(samples))
    )

    diameters = defaultdict(list)
    for result in results:
        if result is not None:
            diameters[result[0]].append(
                {
                    "sample_uuid": result[1],
                    "study_type": result[2],
                    "diameter": result[3],
                }
            )

    print("PV diameters computed for ", len(diameters), " patients")

    with open(args.output_path / "diameters.json", "w") as f:
        json.dump(diameters, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    main(args)
