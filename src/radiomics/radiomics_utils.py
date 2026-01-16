import numpy as np
import SimpleITK as sitk
from radiomics import getFeatureClasses


def load_numpy_to_sitk(image_path, mask_path, original_image_sitk=None):
    """Converts NumPy arrays to SimpleITK images.
    data type is UInt32 for the mask"""

    # first convert the image to float
    image_array = np.load(image_path)
    mask_array = np.load(mask_path)

    image_sitk = sitk.GetImageFromArray(image_array)
    mask_sitk = sitk.GetImageFromArray(mask_array)

    if original_image_sitk is not None:
        image_sitk.CopyInformation(original_image_sitk)
        mask_sitk.CopyInformation(original_image_sitk)
    return image_sitk, mask_sitk


def load_nifti_to_sitk(image_path, mask_path):
    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)
    return image_sitk, mask_sitk


def convert_to_python_type(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(v) for v in obj]
    else:
        return obj


def get_feature_names_and_counts():
    featuresClasses = getFeatureClasses()
    print("\nNumber of features in each class:")
    for class_name, class_obj in featuresClasses.items():
        # Get the feature names for this class
        feature_names = class_obj.getFeatureNames()
        print(f"{class_name}: {len(feature_names)} features")
        print("\n")
        # Optionally print the feature names
        # print(f"Features: {feature_names}\n")
