from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np
import shutil


def convert(path: str, subdir: str = "brats21/training_data"):
    # Target names
    task_name = "Task040_BraTS21_flair.py"
    task_prefix = "BraTS21"

    ###OUTPUT DATA
    # Target paths
    target_base = join(yucca_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # INPUT DATA
    # Input path and names
    base_in = join(path, subdir)
    file_suffix = ".nii.gz"

    # Train/Test Splits
    # We only use the train folder, as the test folder does not contain segmentations
    # to obtain those, submission to the challenge is required (it's from 2008, forget it)
    training_samples, test_samples = train_test_split(subdirs(base_in, join=False), random_state=4215532)

    ###Populate Target Directory###
    for sTr in training_samples:
        src_image_file_path1 = join(base_in, sTr, sTr + "_flair" + file_suffix)
        dst_image_file_path1 = f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz"

        dst_label_path = f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz"
        label = nib.load(join(base_in, sTr, sTr + "_seg" + file_suffix))
        labelarr = label.get_fdata()
        labelarr[labelarr == 4.0] = 3.0
        assert np.all(np.isin(np.unique(labelarr), np.array([0, 1, 2, 3])))
        labelnew = nib.Nifti1Image(labelarr, label.affine, label.header)
        nib.save(labelnew, dst_label_path)

        shutil.copy2(src_image_file_path1, dst_image_file_path1)

    for sTs in test_samples:
        src_image_file_path1 = join(base_in, sTs, sTs + "_flair" + file_suffix)
        dst_image_file_path1 = f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz"

        dst_label_path = f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz"
        label = nib.load(join(base_in, sTs, sTs + "_seg" + file_suffix))
        labelarr = label.get_fdata()
        labelarr[labelarr == 4.0] = 3.0
        assert np.all(np.isin(np.unique(labelarr), np.array([0, 1, 2, 3])))
        labelnew = nib.Nifti1Image(labelarr, label.affine, label.header)
        nib.save(labelnew, dst_label_path)

        shutil.copy2(src_image_file_path1, dst_image_file_path1)

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        ("FLAIR"),
        labels={0: "background", 1: "necrotic tumor core", 2: "peritumoral edematous/invaded tissue", 3: "GD-enhancing tumor"},
        dataset_name=task_name,
        license="hands off!",
        dataset_description="BraTS21",
        dataset_reference="https://www.nitrc.org/projects/msseg, https://arxiv.org/abs/2206.06694",
    )


if __name__ == "__main__":
    convert()
