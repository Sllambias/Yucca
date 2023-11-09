# %%
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, subdirs
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
from yucca.utils.nib_utils import get_nib_orientation, reorient_nib_image, resample_nib_image
from sklearn.model_selection import train_test_split
import nibabel as nib
import nibabel.processing as nibpro
import os

# INPUT DATA
# Input path and names
base_in = "/maps/projects/image/people/zcr545/datasets/ISLES-2022"
file_suffix = ".nii.gz"

# Train/Test Splits
images_dir = join(base_in, "images")
labels_dir = join(base_in, "labels_derivatives")

training_samples, test_samples = train_test_split(subdirs(labels_dir, join=False), random_state=859032)

###OUTPUT DATA
# Target names
task_name = "Task044_ISLES22_DWI"
prefix = "ISLES22_DWI"

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

###Populate Target Directory###
# This is likely also the place to apply any re-orientation, resampling and/or label correction.
for sTr in training_samples:
    image_file = nib.load(join(images_dir, sTr, "ses-0001", "dwi", sTr + "_ses-0001_dwi" + file_suffix))

    label = nib.load(join(labels_dir, sTr, "ses-0001", sTr + "_ses-0001_msk" + file_suffix))
    target_orientation = get_nib_orientation(image_file)
    label_orientation = get_nib_orientation(label)
    label = reorient_nib_image(label, label_orientation, target_orientation)
    label = resample_nib_image(label, target_image=image_file, is_seg=True)

    nib.save(image_file, f"{target_imagesTr}/{prefix}_{sTr}_000.nii.gz")
    nib.save(label, f"{target_labelsTr}/{prefix}_{sTr}.nii.gz")

for sTs in test_samples:
    image_file = nib.load(join(images_dir, sTs, "ses-0001", "dwi", sTs + "_ses-0001_dwi" + file_suffix))

    label = nib.load(join(labels_dir, sTs, "ses-0001", sTs + "_ses-0001_msk" + file_suffix))
    target_orientation = get_nib_orientation(image_file)
    label_orientation = get_nib_orientation(label)
    label = reorient_nib_image(label, label_orientation, target_orientation)
    label = resample_nib_image(label, target_image=image_file, is_seg=True)

    nib.save(image_file, f"{target_imagesTs}/{prefix}_{sTs}_000.nii.gz")
    nib.save(label, f"{target_labelsTs}/{prefix}_{sTs}.nii.gz")

generate_dataset_json(
    join(target_base, "dataset.json"),
    target_imagesTr,
    target_imagesTs,
    ("DWI",),
    labels={0: "background", 1: "Infarct (ischemic stroke lesion)"},
    dataset_name=task_name,
    license="hands off!",
    dataset_description="ISLES-2022",
    dataset_reference="https://arxiv.org/abs/2206.06694",
)
