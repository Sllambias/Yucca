import nibabel as nib
import nibabel.processing as nibpro
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data


def convert(path: str, subdir: str = "decathlon/Task02_Heart"):
    
    # INPUT DATA
    # Define input path and extension
    path = join(path, subdir)
    file_extension = ".nii.gz"

    # OUTPUT DATA
    # Define the task name and prefix
    task_name = "Task022_Heart"
    task_prefix = "Heart"

    # Set target paths
    target_base = join(yucca_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # Split data
    images_dir = join(path, "imagesTr")
    labels_dir = join(path, "labelsTr")
    samples = subfiles(labels_dir, join=False, suffix=file_extension)
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=1243)
    images_dir_tr = images_dir_ts = images_dir
    labels_dir_tr = labels_dir_ts = labels_dir


    # Populate Target Directory
    # This is also the place to apply any re-orientation, resampling and/or label correction.

    for sTr in train_samples:
        image = nib.load(join(images_dir_tr, sTr))
        label = nib.load(join(labels_dir_tr, sTr))
        sTr = sTr[: -len(file_extension)]

        # Orient to RAS and register image-label, using the image as reference.
        #image = nibpro.resample_from_to(image, label, order=3)

        nib.save(image, filename=f"{target_imagesTr}/{task_prefix}_{sTr}_000.nii.gz")
        nib.save(label, filename=f"{target_labelsTr}/{task_prefix}_{sTr}.nii.gz")

    for sTs in test_samples:
        image = nib.load(join(images_dir_ts, sTs))
        label = nib.load(join(labels_dir_ts, sTs))
        sTs = sTs[: -len(file_extension)]

        # Orient to RAS and register image-label, using the image as reference.
        #image = nibpro.resample_from_to(image, label, order=3)

        nib.save(image, filename=f"{target_imagesTs}/{task_prefix}_{sTs}_000.nii.gz")
        nib.save(label, filename=f"{target_labelsTs}/{task_prefix}_{sTs}.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("T1",),
        labels={0: "Background", 1: "Left Atrium"},
        dataset_name=task_name,
        license="CC-BY-SA 4.0",
        dataset_description="Decathlon: Left Atrium Segmentation",
        dataset_reference="King's College London",
    )
