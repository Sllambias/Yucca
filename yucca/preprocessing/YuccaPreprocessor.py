import numpy as np
import torch
import nibabel as nib
import os
import cc3d
from yuccalib.image_processing.objects.BoundingBox import get_bbox_for_foreground
from yuccalib.image_processing.cropping_and_padding import crop_to_box, pad_to_size
from yuccalib.utils.nib_utils import get_nib_spacing, get_nib_orientation, reorient_nib_image
from yuccalib.utils.type_conversions import nib_to_np
from yucca.paths import yucca_preprocessed, yucca_raw_data
from yucca.preprocessing.normalization import normalizer
from multiprocessing import Pool
from skimage.transform import resize
from batchgenerators.utilities.file_and_folder_operations import join, load_json, subfiles, \
    save_pickle, maybe_mkdir_p, isfile


class YuccaPreprocessor(object):
    """
    The default preprocesser. This takes in a plans file (created by an YuccaPlanner) and carries 
    out preprocessing according to rules set by the plan.

    The operations that can be enabled/defined in the YuccaPlanner and carried out by the
    YuccaPreprocessor are:

    The starting orientation - defaults to RAS (for medical images).

    The cropping operation - defaults to crop to nonzero bounding box

    The Transposition operation (along with the reverse transpose operation,
    to be used during inference) - defaults to no transposition if image dimensions and spacings
    are not too anisotropic.

    The Resample operation - defaults to resampling to the median spacing of the dataset.

    The Normalization operation - defaults to standardization = (image - mean) / std
    per modality to preserve ranges to account for CT pixel values representing specific physical
    attributes.

    Additionally it carries out a number of tests and analyzes each image for foreground locations
    which is used later to oversample foreground.
    """
    def __init__(self, plans_path, task=None, threads=12, disable_unittests=False):
        self.name = str(self.__class__.__name__)
        self.task = task
        self.plans_path = plans_path
        self.plans = load_json(plans_path)
        self.threads = threads
        self.disable_unittests = disable_unittests

        # lists for information we would like to attain
        self.transpose_forward = []
        self.transpose_backward = []
        self.target_spacing = []

    def initialize_paths(self):
        self.target_dir = join(yucca_preprocessed, self.task, self.plans['plans_name'])
        self.input_dir = join(yucca_raw_data, self.task)
        self.imagepaths = subfiles(join(self.input_dir, 'imagesTr'), suffix='.nii.gz')

    def initialize_properties(self):
        """
        here we basically set up things that are needed for preprocessing during training,
        but that aren't necessary during inference
        """
        self.dataset_properties = self.plans['dataset_properties']
        self.intensities = self.dataset_properties['intensities']

        # op values
        self.transpose_forward = np.array(self.plans['transpose_forward'])
        self.transpose_backward = np.array(self.plans['transpose_backward'])
        self.target_spacing = np.array(self.plans['target_spacing'])

    def run(self):
        self.initialize_properties()
        self.initialize_paths()
        maybe_mkdir_p(self.target_dir)
        subject_ids = subfiles(join(self.input_dir, 'labelsTr'), suffix='.nii.gz', join=False)

        print(f"{'Preprocessing Task:':25.25} {self.task} \n"
              f"{'Using Planner:':25.25} {self.plans_path} \n"
              f"{'Crop to nonzero:':25.25} {self.plans['crop_to_nonzero']} \n"
              f"{'Normalization scheme:':25.25} {self.plans['normalization_scheme']} \n"
              f"{'Transpose Forward:':25.25} {self.transpose_forward} \n"
              f"{'Transpose Backward:':25.25} {self.transpose_backward} \n")
        p = Pool(self.threads)
        p.map(self._preprocess_train_subject, subject_ids)
        p.close()
        p.join()

    def _preprocess_train_subject(self, subject_id):
        image_props = {}
        subject_id = subject_id[:-len('.nii.gz')]
        print(f"Preprocessing: {subject_id}")
        arraypath = join(self.target_dir, subject_id + '.npy')
        picklepath = join(self.target_dir, subject_id + '.pkl')

        if isfile(arraypath) and isfile(picklepath):
            print(f"Case: {subject_id} already exists. Skipping.")
            return
        # First find relevant images by their paths and save them in the image property pickle
        # Then load them as images
        # The '_' in the end is to avoid treating Case_4_000 AND Case_42_000 as different versions
        # of the seg named Case_4 as both would start with "Case_4", however only the correct one is
        # followed by an underscore 
        imagepaths = [impath for impath in self.imagepaths if os.path.split(impath)[-1].startswith(subject_id + '_')]
        image_props['image files'] = imagepaths
        images = [nib.load(image) for image in imagepaths]

        # Do the same with segmentation
        seg = join(self.input_dir, 'labelsTr', subject_id + '.nii.gz')
        image_props['segmentation file'] = seg
        seg = nib.load(seg)
        
        if not self.disable_unittests:
            assert len(images) > 0, f"found no images for {subject_id + '_'}, " \
                f"attempted imagepaths: {imagepaths}"

            assert len(images[0].shape) == self.plans['dataset_properties']['data_dimensions'], \
                f"image should be shape (x, y(, z)) but is {images[0].shape}"

            # make sure images and labels are correctly registered
            assert images[0].shape == seg.shape, f"Sizes do not match for {subject_id}" \
                f"Image is: {images[0].shape} while the seg is {seg.shape}"

            assert np.allclose(get_nib_spacing(images[0]), get_nib_spacing(seg)), \
                f"Spacings do not match for {subject_id}" \
                f"Image is: {get_nib_spacing(images[0])} while the seg is {get_nib_spacing(seg)}"

            assert get_nib_orientation(images[0]) == get_nib_orientation(seg), \
                f"Directions do not match for {subject_id}" \
                f"Image is: {get_nib_orientation(images[0])} while the seg is {get_nib_orientation(seg)}"

            # Make sure all modalities are correctly registered
            if len(images) > 1:
                for image in images:
                    assert images[0].shape == image.shape, f"Sizes do not match for {subject_id}" \
                        f"One is: {images[0].shape} while another is {image.shape}"

                    assert np.allclose(get_nib_spacing(images[0]), get_nib_spacing(image)), \
                        f"Spacings do not match for {subject_id}" \
                        f"One is: {get_nib_spacing(images[0])} while another is {get_nib_spacing(image)}"

                    assert get_nib_orientation(images[0]) == get_nib_orientation(image), \
                        f"Directions do not match for {subject_id}" \
                        f"One is: {get_nib_orientation(images[0])} while another is {get_nib_orientation(image)}"

        original_spacing = get_nib_spacing(images[0])
        original_size = np.array(images[0].shape)

        if self.target_spacing.size:
            target_spacing = self.target_spacing
        else:
            target_spacing = original_spacing

        # If qform and sform are both missing the header is corrupt and we do not trust the 
        # direction from the affine
        # Make sure you know what you're doing
        if images[0].get_qform(coded=True)[1] or images[0].get_sform(coded=True)[1]:
            original_orientation = get_nib_orientation(images[0])
            final_direction = self.plans['target_coordinate_system']
            images = [nib_to_np(reorient_nib_image(image, original_orientation, final_direction)) for image in images]
            seg = nib_to_np(reorient_nib_image(seg, original_orientation, final_direction))
        else:
            original_orientation = 'INVALID'
            final_direction = 'INVALID'
            images = [nib_to_np(image) for image in images]
            seg = nib_to_np(seg)

        # Check if the ground truth only contains expected values
        expected_labels = np.array(self.plans['dataset_properties']['classes'], dtype=np.float32)
        actual_labels = np.unique(seg).astype(np.float32)
        assert np.all(np.isin(actual_labels, expected_labels)), f"Unexpected labels found for {subject_id} \n"\
            f"expected: {expected_labels} \n"\
            f"found: {actual_labels}"

        # Cropping is performed to save computational resources. We are only removing background.
        if self.plans['crop_to_nonzero']:
            nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
            image_props['crop_to_nonzero'] = nonzero_box
            for i in range(len(images)):
                images[i] = crop_to_box(images[i], nonzero_box)
            seg = crop_to_box(seg, nonzero_box)
        else:
            image_props['crop_to_nonzero'] = self.plans['crop_to_nonzero']

        images, seg = self._resample_and_normalize_case(images, seg,
                                                   self.plans['normalization_scheme'],
                                                   self.transpose_forward,
                                                   original_spacing,
                                                   target_spacing)

        # Stack and fix dimensions
        images = np.vstack((np.array(images), np.array(seg)[np.newaxis]))

        # now AFTER transposition etc., we get some (no need to get all)
        # locations of foreground, that we will later use in the
        # oversampling of foreground classes
        foreground_locs = np.array(np.nonzero(images[-1])).T[::10]
        numbered_ground_truth, ground_truth_numb_lesion = cc3d.connected_components(images[-1], connectivity=26, return_N=True)
        if ground_truth_numb_lesion == 0:
            object_sizes = 0
        else:
            object_sizes = [i * np.prod(target_spacing) for i in np.unique(numbered_ground_truth, return_counts=True)[-1][1:]]
        
        final_size = list(images[0].shape)

        # save relevant values
        image_props['original_spacing'] = original_spacing
        image_props['original_size'] = original_size
        image_props['original_orientation'] = original_orientation
        image_props['new_spacing'] = target_spacing[self.transpose_forward].tolist()
        image_props['new_size'] = final_size
        image_props['new_direction'] = final_direction
        image_props['foreground_locations'] = foreground_locs
        image_props['n_cc'] = ground_truth_numb_lesion
        image_props['size_cc'] = object_sizes


        print(
            f"size before: {original_size} size after: {image_props['new_size']} \n"
            f"spacing before: {original_spacing} spacing after: {image_props['new_spacing']} \n"
            f"Saving {subject_id} in {arraypath} \n")

        # save the image
        np.save(arraypath, images)

        # save metadata as .pkl
        save_pickle(image_props, picklepath)

    def _resample_and_normalize_case(self, images: list, seg: np.ndarray = None,
                                     norm_op=None, transpose=None, original_spacing=None,
                                     target_spacing=None):

        # Normalize and Transpose images to target view.
        # Transpose segmentations to target view.
        assert len(images) == len(norm_op) == len(self.intensities), "number of images, " \
            "normalization  operations and intensities does not match. \n"\
            f"len(images) == {len(images)} \n"\
            f"len(norm_op) == {len(norm_op)} \n"\
            f"len(self.intensities) == {len(self.intensities)} \n"

        for i in range(len(images)):
            images[i] = normalizer(images[i], scheme=norm_op[i], intensities=self.intensities[i])
            assert len(images[i].shape) == len(transpose), "image and transpose axes do not match. \n"\
                f"images[i].shape == {images[i].shape} \n"\
                f"transpose == {transpose} \n"\
                f"len(images[i].shape) == {len(images[i]).shape} \n"\
                f"len(transpose) == {len(transpose)} \n"
            images[i] = images[i].transpose(transpose)
        print(f"Normalized with: {norm_op[0]} \n"
              f"Transposed with: {transpose}")

        shape_t = images[0].shape
        original_spacing_t = original_spacing[transpose]
        target_spacing_t = target_spacing[transpose]

        # Find new shape based on the target spacing
        target_shape = np.round((original_spacing_t / target_spacing_t).astype(float) * shape_t).astype(int)

        # Resample to target shape and spacing
        for i in range(len(images)):
            try:
                images[i] = resize(images[i], output_shape=target_shape, order=3)
            except OverflowError:
                print("Unexpected values in either shape or image for resize")
        if seg is not None:
            seg = seg.transpose(transpose)
            try:
                seg = resize(seg, output_shape=target_shape, order=0, anti_aliasing=False)
            except OverflowError:
                print("Unexpected values in either shape or seg for resize")
            return images, seg
        
        return images

    def preprocess_case_for_inference(self, images: list | tuple, patch_size: tuple):
        """
        Will reorient ONLY if we have valid qform or sform codes.
        with coded=True the methods will return {affine or None} and {0 or 1}.
        If both are 0 we cannot rely on headers for orientations and will
        instead assume images are in the desired orientation already.

        Afterwards images will be normalized and transposed as specified by the
        plans file also used in training.

        Finally images are resampled to the required spacing/size and returned
        as torch tensors of the required shape (b, c, x, y, (z))
        """
        assert isinstance(images, (list, tuple)), "image(s) should be a list or tuple, even if only one "\
            "image is passed"
        self.initialize_properties()
        image_properties = {}
        images = [nib.load(image[0]) if isinstance(image, tuple) else nib.load(image) for image in images]

        image_properties['original_spacing'] = get_nib_spacing(images[0])
        image_properties['original_shape'] = np.array(images[0].shape)
        image_properties['qform'] = images[0].get_qform()
        image_properties['sform'] = images[0].get_sform()

        assert len(image_properties['original_shape']) in [2, 3], "images must be either 2D or 3D for preprocessing" 

        # Check if header is valid and then attempt to orient to target orientation.
        if (images[0].get_qform(coded=True)[1] or images[0].get_sform(coded=True)[1] and
            self.plans.get('target_coordinate_system')):
                image_properties['reoriented'] = True
                original_orientation = get_nib_orientation(images[0])
                image_properties['original_orientation'] = original_orientation
                images = [reorient_nib_image(image, original_orientation,
                                         self.plans['target_coordinate_system']) for image in images]
                image_properties['new_orientation'] = get_nib_orientation(images[0])
        else:
            print("Insufficient header information. Reorientation will not be attempted.")
            image_properties['reoriented'] = False

        image_properties['affine'] = images[0].affine
        images = [nib_to_np(image) for image in images]

        image_properties['uncropped_shape'] = np.array(images[0].shape)

        if self.plans['crop_to_nonzero']:
            nonzero_box = get_bbox_for_foreground(images[0], background_label=0)
            for i in range(len(images)):
                images[i] = crop_to_box(images[i], nonzero_box)
            image_properties['nonzero_box'] = nonzero_box

        image_properties['cropped_shape'] = np.array(images[0].shape)

        images = self._resample_and_normalize_case(images,
                                                   norm_op=self.plans['normalization_scheme'],
                                                   transpose=self.transpose_forward,
                                                   original_spacing=image_properties['original_spacing'],
                                                   target_spacing=self.target_spacing)

        # From this point images are shape (1, c, x, y, z)
        image_properties['resampled_transposed_shape'] = np.array(images[0].shape)
        
        for i in range(len(images)):
            images[i], padding = pad_to_size(images[i], patch_size)
        image_properties['padded_shape'] = np.array(images[0].shape)
        image_properties['padding'] = padding

        # Stack and fix dimensions
        images = np.stack(images)[np.newaxis]

        return torch.tensor(images, dtype=torch.float32), image_properties

    def reverse_preprocessing(self, images: np.ndarray, image_properties: dict):
        """
        At this point images are potentially: cropped, transposed, resized and padded (in this order).

        First we undo the padding
        Then we reverse resizing to ensure correct spacing
        Then we transpose back to the original view 
        Finally we re-seat the cropped out part (if cropping is enabled)

        Expected shape of images are:
        (b, c, x, y(, z))

        The original orientation of the image will be re-applied when saving the prediction
        """
    
        nclasses = len(self.plans['dataset_properties']['classes'])
        original_shape = image_properties['original_shape']
        canvas = np.zeros((1, nclasses, *image_properties['uncropped_shape']))
        shape_after_crop = image_properties['cropped_shape']
        shape_after_crop_transposed = shape_after_crop[self.transpose_forward]
        pad = image_properties['padding']

        for b in range(images.shape[0]):

            for c in range(images.shape[1]):
                image = images[b,c]

                assert np.all(image.shape == image_properties['padded_shape']), f"Reversing padding: "\
                    f"image should be of shape: {image_properties['padded_shape']}"\
                    f"but is: {image.shape}"
                shape = image.shape
                if len(pad) > 5:
                    image = image[pad[0]:shape[0]-pad[1], pad[2]:shape[1]-pad[3], pad[4]:shape[2]-pad[5]]
                elif len(pad) < 5:
                    image = image[pad[0]:shape[0]-pad[1], pad[2]:shape[1]-pad[3]]

                assert np.all(image.shape == image_properties['resampled_transposed_shape']), f"Reversing resampling and tranposition: "\
                    f"image should be of shape: {image_properties['resampled_transposed_shape']}"\
                    f"but is: {image.shape}"
                image = resize(image, output_shape=shape_after_crop_transposed, order=3).transpose(self.transpose_backward)
                
                assert np.all(image.shape == image_properties['cropped_shape']), f"Reversing cropping: "\
                    f"image should be of shape: {image_properties['cropped_shape']}"\
                    f"but is: {image.shape}"

                if self.plans['crop_to_nonzero']:
                    bbox = image_properties['nonzero_box']
                    if len(bbox) > 5:
                        slices = (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]), slice(bbox[4], bbox[5]))
                    elif len(bbox) < 5:
                        slices = (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]))
                    canvas[b, c][slices] = image
        return canvas

