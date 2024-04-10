from yucca.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np


# Stolen from Batchgenerators to avoid import error caused by deprecated modules imported in
# Batchgenerators.
def augment_gamma(
    data_sample,
    gamma_range=(0.5, 2),
    invert_image=False,
    epsilon=1e-7,
    per_channel=False,
    clip_to_input_range=False,
):
    if invert_image:
        data_sample = -data_sample

    if not per_channel:
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        img_min = data_sample.min()
        img_max = data_sample.max()
        img_range = img_max - img_min
        data_sample = np.power(((data_sample - img_min) / float(img_range + epsilon)), gamma) * img_range + img_min
        if clip_to_input_range:
            data_sample = np.clip(data_sample, a_min=img_min, a_max=img_max)
    else:
        for c in range(data_sample.shape[0]):
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            img_min = data_sample[c].min()
            img_max = data_sample[c].max()
            img_range = img_max - img_min
            data_sample[c] = (
                np.power(((data_sample[c] - img_min) / float(img_range + epsilon)), gamma) * float(img_range + epsilon)
                + img_min
            )
            if clip_to_input_range:
                data_sample[c] = np.clip(data_sample[c], a_min=img_min, a_max=img_max)
    if invert_image:
        data_sample = -data_sample
    return data_sample


class Gamma(YuccaTransform):
    """
    WRAPPER FOR NNUNET AUGMENT GAMMA: https://github.com/MIC-DKFZ/batchgenerators/blob/8822a08a7dbfa4986db014e6a74b040778164ca6/batchgenerators/augmentations/color_augmentations.py

    Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

    :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
    larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
    Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
    smaller than 1 and the other half with > 1
    :param invert_image: whether to invert the image before applying gamma augmentation
    :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
    the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
    can also be callable (signature retain_stats() -> bool)
    """

    def __init__(
        self,
        data_key="image",
        p_per_sample=1,
        p_invert_image=0.05,
        gamma_range=(0.5, 2.0),
        per_channel=True,
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.gamma_range = gamma_range
        self.p_invert_image = p_invert_image
        self.per_channel = per_channel
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(p_invert_image):
        # No parameters to retrieve
        do_invert = False
        if np.random.uniform() < p_invert_image:
            do_invert = True
        return do_invert

    def __gamma__(self, image, gamma_range, invert_image, per_channel):
        return augment_gamma(
            image,
            gamma_range,
            invert_image,
            per_channel,
            clip_to_input_range=self.clip_to_input_range,
        )

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                do_invert = self.get_params(self.p_invert_image)
                data_dict[self.data_key][b] = self.__gamma__(
                    data_dict[self.data_key][b],
                    self.gamma_range,
                    do_invert,
                    per_channel=self.per_channel,
                )
        return data_dict
