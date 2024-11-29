import numpy as np
import torch
import os
import logging
from typing import Union, Literal, Optional
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, isfile
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch


class YuccaTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = None,
        composed_transforms=None,
        task_type: Literal["classification", "segmentation", "self-supervised", "contrastive"] = "segmentation",
        allow_missing_modalities=False,
        p_oversample_foreground=0.33,
    ):
        self.all_cases = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.task_type = task_type
        assert task_type in ["classification", "segmentation", "self-supervised", "contrastive"]
        self.supervised = self.task_type in ["classification", "segmentation"]
        self.label_dtype = label_dtype
        self.allow_missing_modalities = allow_missing_modalities
        self.already_loaded_cases = {}

        # for segmentation and classification we override the default None
        # because arrays are saved as floats and we want them to be ints.
        if self.label_dtype is None:
            if self.supervised:
                self.label_dtype = torch.int32

        self.croppad = CropPad(patch_size=self.patch_size, p_oversample_foreground=p_oversample_foreground)
        self.to_torch = NumpyToTorch(label_dtype=self.label_dtype)

        self._keep_in_ram = keep_in_ram

    @property
    def keep_in_ram(self):
        if self._keep_in_ram is not None:
            return self._keep_in_ram
        if len(self.all_cases) < 50:
            self._keep_in_ram = True
        else:
            logging.debug("Large dataset detected. Will not keep cases in RAM during training.")
            self._keep_in_ram = False
        return self._keep_in_ram

    def load_and_maybe_keep_pickle(self, path):
        path = path + ".pkl"
        if not self.keep_in_ram:
            return load_pickle(path)
        if path in self.already_loaded_cases:
            return self.already_loaded_cases[path]
        self.already_loaded_cases[path] = load_pickle(path)
        return self.already_loaded_cases[path]

    def load_and_maybe_keep_volume(self, path):
        path = path + ".npy"
        if not self.keep_in_ram:
            if isfile(path):
                try:
                    return np.load(path, "r")
                except ValueError:
                    return np.load(path, allow_pickle=True)
            else:
                print("uncompressed data was not found.")

        if isfile(path):
            if path in self.already_loaded_cases:
                return self.already_loaded_cases[path]
            try:
                self.already_loaded_cases[path] = np.load(path, "r")
            except ValueError:
                self.already_loaded_cases[path] = np.load(path, allow_pickle=True)
        else:
            print("uncompressed data was not found.")

        return self.already_loaded_cases[path]

    def __len__(self):
        return len(self.all_cases)

    def __getitem__(self, idx):
        # remove extension if file splits include extensions
        case, _ = os.path.splitext(self.all_cases[idx])
        data = self.load_and_maybe_keep_volume(case)
        metadata = self.load_and_maybe_keep_pickle(case)

        if self.allow_missing_modalities:
            image, label = self.unpack_with_zeros(data, supervised=self.supervised)
        else:
            image, label = self.unpack(data, supervised=self.supervised)

        data_dict = {"file_path": case}  # metadata that can be very useful for debugging.
        if self.task_type in ["classification", "segmentation"]:
            data_dict.update({"image": image, "label": label})
        elif self.task_type == "self-supervised":
            data_dict.update({"image": image})
        elif self.task_type == "contrastive":
            view1 = self._transform({"image": image}, case)["image"]
            view2 = self._transform({"image": image}, case)["image"]
            data_dict.update({"image": (view1, view2)})
            return data_dict
        else:
            logging.error(f"Task Type not recognized. Found {self.task_type}")
        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata):
        data_dict = self.croppad(data_dict, metadata)
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return self.to_torch(data_dict)

    def unpack(self, data, supervised: bool):
        if supervised:
            return data[:-1], data[-1:]
        return data, None

    def unpack_with_zeros(self, data, supervised: bool):
        assert data.dtype == "object", "allow missing modalities is true but dtype is not object"

        # First find the array with the largest array.
        # in classification this avoids setting the zero array to the 1d array with classes
        sizes = [i.size for i in data]
        idx_largest_array = np.where(sizes == np.max(sizes))[0][0]

        # replace missing modalities with zero-filed arrays
        for idx, i in enumerate(data):
            if i.size == 0:
                data[idx] = np.zeros(data[idx_largest_array].squeeze().shape)

        # unpack array into images and (maybe) labels
        if supervised:
            images = np.array([mod for mod in data[:-1]])
            label = data[-1:][0]
        else:
            images = np.array([mod for mod in data])
            label = None
        return images, label


class YuccaTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_data_dir: str,
        pred_save_dir: str,
        overwrite_predictions: bool = False,
        suffix="nii.gz",
        pred_include_cases: list = None,
    ):
        self.data_path = raw_data_dir
        self.pred_save_dir = pred_save_dir
        self.overwrite = overwrite_predictions
        self.suffix = suffix
        self.pred_include_cases = pred_include_cases
        self.unique_cases = np.unique(
            [i[: -len("_000." + suffix)] for i in subfiles(self.data_path, suffix=self.suffix, join=False)]
        )
        assert len(self.unique_cases) > 0, f"No cases found in {self.data_path}. Looking for files with suffix: {self.suffix}"

        self.cases_already_predicted = np.unique(
            [i[: -len("." + suffix)] for i in subfiles(self.pred_save_dir, suffix=self.suffix, join=False)]
        )
        logging.info(f"Found {len(self.cases_already_predicted)} already predicted cases. Overwrite: {self.overwrite}")
        if not self.overwrite:
            self.unique_cases = [i for i in self.unique_cases if i not in self.cases_already_predicted]
        if self.pred_include_cases is not None:
            self.unique_cases = [i for i in self.unique_cases if i in self.pred_include_cases]

    def __len__(self):
        return len(self.unique_cases)

    def __getitem__(self, idx):
        # Here we generate the paths to the cases along with their ID which they will be saved as.
        # we pass "case" as a list of strings and case_id as a string to the dataloader which
        # will convert them to a list of tuples of strings and a tuple of a string.
        # i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]
        case_id = self.unique_cases[idx]
        image_paths = [
            impath
            for impath in subfiles(self.data_path, suffix=self.suffix)
            if os.path.split(impath)[-1][: -len("_000." + self.suffix)] == case_id
        ]
        return {"data_paths": image_paths, "extension": self.suffix, "case_id": case_id}


class YuccaTestPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        preprocessed_data_dir: str,
        pred_save_dir: str,
        overwrite_predictions: bool = False,
        suffix: str = "pt",  # noqa U100
        pred_include_cases: list = None,
    ):
        self.data_path = preprocessed_data_dir
        self.pred_save_dir = pred_save_dir
        self.overwrite = overwrite_predictions
        self.data_suffix = "." + suffix
        self.prediction_suffix = ".nii.gz"
        self.pred_include_cases = pred_include_cases
        self.unique_cases = np.unique(
            [i[: -len(self.data_suffix)] for i in subfiles(self.data_path, suffix=self.data_suffix, join=False)]
        )
        assert (
            len(self.unique_cases) > 0
        ), f"No cases found in {self.data_path}. Looking for files with suffix: {self.data_suffix}"

        self.cases_already_predicted = np.unique(
            [
                i[: -len(self.prediction_suffix)]
                for i in subfiles(self.pred_save_dir, suffix=self.prediction_suffix, join=False)
            ]
        )
        logging.info(f"Found {len(self.cases_already_predicted)} already predicted cases. Overwrite: {self.overwrite}")
        if not self.overwrite:
            self.unique_cases = [i for i in self.unique_cases if i not in self.cases_already_predicted]
        if self.pred_include_cases is not None:
            self.unique_cases = [i for i in self.unique_cases if i in self.pred_include_cases]

    def __len__(self):
        return len(self.unique_cases)

    def __getitem__(self, idx):
        # Here we generate the paths to the cases along with their ID which they will be saved as.
        # we pass "case" as a list of strings and case_id as a string to the dataloader which
        # will convert them to a list of tuples of strings and a tuple of a string.
        # i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]
        case_id = self.unique_cases[idx]
        path = os.path.join(self.data_path, case_id + self.data_suffix)

        if self.data_suffix == ".pt":
            data = torch.load(path, weights_only=False)
        elif self.data_suffix == ".npy":
            data = torch.tensor(np.load(path, allow_pickle=True))

        data_properties = load_pickle(os.path.join(self.data_path, case_id + ".pkl"))
        return {"data": data, "data_properties": data_properties, "case_id": case_id}


if __name__ == "__main__":
    import torch
    from yucca.paths import get_preprocessed_data_path
    from batchgenerators.utilities.file_and_folder_operations import join
    from yucca.modules.data.samplers import InfiniteRandomSampler

    files = subfiles(join(get_preprocessed_data_path(), "Task001_OASIS/YuccaPlanner"), suffix="npy")
    ds = YuccaTrainDataset(files, patch_size=(12, 12, 12))
    sampler = InfiniteRandomSampler(ds)
    dl = torch.utils.data.DataLoader(ds, num_workers=2, batch_size=2, sampler=sampler)
