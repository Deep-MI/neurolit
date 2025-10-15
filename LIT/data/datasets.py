import os

from torch.utils.data import Dataset
import numpy as np

from LIT.data import conform

from monai.data import CacheDataset
from monai import transforms
import nibabel as nib


def get_test_sample():
    root_idr = '/groups/ag-reuter/projects/fastsurfer-tumor/fastsurfer-tumor-pipeline/tumor_validation_data/cleaned_hires_dataset/data/Rhineland'
    filepath_image = root_idr + '/0078875e-c7b8-4d60-b055-c0d3a6a693bb/orig_deformed_masked.nii.gz'
    filepath_mask = root_idr + '/0078875e-c7b8-4d60-b055-c0d3a6a693bb/tumor_mask_conformed.mgz'

    nib_image = nib.load(filepath_image)
    nib_mask = nib.load(filepath_mask)

    assert(conform.is_conform(nib_image, conform_vox_size='min', check_dtype=False, verbose=False)), "Image is not conform"
    assert(conform.is_conform(nib_mask, conform_vox_size='min', check_dtype=False, verbose=False)), "Mask is not conform"

    sample = {
        'image': nib_image.get_fdata(),
        'mask': nib_mask.get_fdata(),
        'zooms': float(nib_image.header.get_zooms()[0]) # conformed image must be isotropic
    }

    test_tr = transforms.Compose([
        transforms.AddChanneld(keys=['image', 'mask']),
        transforms.ScaleIntensityd(keys=['image'])
    ])

    sample = test_tr(sample)

    return sample


def get_dataset(csv_file, transforms=None, size="standard"):
    """Get a dataset from a CSV file.
    
    Args:
        csv_file (str): Path to CSV file containing file paths
        transforms (callable, optional): Transforms to apply to the data. Defaults to None.
        size (str, optional): If "small", only loads first 3 samples. Defaults to "standard".
    
    Returns:
        CacheDataset: Dataset containing the loaded files
    """
    with open(csv_file, "r") as f:
        if size == 'small':
            files = [{"image":os.path.join(line.rstrip(), 'mri/orig.mgz')} for line in f.readlines()[:3]]
        else:
            files = [{"image":os.path.join(line.rstrip(), 'mri/orig.mgz')} for line in f.readlines()]
    
    dataset = CacheDataset(data=files, transform=transforms, cache_rate=1.0, num_workers=4)
    return dataset

def get_base_dataset(size="big", transforms=None):
    """Get training and validation datasets from CSV files.
    
    Args:
        size (str, optional): Size of training dataset to use. Must be one of ["big", "small", "standard"]. 
            "big" uses 1268 subjects, "small" uses 120 subjects. Defaults to "big".
        transforms (callable, optional): Transforms to apply to the data. Defaults to None.

    Returns:
        tuple: Tuple containing:
            - train_dataset (CacheDataset): Training dataset
            - val_dataset (CacheDataset): Validation dataset
    """
    assert(size in ["big", "small", "standard"])

    data_file_dir = os.getcwd()
    if size == 'big':
        train_csv = os.path.join(data_file_dir, "data_csvs/trainingset_1268_subjects_1mm_fastsurfer_hires_hcp_rh.txt")
    else:
        train_csv = os.path.join(data_file_dir, "data_csvs/trainingset_120_subjects_1mm_fastsurfer_hires_hcp_rh.txt")
    
    val_csv = os.path.join(data_file_dir, "data_csvs/validationset_80_subjects_1mm_fastsurfer_hires_hcp_rh.txt")

    train_dataset = get_dataset(train_csv, transforms=transforms, size=size)
    val_dataset = get_dataset(val_csv, transforms=transforms, size=size)

    return train_dataset, val_dataset

class SlicedDataset(Dataset):
    """A dataset that extracts slices from 3D images with a specified thickness.

    This dataset wraps another dataset containing 3D images and provides access to 2D slices
    with a configurable thickness along a specified axis. Each slice is returned with the
    slice thickness as the channel dimension.

    Args:
        dataset (Dataset): Base dataset containing 3D images
        thickness (int): Thickness of slices to extract (must be odd)
        ax (int): Axis along which to extract slices (0=sagittal, 1=coronal, 2=axial)
        slice_per_img (int, optional): Number of slices to extract per image. If None,
            extracts all possible slices. Defaults to None.
        transform (callable, optional): Transform to apply to extracted slices.
            Defaults to None.

    Attributes:
        dataset (Dataset): The base dataset
        thickness (int): Slice thickness
        ax (int): Slicing axis
        slice_per_img (list): Number of slices per image
        transform (callable): Transform function
        slice_per_img_cumsum (ndarray): Cumulative sum of slices per image
        len (int): Total number of slices across all images
    """

    def __init__(self, dataset, thickness, ax, slice_per_img=None, transform=None):
        assert(thickness % 2 == 1), "Thickness must be odd"
        self.dataset = dataset
        self.thickness = thickness
        self.ax = ax
        if slice_per_img is None:
            self.slice_per_img = [sample['image'].shape[ax+1] - thickness for sample in self.dataset]
        else:
            self.slice_per_img = [slice_per_img] * len(self.dataset)
        self.transform = transform

        self.slice_per_img_cumsum = np.cumsum(self.slice_per_img)
        self.len = np.sum(self.slice_per_img)
    
    def get_slice_axis(self, slice_index):
        """Get slice indices for extracting a slice at the given index.

        Args:
            slice_index (int): Index of slice to extract

        Returns:
            tuple: Tuple of slice objects for indexing the image array
        """
        slice_axis = [slice(None)] * (self.ax+2)
        slice_axis[self.ax+1] = slice(slice_index, slice_index + self.thickness)
        return tuple(slice_axis)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """Get a slice from the dataset at the specified index.

        Maps the flat index to an image and slice index, extracts the slice,
        and ensures the slice thickness becomes the channel dimension.

        Args:
            index (int): Index of slice to retrieve

        Returns:
            dict: Dictionary containing the slice under 'image' key and any additional
                metadata from the base dataset
        """
        img_index = np.searchsorted(self.slice_per_img_cumsum, index)
        slice_index = index - self.slice_per_img_cumsum[img_index-1] if img_index > 0 else index
        img = self.dataset[img_index]['image']
        img = img[self.get_slice_axis(slice_index)]
        # swap axis to put slice axis first
        img = img.swapaxes(1, self.ax+1)
        img = img.squeeze(0)
        if self.transform is not None:
            img = self.transform(img)

        return_dict = {
            'image': img,
            **{k: v for k, v in self.dataset[img_index].items() if k != 'image'}
        }

        return return_dict
