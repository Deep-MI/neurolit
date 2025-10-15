from typing import List, Tuple, Dict, Any
import monai
from monai import transforms
import torch

class Subsampled(transforms.MapTransform):
    """Transform that subsamples input data and pads to a specified size.
    
    Args:
        keys (list): List of keys to apply transform to
        spatial_size (tuple): Target spatial size for output (h,w,d)
        size_reduction (int, optional): Factor by which to subsample. Defaults to 2.
    """

    def __init__(self, keys: List[str], spatial_size: Tuple[int, int, int], size_reduction: int = 2) -> None:
        super().__init__(keys)
        self.make_meta_tensor = lambda x,y : monai.data.MetaTensor(x,meta=y)
        self.image_shape = spatial_size
        self.size_reduction = size_reduction

    def _pad_image(self, img: torch.Tensor, max_out: Tuple[int, int, int]) -> torch.Tensor:
        # Get correct size = max along shape
        assert(len(max_out) == 3), 'max_out must be 3D'
        c, h, w, d = img.shape
        #LOGGER.info("Padding image from {0} to {1}x{1}x{1}".format(img.shape, max_out))
        padded_img = torch.zeros((c, max_out[0], max_out[1], max_out[2]), dtype=img.dtype)
        
        padded_img[:, 0: h, 0: w, 0:d] = img
        padded_img = self.make_meta_tensor(padded_img, img.meta)
        return padded_img

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        #assert(data.shape[1:]//2 <= self.image_shape), 'image_shape must be at least half of data.shape[1:]'
        for key in data.keys():
            if key in self.keys:
                img = data[key][:,::self.size_reduction,::self.size_reduction,::self.size_reduction]
                # pad from top left to match IMAGE_SHAPE
                data[key] = self._pad_image(img, self.image_shape)

                try:
                    #data[key].meta['spatial_shape'] = data[key].shape[1:]
                    data[key].meta['delta'] = data[key].meta['delta'] * self.size_reduction
                except KeyError:
                    pass

        return data
    
class ScaleAugmentation(transforms.MapTransform):
    """Transform that randomly scales the voxel size metadata.
    
    Args:
        keys (list): List of keys to apply transform to
        scale_range (tuple): Range of possible scale factors (min, max)
    """

    def __init__(self, keys: List[str], scale_range: Tuple[float, float]) -> None:
        super().__init__(keys)
        self.scale_range = scale_range

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in data.keys():
            if key in self.keys:
                s = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
                data[key].meta['delta'] = data[key].meta['delta'] * s
        return data
    
# create do nothing transform
class Identityd(transforms.MapTransform):
    """Transform that returns data unchanged.
    
    Used as a placeholder when no transform is needed.
    """
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
