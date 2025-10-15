# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import sys
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
# Suppress FutureWarning about torch.cuda.amp.autocast deprecation (comes from monai-generative)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='`torch.cuda.amp.autocast\\(args...\\)` is deprecated')


# supress warning on loading matplotlib
import matplotlib
from monai import transforms
import numpy as np
from numpy.typing import NDArray
import nibabel as nib
import nibabel.processing
import torch
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from torch.amp import autocast # previous: from torch.cuda.amp import autocast
import torch.nn.functional as F
from LIT.networks.DiffusionUnet import DiffusionModelUNetVINN

from LIT.data import conform
from LIT.utils.plotting import plot_batch, plot_inpainting
from LIT.inference import *



# use Agg backend on server
if os.environ.get('DISPLAY','') == '':
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.makedirs('/tmp/', exist_ok=True)
    os.environ['MPLCONFIGDIR'] = '/tmp'
    matplotlib.use('Agg')


# Custom types
PathLike = Union[str, Path]
NiftiImage = nib.Nifti1Image
ModelDict = Dict[str, torch.nn.Module]
VolumeSlice = Tuple[Union[int, slice], ...]
AffineMatrix = NDArray[np.float64]


def dilate_mask(mask: torch.Tensor, num_iterations: int, kernel_size: int = 3) -> torch.Tensor:
    """Dilate a binary mask multiple times using max pooling.

    Args:
        mask: Input binary mask tensor
        num_iterations: Number of times to apply dilation
        kernel_size: Size of the dilation kernel (must be odd), defaults to 3

    Returns:
        Dilated mask tensor of same shape as input
    """
    if kernel_size % 2 != 1:
        raise ValueError("Kernel size must be odd")

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)

    # Add batch and channel dimensions if needed
    orig_shape = mask.shape
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)

    # Perform dilation multiple times
    dilated = mask
    padding = kernel_size // 2
    for _ in range(num_iterations):
        dilated = F.max_pool3d(
            dilated,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

    # Restore original shape
    if len(orig_shape) == 3:
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated




def conform_nifti(image: NiftiImage) -> NiftiImage:
    """Conform NIfTI image to standard orientation and voxel size."""
    if len(image.shape) > 3 and image.shape[3] != 1:
        raise ValueError(f"Multiple input frames ({image.shape[3]}) not supported!")

    _vox_size: str = "min"
    try:
        if conform.is_conform(image, conform_vox_size=_vox_size, verbose=False):
            return image
        return conform.conform(image, order=2, conform_vox_size=_vox_size)
    except ValueError as e:
        raise ValueError(e.args[0])

def get_slice_from_volume(
    volume: torch.Tensor,
    slice_dim: int,
    slice_cut: int,
    thickness: int
) -> torch.Tensor:
    """Extract a slice from a volume with given thickness."""
    threed_to_twod_slice: List[slice] = [slice(None)] * 3
    threed_to_twod_slice[slice_dim] = slice(slice_cut - thickness//2, slice_cut + thickness//2 + 1)
    return volume[tuple(threed_to_twod_slice)]

def inpaint_volume(
    models: ModelDict,
    val_image: torch.Tensor,
    mask: torch.Tensor,
    val_image_masked: torch.Tensor,
    scale_factor: Optional[float] = None,
    out_dir: Optional[PathLike] = None,
    slice_dim: Optional[int] = None,
    slice_input: bool = True,
    SAVE_VOLUMES: bool = True,
    SAVE_IMAGES: bool = True,
    device: str = 'cuda',
    DDIM: bool = False,
    val_image_nib: Optional[NiftiImage] = None
) -> torch.Tensor:
    """Inpaints a volume using trained diffusion models.

    Args:
        models: Dictionary mapping view names to model instances
        val_image: Input image tensor of shape (B, C, H, W, D)
        mask: Binary mask tensor of same shape as val_image
        val_image_masked: Masked input image tensor
        scale_factor: Optional scaling factor for the output
        out_dir: Directory to save outputs
        slice_dim: Dimension to slice along for 2D models
        slice_input: Whether to process input as slices
        SAVE_VOLUMES: Whether to save intermediate volumes
        SAVE_IMAGES: Whether to save intermediate images
        device: Device to run inference on
        DDIM: Whether to use DDIM sampling
        val_image_nib: Original NIfTI image for header/affine info

    Returns:
        Inpainted image tensor of same shape as input
    """

    # Input validation with type checking
    if not isinstance(models, dict):
        raise TypeError("models must be a dictionary")
    if not isinstance(val_image, torch.Tensor):
        raise TypeError("val_image must be a torch.Tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")

    # Validate inputs
    if not (mask > 0).any() or not (mask == 0).any():
        raise ValueError("Mask must have both zero and non-zero values")

    test_model = next(iter(models.values()))
    is_2d_model = test_model.conv_in.spatial_dims == 2

    # Set up volume slicing
    volume_only_slice = (0, slice(None), slice(None), slice(None)) if is_2d_model else (0, 0, slice(None), slice(None), slice(None))

    if not slice_input and is_2d_model:
        if slice_dim not in [0, 1, 2]:
            raise ValueError("slice_dim must be 0, 1 or 2 for 2D models with slice_input=False")

    slice_dim = slice_dim or 0  # Default to first dimension


    # Current mean calculation could fail with empty mask
    mask_indices = torch.where(mask[volume_only_slice].bool())
    if not mask_indices[0].numel():
        raise ValueError("No valid mask indices found")
    SLICE_CUT = torch.mean(torch.stack(mask_indices), dtype=torch.float32, dim=1).int()

    # Save intermediate results
    if SAVE_VOLUMES or SAVE_IMAGES:
        affine_header = (val_image_nib.affine, val_image_nib.header) if val_image_nib else (np.eye(4), None)

        if SAVE_VOLUMES:
            os.makedirs(os.path.join(out_dir, 'inpainting_volumes'), exist_ok=True)
            for name, data in [('original_image', val_image), ('mask', mask), ('masked_image', val_image_masked)]:
                nib.save(nib.Nifti1Image(data[volume_only_slice].cpu().numpy(), *affine_header),
                        os.path.join(out_dir, f'inpainting_volumes/inpainting_{name}.nii.gz'))

        if SAVE_IMAGES:
            os.makedirs(os.path.join(out_dir, 'inpainting_images'), exist_ok=True)
            for name, data in [('original_image', val_image), ('mask', mask), ('masked_image', val_image_masked)]:
                plot_batch(data, os.path.join(out_dir, f'inpainting_images/inpainting_{name}.png'), 
                          slice_cut=SLICE_CUT)

    # Setup models and scheduler
    for model in models.values():
        model.eval()
    

    if DDIM:
        steps = 10
        scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195, clip_sample=False)
        print('Using DDIM scheduler with', steps, 'steps')
    else:
        steps = 1000
        scheduler = DDPMScheduler(num_train_timesteps=steps, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=steps, device=device)

    # Prepare inputs
    mask = (mask > 0).to(device)
    val_image = val_image.to(device)
    val_image_masked = val_image_masked.to(device)

    # Run inpainting
    print('Using 2.5D inpainting with view aggregation')
    Inpainter = OffsetTwoAndHalfDInpaintingInferer(
        inference_steps=steps,
        scheduler=scheduler,
        diffusion_model_dict=models
    )

    #import pdb; pdb.set_trace()
    with torch.inference_mode(), autocast(enabled=True, device_type='cuda'):
        val_image_inpainted = Inpainter(
            mask=mask[0],
            image_masked=val_image_masked[0],
            num_resample_steps=10,
            num_resample_jumps=15,
            batch_size=8,
            get_intermediates=False,
            scale_factor=scale_factor
        )
    val_image_inpainted = val_image_inpainted.unsqueeze(0)

    # Save results
    if SAVE_IMAGES:
        plot_inpainting(val_image, val_image_masked, val_image_inpainted,
                       out_file=os.path.join(out_dir, 'inpainting_images/inpainting_result.png'),
                       SLICE_CUT=SLICE_CUT, cut_dim=0)


        # ##### plotting of intermediates
        # if len(models) == 3: # view agg gives 3d intermediates
        #     slice_c = SLICE_CUT
        # elif slice_dim == 0:
        #     slice_c = SLICE_CUT[[1,2]]
        # elif slice_dim == 1:
        #     slice_c = SLICE_CUT[[0,2]]
        # elif slice_dim == 2:
        #     slice_c = SLICE_CUT[[0,1]]
        # plot_batch(intermediates, os.path.join(out_dir,'inpainting_images/inpainting_intermediates.png'), 
        #            slice_cut=slice_c)

    if SAVE_VOLUMES:
        nib.save(nib.Nifti1Image(val_image_inpainted[volume_only_slice].cpu().numpy() * 255, *affine_header),
                 os.path.join(out_dir, 'inpainting_volumes/inpainting_result.nii.gz'))
        print('Saved inpainting result as inpainting_volumes/inpainting_result.nii.gz')

    print('Finished inpainting')
    return val_image_inpainted


def main():
    SAVE_VOLUMES = True
    SAVE_IMAGES = True

    parser = argparse.ArgumentParser(description='Train a 3D DDPM model')
    parser.add_argument('-o','--out_dir', type=str, default='debug_run', help='experiment output directory')
    parser.add_argument('-i', '--input_image', type=str, help='input image', required=True)
    parser.add_argument('-m', '--mask_image', type=str, help='input mask', default=None, required=False)
    parser.add_argument('--dilate', type=int, help='number of pixels to dilate the mask by',
                        required=False, default=0)
    parser.add_argument('-c_coronal', '--checkpoint_coronal',
                        type=str, help='checkpoint to load for inference in coronal plane',
                        default=None, required=False)
    parser.add_argument('-c_axial', '--checkpoint_axial', 
                        type=str, help='checkpoint to load for inference in axial plane', 
                        default=None, required=False)
    parser.add_argument('-c_sagittal', '--checkpoint_sagittal',
                        type=str, help='checkpoint to load for inference in sagittal plane',
                        default=None, required=False)

    args = parser.parse_args()


    # load models
    model_state_dicts = {}
    if args.checkpoint_coronal is not None:
        model_state_dicts['coronal'] = torch.load(args.checkpoint_coronal, weights_only=True)
    if args.checkpoint_axial is not None:
        model_state_dicts['axial'] = torch.load(args.checkpoint_axial, weights_only=True)
    if args.checkpoint_sagittal is not None:
        model_state_dicts['sagittal'] = torch.load(args.checkpoint_sagittal, weights_only=True)


    # setup model
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    SLICE_THICKNESS = 7
    # make model out of weights only
    model_dict = {}
    for model_name, model_state_dict in model_state_dicts.items():
        model_dict[model_name] = DiffusionModelUNetVINN(
            spatial_dims=2,
            internal_size=(128,128),
            in_channels=SLICE_THICKNESS,
            out_channels=SLICE_THICKNESS,
            num_channels=[128, 256, 512],#[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            norm_num_groups=4,
            use_fp16_VINN=False,
            is_vinn=True,
            interpolation_mode='bilinear',
        )
        model_dict[model_name].load_state_dict(model_state_dict)
        model_dict[model_name].to(device)


    # Add compilation for PyTorch 2.0+
    #print(f'Torch version: {torch.__version__}')
    # if torch.__version__ >= "2.0.0" and (isinstance(device, torch.device) and device.type == 'cuda'):
    #     print("Compiling models with torch.compile()...")
    #     try:
    #         for name, model in model_dict.items():
    #             print(f"Compiling {name} model...")
    #             model_dict[name] = torch.compile(
    #                 model,
    #                 mode="reduce-overhead",
    #                 fullgraph=True,
    #                 dynamic=False
    #             )
    #                 # options={
    #                 #     "triton.unique_kernel_names": True,
    #                 #     "max_autotune": True,
    #                 #     "layout_optimization": True
    #                 # }
    #             #) # other backend options: 'inductor', 'aot_eager', 'aot_eager_numba'
    #         print("Model compilation complete!")
    #     except Exception as e:
    #         print(f"Warning: Model compilation failed with error: {e}")
    #         print("Continuing with uncompiled models...")

    # setup parameters (i.e. whether to use view aggregation, 2d or 3d model)
    model_to_dim = {'coronal': 2, 'axial': 1, 'sagittal': 0}
    if len(model_dict) == 0:    
        sys.exit("ERROR: At least one checkpoint must be specified", file=sys.stderr)
    elif len(model_dict) == 1:
        DIM = model_to_dim[list(model_dict.keys())[0]]
        VIEW_AGG = False
        IS_2D = list(model_dict.values())[0].conv_in.spatial_dims == 2
    elif len(model_dict) == 3:
        DIM = 0
        VIEW_AGG = True
        IS_2D = True
    else:
        sys.exit(f"ERROR: One or three checkpoints must be specified, but got {len(model_dict)}"
                 , file=sys.stderr)

    assert(list(model_dict.values())[0].is_vinn)

    val_image_nib = nib.load(args.input_image)
    val_image_nib = conform_nifti(val_image_nib)

    val_image = torch.from_numpy(val_image_nib.get_fdata()).float()

    mask_nib = nib.load(args.mask_image)
    # resample mask to image affine
    mask_nib = nibabel.processing.resample_from_to(mask_nib, val_image_nib, order=0, mode='constant', cval=0)

    mask = torch.from_numpy(mask_nib.get_fdata()).float()

    if args.dilate > 0:
        mask = dilate_mask(mask, args.dilate)


    INTERNAL_SHAPE = list(model_dict.values())[0].internal_size
    zooms = val_image_nib.header.get_zooms()
    internal_res_mm = 256 / INTERNAL_SHAPE[0]
    scale_factor = internal_res_mm / zooms[0]

    val_sample = {'image': val_image, 'mask': mask}

    if not os.path.exists(os.path.join(args.out_dir,'inpainting_images')) or not os.path.exists(os.path.join(args.out_dir,'inpainting_volumes')):
        os.makedirs(os.path.join(args.out_dir,'inpainting_images'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir,'inpainting_volumes'), exist_ok=True)
        print('Created output directory:', args.out_dir)
    else:
        print('Output directory already exists:', args.out_dir)


    tr = [
        #transforms.AddChanneld(keys=['image', 'mask']),
        transforms.EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
        transforms.ScaleIntensityd(keys=['image']),
    ]

    data_transform = transforms.Compose(tr)
    val_sample_preproc = data_transform(val_sample)



    assert(val_sample_preproc['image'].shape == val_sample_preproc['mask'].shape), f"Image and mask must have the same shape, but got {val_sample_preproc['image'].shape} and {val_sample_preproc['mask'].shape}"
    val_image = val_sample_preproc['image']
    mask = val_sample_preproc['mask']

    val_image_masked = val_image * (~(mask > 0)).float()

    inpaint_volume(models=model_dict, val_image=val_image, mask=mask, val_image_masked=val_image_masked, scale_factor=scale_factor,
                   out_dir=args.out_dir, SAVE_VOLUMES=SAVE_VOLUMES, SAVE_IMAGES=SAVE_IMAGES,
                   device=device, slice_input=False, slice_dim=DIM, val_image_nib=val_image_nib, DDIM=False)


if __name__ == "__main__":
    main()
