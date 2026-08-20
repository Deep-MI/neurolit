# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress FutureWarning about deprecated cuda.cudart module
# Must be before torch/monai imports to catch the warning during import
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cuda.cudart.*")  # noqa: E402

# suppress warning on loading matplotlib
import matplotlib  # noqa: E402
import nibabel as nib  # noqa: E402
import nibabel.processing  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from monai import transforms  # noqa: E402
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from torch.amp import autocast  # noqa: E402  # previous: from torch.cuda.amp import autocast

from neurolit.data import conform  # noqa: E402
from neurolit.inference import OffsetTwoAndHalfDInpaintingInferer  # noqa: E402
from neurolit.networks.DiffusionUnet import DiffusionModelUNetVINN  # noqa: E402
from neurolit.utils.geometry_policy import (  # noqa: E402
    default_conform_kwargs,
    vinn_scale_factor_from_zooms,
)
from neurolit.utils.inference_io import (  # noqa: E402
    compute_model_min_size,
    crop_after_inference,
    pad_for_inference,
    resample_result_to_reference,
)
from neurolit.utils.log import get_logger  # noqa: E402
from neurolit.utils.plotting import plot_batch, plot_inpainting  # noqa: E402

logger = get_logger(__name__)

# use Agg backend on server
if os.environ.get("DISPLAY", "") == "":
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.makedirs("/tmp/", exist_ok=True)
    os.environ["MPLCONFIGDIR"] = "/tmp"
    matplotlib.use("Agg")


# Custom types
PathLike = str | Path
NiftiImage = nib.Nifti1Image
ModelDict = dict[str, torch.nn.Module]
VolumeSlice = tuple[int | slice, ...]
AffineMatrix = NDArray[np.float64]


def positive_int(value: str) -> int:
    """Parse a positive integer for a command-line argument."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def resolve_inference_device(device: str) -> torch.device:
    """Resolve the requested inference device.

    Parameters
    ----------
    device : str
        Requested device name: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns
    -------
    torch.device
        Resolved torch device.
    """
    normalized_device = device.lower()
    if normalized_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference, but no CUDA device is available.")
        return torch.device("cuda")
    if normalized_device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device}'. Expected one of: auto, cpu, cuda.")


def dilate_mask(mask: torch.Tensor, num_iterations: int, kernel_size: int = 3) -> torch.Tensor:
    """Dilate a binary mask using repeated max pooling.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor to dilate.
    num_iterations : int
        Number of dilation steps to apply.
    kernel_size : int, optional
        Size of the dilation kernel (must be odd), by default 3.

    Returns
    -------
    torch.Tensor
        Dilated mask tensor of the same shape as the input.
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
        dilated = F.max_pool3d(dilated, kernel_size=kernel_size, stride=1, padding=padding)

    # Restore original shape
    if len(orig_shape) == 3:
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated


def conform_nifti(image: NiftiImage, *, min_auto_img_size: int | None = None) -> NiftiImage:
    """Conform a NIfTI image to the repository orientation/voxel standard.

    Parameters
    ----------
    image : NiftiImage
        Input image that should be conformed.
    min_auto_img_size : int, optional
        Minimum side length for automatic conforming. By default, the input
        field of view determines the target size.

    Returns
    -------
    NiftiImage
        Conformed image with standardized affine/voxel size.
    """
    if len(image.shape) > 3 and image.shape[3] != 1:
        raise ValueError(f"Multiple input frames ({image.shape[3]}) not supported!")

    conform_kwargs = default_conform_kwargs(min_auto_img_size=min_auto_img_size)
    is_conform_kwargs = {
        "vox_size": conform_kwargs["vox_size"],
        "img_size": conform_kwargs["img_size"],
        "orientation": conform_kwargs["orientation"],
        "threshold_1mm": conform_kwargs["threshold_1mm"],
        "min_auto_img_size": conform_kwargs["min_auto_img_size"],
        "verbose": False,
    }
    if conform_kwargs["dtype"] is not None:
        is_conform_kwargs["dtype"] = conform_kwargs["dtype"]

    try:
        if conform.is_conform(image, **is_conform_kwargs):
            return image
        return conform.conform(
            image,
            order=2,
            vox_size=conform_kwargs["vox_size"],
            img_size=conform_kwargs["img_size"],
            orientation=conform_kwargs["orientation"],
            dtype=conform_kwargs["dtype"],
            rescale=conform_kwargs["rescale"],
            threshold_1mm=conform_kwargs["threshold_1mm"],
            min_auto_img_size=conform_kwargs["min_auto_img_size"],
        )
    except ValueError as e:
        raise ValueError(e.args[0]) from e


def get_slice_from_volume(volume: torch.Tensor, slice_dim: int, slice_cut: int, thickness: int) -> torch.Tensor:
    """Extract a slice from a volume with a specified thickness.

    Parameters
    ----------
    volume : torch.Tensor
        Tensor representing the volume to slice.
    slice_dim : int
        Dimension to slice along.
    slice_cut : int
        Index at the center of the slice.
    thickness : int
        Total thickness of the slice (number of voxels).

    Returns
    -------
    torch.Tensor
        Extracted slice tensor.
    """
    threed_to_twod_slice: list[slice] = [slice(None)] * 3
    threed_to_twod_slice[slice_dim] = slice(slice_cut - thickness // 2, slice_cut + thickness // 2 + 1)
    return volume[tuple(threed_to_twod_slice)]


def inpaint_volume(
    models: ModelDict,
    val_image: torch.Tensor,
    mask: torch.Tensor,
    val_image_masked: torch.Tensor,
    scale_factor: float | None = None,
    out_dir: PathLike | None = None,
    slice_dim: int | None = None,
    slice_input: bool = True,
    SAVE_VOLUMES: bool = True,
    SAVE_IMAGES: bool = True,
    device: torch.device | str = "cuda",
    DDIM: bool = False,
    val_image_nib: NiftiImage | None = None,
    reference_image_nib: NiftiImage | None = None,
    pad_multiple: int = 16,
    num_inference_steps: int = 1000,
    batch_size: int = 8,
) -> torch.Tensor:
    """Inpaint a volume using the trained diffusion models.

    Parameters
    ----------
    models : ModelDict
        Dictionary mapping view names to model instances.
    val_image : torch.Tensor
        Input image tensor (B, C, H, W, D).
    mask : torch.Tensor
        Binary mask tensor of the same shape as ``val_image``.
    val_image_masked : torch.Tensor
        Masked version of the input image.
    scale_factor : Optional[float], optional
        Scaling factor applied during inference, by default ``None``.
    out_dir : Optional[PathLike], optional
        Directory to save outputs, by default ``None``.
    slice_dim : Optional[int], optional
        Dimensionality slice direction for 2D models, by default ``None``.
    slice_input : bool, optional
        Whether to slice the input volume, by default ``True``.
    SAVE_VOLUMES : bool, optional
        Whether to persist intermediate volumes, by default ``True``.
    SAVE_IMAGES : bool, optional
        Whether to persist intermediate images, by default ``True``.
    device : str, optional
        Device identifier (e.g., ``"cuda"``), by default ``"cuda"``.
    DDIM : bool, optional
        Whether to use DDIM sampling instead of DDPM, by default ``False``.
    val_image_nib : Optional[NiftiImage], optional
        Original NIfTI image used for metadata, by default ``None``.

    Returns
    -------
    torch.Tensor
        Inpainted volume with the same shape as the input.
    """
    if isinstance(device, str):
        device = torch.device(device)

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
            os.makedirs(os.path.join(out_dir, "inpainting_volumes"), exist_ok=True)
            for name, data in [("original_image", val_image), ("mask", mask), ("masked_image", val_image_masked)]:
                nib.save(
                    nib.Nifti1Image(data[volume_only_slice].cpu().numpy(), *affine_header),
                    os.path.join(out_dir, f"inpainting_volumes/inpainting_{name}.nii.gz"),
                )

        if SAVE_IMAGES:
            os.makedirs(os.path.join(out_dir, "inpainting_images"), exist_ok=True)
            for name, data in [("original_image", val_image), ("mask", mask), ("masked_image", val_image_masked)]:
                plot_batch(data, os.path.join(out_dir, f"inpainting_images/inpainting_{name}.png"), slice_cut=SLICE_CUT)

    # Setup models and scheduler
    for model in models.values():
        model.eval()

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be > 0")

    if DDIM:
        steps = num_inference_steps
        scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195, clip_sample=False)
        logger.info(f"Using DDIM scheduler with {steps} steps")
    else:
        steps = num_inference_steps
        scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=steps, device=device)

    # Prepare inputs
    mask = (mask > 0).to(device)
    val_image = val_image.to(device)
    val_image_masked = val_image_masked.to(device)

    model_min_size = compute_model_min_size(
        getattr(test_model, "internal_size", None),
        multiple=pad_multiple,
        strict_larger=True,
    )

    # Deterministic inference-only padding for unsupported spatial dimensions.
    val_image, mask, val_image_masked, pad_metadata = pad_for_inference(
        val_image,
        mask,
        val_image_masked,
        multiple=pad_multiple,
        min_size=model_min_size,
    )

    # Run inpainting
    logger.info("Using 2.5D inpainting with view aggregation")
    Inpainter = OffsetTwoAndHalfDInpaintingInferer(inference_steps=steps, scheduler=scheduler, diffusion_model_dict=models)

    # import pdb; pdb.set_trace()
    with torch.inference_mode(), autocast(device_type=device.type, enabled=device.type == "cuda"):
        val_image_inpainted = Inpainter(
            mask=mask[0],
            image_masked=val_image_masked[0],
            num_resample_steps=10,
            num_resample_jumps=15,
            batch_size=batch_size,
            get_intermediates=False,
            scale_factor=scale_factor,
        )
    val_image_inpainted = val_image_inpainted.unsqueeze(0)
    val_image_inpainted = crop_after_inference(val_image_inpainted, pad_metadata)
    val_image = crop_after_inference(val_image, pad_metadata)
    val_image_masked = crop_after_inference(val_image_masked, pad_metadata)

    # Save results
    if SAVE_IMAGES:
        plot_inpainting(
            val_image,
            val_image_masked,
            val_image_inpainted,
            out_file=os.path.join(out_dir, "inpainting_images/inpainting_result.png"),
            SLICE_CUT=SLICE_CUT,
            cut_dim=0,
        )

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
        output_data = val_image_inpainted[volume_only_slice].cpu().numpy() * 255
        if reference_image_nib is not None and val_image_nib is not None:
            output_nib = resample_result_to_reference(output_data, val_image_nib, reference_image_nib, order=1)
        else:
            output_nib = nib.Nifti1Image(output_data, *affine_header)
        nib.save(output_nib, os.path.join(out_dir, "inpainting_volumes/inpainting_result.nii.gz"))
        logger.info("Saved inpainting result as inpainting_volumes/inpainting_result.nii.gz")

    logger.info("Finished inpainting")
    return val_image_inpainted


def main(argv=None):
    """Entry point for the inpainting CLI (debug mode).

    Parses CLI arguments, prepares models, and runs ``inpaint_volume``.
    """
    SAVE_VOLUMES = True
    SAVE_IMAGES = True

    parser = argparse.ArgumentParser(description="Run 3D image inpainting using a pretrained DDPM model")
    parser.add_argument("-o", "--out_dir", type=str, default="debug_run", help="experiment output directory")
    parser.add_argument("-i", "--input_image", type=str, help="input image", required=True)
    parser.add_argument("-m", "--mask_image", type=str, help="input mask", default=None, required=False)
    parser.add_argument("--dilate", type=int, help="number of pixels to dilate the mask by", required=False, default=0)
    parser.add_argument("--keepgeom", action="store_true", help="Keep native output geometry while running inference in internal space")
    parser.add_argument(
        "--min_auto_img_size",
        "--min-auto-img-size",
        type=positive_int,
        default=None,
        help="Optional minimum side length for automatic conforming",
    )
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="Number of diffusion inference iterations (default: 1000)")
    parser.add_argument(
        "-c_coronal", "--checkpoint_coronal", type=str, help="checkpoint to load for inference in coronal plane", default=None, required=False
    )
    parser.add_argument(
        "-c_axial", "--checkpoint_axial", type=str, help="checkpoint to load for inference in axial plane", default=None, required=False
    )
    parser.add_argument(
        "-c_sagittal", "--checkpoint_sagittal", type=str, help="checkpoint to load for inference in sagittal plane", default=None, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="number of slices to process per GPU batch (default: 8); reduce to lower GPU memory usage"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device to use (default: auto)",
    )

    args = parser.parse_args(argv)

    # load models
    device = resolve_inference_device(args.device)
    logger.info(f"Using inference device: {device.type}")

    model_state_dicts = {}
    if args.checkpoint_coronal is not None:
        model_state_dicts["coronal"] = torch.load(args.checkpoint_coronal, map_location=device, weights_only=True)
    if args.checkpoint_axial is not None:
        model_state_dicts["axial"] = torch.load(args.checkpoint_axial, map_location=device, weights_only=True)
    if args.checkpoint_sagittal is not None:
        model_state_dicts["sagittal"] = torch.load(args.checkpoint_sagittal, map_location=device, weights_only=True)

    # setup model
    SLICE_THICKNESS = 7
    # make model out of weights only
    model_dict = {}
    for model_name, model_state_dict in model_state_dicts.items():
        model_dict[model_name] = DiffusionModelUNetVINN(
            spatial_dims=2,
            internal_size=(128, 128),
            in_channels=SLICE_THICKNESS,
            out_channels=SLICE_THICKNESS,
            num_channels=[128, 256, 512],  # [256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            norm_num_groups=4,
            use_fp16_VINN=False,
            is_vinn=True,
            interpolation_mode="bilinear",
        )
        model_dict[model_name].load_state_dict(model_state_dict)
        model_dict[model_name].to(device)

    # Add compilation for PyTorch 2.0+
    # print(f'Torch version: {torch.__version__}')
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
    model_to_dim = {"coronal": 2, "axial": 1, "sagittal": 0}
    if len(model_dict) == 0:
        print("ERROR: At least one checkpoint must be specified", file=sys.stderr)
        sys.exit(1)
    elif len(model_dict) == 1:
        DIM = model_to_dim[list(model_dict.keys())[0]]
    elif len(model_dict) == 3:
        DIM = 0
    else:
        print(f"ERROR: One or three checkpoints must be specified, but got {len(model_dict)}", file=sys.stderr)
        sys.exit(1)

    assert list(model_dict.values())[0].is_vinn

    val_image_native_nib = nib.load(args.input_image)
    val_image_nib = conform_nifti(val_image_native_nib, min_auto_img_size=args.min_auto_img_size)

    val_image = torch.from_numpy(val_image_nib.get_fdata()).float()

    mask_nib = nib.load(args.mask_image)
    # resample mask to image affine
    mask_nib = nibabel.processing.resample_from_to(mask_nib, val_image_nib, order=0, mode="constant", cval=0)

    mask = torch.from_numpy(mask_nib.get_fdata()).float()

    if args.dilate > 0:
        mask = dilate_mask(mask, args.dilate)

    INTERNAL_SHAPE = list(model_dict.values())[0].internal_size
    zooms = val_image_nib.header.get_zooms()
    scale_factor = vinn_scale_factor_from_zooms(INTERNAL_SHAPE, zooms)

    val_sample = {"image": val_image, "mask": mask}

    if not os.path.exists(os.path.join(args.out_dir, "inpainting_images")) or not os.path.exists(os.path.join(args.out_dir, "inpainting_volumes")):
        os.makedirs(os.path.join(args.out_dir, "inpainting_images"), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "inpainting_volumes"), exist_ok=True)
        logger.info(f"Created output directory: {args.out_dir}")
    else:
        logger.info(f"Output directory already exists: {args.out_dir}")

    tr = [
        # transforms.AddChanneld(keys=['image', 'mask']),
        transforms.EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
        transforms.ScaleIntensityd(keys=["image"]),
    ]

    data_transform = transforms.Compose(tr)
    val_sample_preproc = data_transform(val_sample)

    assert val_sample_preproc["image"].shape == val_sample_preproc["mask"].shape, (
        f"Image and mask must have the same shape, but got {val_sample_preproc['image'].shape} and {val_sample_preproc['mask'].shape}"
    )
    val_image = val_sample_preproc["image"]
    mask = val_sample_preproc["mask"]

    val_image_masked = val_image * (~(mask > 0)).float()

    inpaint_volume(
        models=model_dict,
        val_image=val_image,
        mask=mask,
        val_image_masked=val_image_masked,
        scale_factor=scale_factor,
        out_dir=args.out_dir,
        SAVE_VOLUMES=SAVE_VOLUMES,
        SAVE_IMAGES=SAVE_IMAGES,
        device=device,
        slice_input=False,
        slice_dim=DIM,
        val_image_nib=val_image_nib,
        DDIM=False,
        reference_image_nib=val_image_native_nib if args.keepgeom else None,
        num_inference_steps=args.num_inference_steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
