import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np
from platformdirs import user_data_dir

from neurolit._version import get_version_with_hash
from neurolit.inpaint_image import main as inpaint_main
from neurolit.inpaint_image import positive_int
from neurolit.utils.download_checkpoints import main as download_main
from neurolit.utils.geometry_policy import FASTSURFER_MIN_AUTO_IMG_SIZE


def _copy_file(src: Path, dst: Path) -> None:
    """Copy a file to ``dst``, replacing an existing destination if needed."""
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _copy_if_exists(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` when the source exists."""
    if src.exists():
        _copy_file(src, dst)


def _resample_mask_to_reference(src: Path, reference: Path, dst: Path) -> None:
    """Resample a mask to ``reference`` geometry using nearest-neighbor interpolation."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    mask_img = nib.load(str(src))
    reference_img = nib.load(str(reference))
    resampled = nibabel.processing.resample_from_to(mask_img, reference_img, order=0, mode="constant", cval=0)
    mask_dtype = mask_img.get_data_dtype()
    mask_data = np.asanyarray(resampled.dataobj).astype(mask_dtype, copy=False)
    header = reference_img.header.copy()
    header.set_data_dtype(mask_dtype)
    nib.save(nib.Nifti1Image(mask_data, reference_img.affine, header), str(dst))


def run_lit():
    """Run the neuroLIT CLI.

    This function serves as the main entry point for the Lesion Inpainting Tool
    command-line interface. It parses command-line arguments, validates inputs,
    and initiates the inpainting process.
    """
    parser = argparse.ArgumentParser(description="neuroLIT: Neuro Lesion Inpainting Tool", add_help=False)

    # Required/Common arguments (as used in run_lit.sh)
    parser.add_argument("-i", "--input_image", "--t1", help="Input T1w image")
    parser.add_argument("-m", "--lesion_mask", "--mask_image", help="Lesion mask")
    parser.add_argument("-o", "--sd", "--out_dir", "--output_dir", "--output_directory", help="Output directory")

    # Optional arguments
    parser.add_argument("--dilate", type=int, default=0, help="Number of times to dilate the lesion mask (default: 0)")
    parser.add_argument("--keepgeom", action="store_true", help="Preserve native output geometry")
    parser.add_argument(
        "--min_auto_img_size",
        "--min-auto-img-size",
        type=positive_int,
        default=None,
        help="Optional minimum side length for automatic conforming",
    )
    parser.add_argument(
        "--fastsurfer_dir",
        action="store_true",
        help="Treat the output directory as a FastSurfer subject directory and materialize FastSurfer-style outputs",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device to use (default: auto)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Slices per GPU batch (default: 8); reduce to lower GPU memory usage",
    )
    # Other arguments
    parser.add_argument("-h", "--help", action="store_true", help="Show help message and exit")
    parser.add_argument("-v", "--version", action="version", version=get_version_with_hash(), help="Print version number and exit")

    args, unknown = parser.parse_known_args()

    if args.help or (not args.input_image and not args.lesion_mask and not args.sd):
        print("Usage: lit-inpainting -i <input_t1w> -m <lesion_mask> -o <output_dir>")
        print("Required arguments:")
        print("  -i, --input_image     : Input T1w image")
        print("  -m, --lesion_mask, --mask_image : Lesion mask")
        print("  -o, --sd, --out_dir, --output_directory : Output directory")
        print("Optional arguments:")
        print("  --dilate              : Number of times to dilate the lesion mask (default: 0)")
        print("  --keepgeom            : Preserve native output geometry")
        print("  --min-auto-img-size   : Optional minimum side length for automatic conforming")
        print("  --fastsurfer_dir      : Treat output_directory as a FastSurfer subject directory")
        print("  --device              : Inference device: auto, cpu, or cuda (default: auto)")
        print("  --batch_size          : Slices per GPU batch (default: 8); reduce to lower GPU memory usage")
        print("Other arguments:")
        print("  -v, --version         : Print version number and exit")
        print("  -h, --help            : Show help message and exit")
        print("")
        print("If you use neuroLIT for research publications, please cite:")
        print("")
        print("Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole")
        print("  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Imaging Neuroscience 2025.")
        print("  https://doi.org/10.1162/imag_a_00446")
        sys.exit(0)

    if not args.input_image or not args.sd or not args.lesion_mask:
        print("Error: Input image, lesion mask, and output directory are required")
        sys.exit(1)
    # Validate input files
    input_image = Path(args.input_image).resolve()
    if not input_image.exists():
        print(f"Error: Input file {input_image} does not exist")
        sys.exit(1)

    mask_image = None
    if args.lesion_mask:
        mask_image = Path(args.lesion_mask).resolve()
        if not mask_image.exists():
            print(f"Error: Mask file {mask_image} does not exist")
            sys.exit(1)

    out_dir = Path(args.sd).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    min_auto_img_size = args.min_auto_img_size
    if args.fastsurfer_dir:
        min_auto_img_size = max(FASTSURFER_MIN_AUTO_IMG_SIZE, min_auto_img_size or 0)

    # Download checkpoints
    print("Checking/Downloading checkpoints...")
    download_main(argv=[])

    # Get checkpoint paths
    weights_dir = Path(user_data_dir("LIT", "Deep-MI")) / "weights"
    ckpt_coronal = weights_dir / "model_coronal.pt"
    ckpt_axial = weights_dir / "model_axial.pt"
    ckpt_sagittal = weights_dir / "model_sagittal.pt"

    # Check for required model files
    for model in [ckpt_coronal, ckpt_axial, ckpt_sagittal]:
        if not model.exists():
            print(f"Error: Required model file not found: {model}")
            sys.exit(1)

    if args.fastsurfer_dir:
        public_inpainted_img = out_dir / "mri" / "inpainted.lit.nii.gz"
        processed_mask_img = out_dir / "mri" / "mask.lit.nii.gz"
        public_mask_img = out_dir / "mri" / "orig" / "mask.lit.nii.gz"
        public_original_img = out_dir / "mri" / "orig" / "inpainting_original_image.lit.nii.gz"
        public_masked_img = out_dir / "mri" / "orig" / "inpainting_masked_image.lit.nii.gz"
        public_result_png = out_dir / "scripts" / "inpainting_result.lit.png"
        public_mask_png = out_dir / "scripts" / "inpainting_mask.lit.png"
        public_original_png = out_dir / "scripts" / "inpainting_original_image.lit.png"
        public_masked_png = out_dir / "scripts" / "inpainting_masked_image.lit.png"

        required_outputs = (
            public_inpainted_img,
            processed_mask_img,
            public_mask_img,
            public_original_img,
            public_masked_img,
            public_result_png,
            public_mask_png,
            public_original_png,
            public_masked_png,
        )
        if all(path.exists() for path in required_outputs):
            print(f"FastSurfer-compatible outputs already exist in {out_dir}")
        else:
            with tempfile.TemporaryDirectory(prefix="lit-inpainting.") as tmpdir:
                work_dir = Path(tmpdir)
                inpainted_img = work_dir / "inpainting_volumes" / "inpainting_result.nii.gz"
                print("Running inpainting...")

                inpaint_argv = [
                    "--input_image",
                    str(input_image),
                    "--mask_image",
                    str(mask_image),
                    "--out_dir",
                    str(work_dir),
                    "--checkpoint_axial",
                    str(ckpt_axial),
                    "--checkpoint_sagittal",
                    str(ckpt_sagittal),
                    "--checkpoint_coronal",
                    str(ckpt_coronal),
                    "--dilate",
                    str(args.dilate),
                    "--device",
                    args.device,
                    "--batch_size",
                    str(args.batch_size),
                ]

                if args.keepgeom:
                    inpaint_argv.append("--keepgeom")
                if min_auto_img_size is not None:
                    inpaint_argv.extend(["--min_auto_img_size", str(min_auto_img_size)])

                # Forward any unknown arguments
                inpaint_argv.extend(unknown)

                inpaint_main(inpaint_argv)

                _copy_file(inpainted_img, public_inpainted_img)
                if args.keepgeom:
                    _resample_mask_to_reference(work_dir / "inpainting_volumes" / "inpainting_mask.nii.gz", public_inpainted_img, processed_mask_img)
                else:
                    _copy_file(work_dir / "inpainting_volumes" / "inpainting_mask.nii.gz", processed_mask_img)
                _copy_file(mask_image, public_mask_img)
                _copy_if_exists(work_dir / "inpainting_volumes" / "inpainting_original_image.nii.gz", public_original_img)
                _copy_if_exists(work_dir / "inpainting_volumes" / "inpainting_masked_image.nii.gz", public_masked_img)
                _copy_if_exists(work_dir / "inpainting_images" / "inpainting_result.png", public_result_png)
                _copy_if_exists(work_dir / "inpainting_images" / "inpainting_mask.png", public_mask_png)
                _copy_if_exists(work_dir / "inpainting_images" / "inpainting_original_image.png", public_original_png)
                _copy_if_exists(work_dir / "inpainting_images" / "inpainting_masked_image.png", public_masked_png)

            print(f"Materialized FastSurfer-compatible outputs in {out_dir}")
    else:
        work_dir = out_dir
        inpainted_img = work_dir / "inpainting_volumes" / "inpainting_result.nii.gz"
        if not inpainted_img.exists():
            print("Running inpainting...")

            inpaint_argv = [
                "--input_image",
                str(input_image),
                "--mask_image",
                str(mask_image),
                "--out_dir",
                str(work_dir),
                "--checkpoint_axial",
                str(ckpt_axial),
                "--checkpoint_sagittal",
                str(ckpt_sagittal),
                "--checkpoint_coronal",
                str(ckpt_coronal),
                "--dilate",
                str(args.dilate),
                "--device",
                args.device,
                "--batch_size",
                str(args.batch_size),
            ]

            if args.keepgeom:
                inpaint_argv.append("--keepgeom")
            if min_auto_img_size is not None:
                inpaint_argv.extend(["--min_auto_img_size", str(min_auto_img_size)])

            # Forward any unknown arguments
            inpaint_argv.extend(unknown)

            inpaint_main(inpaint_argv)
        else:
            print(f"Inpainted image already exists: {inpainted_img}")

    if args.fastsurfer_dir:
        print(f"FastSurfer-compatible outputs available in {out_dir / 'mri'} and {out_dir / 'scripts'}")

    print("Finished inpainting")


if __name__ == "__main__":
    run_lit()
