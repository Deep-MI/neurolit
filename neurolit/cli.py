import argparse
import sys
from pathlib import Path

from platformdirs import user_data_dir

from neurolit._version import get_version_with_hash
from neurolit.inpaint_image import main as inpaint_main
from neurolit.utils.download_checkpoints import main as download_main


def run_lit():
    """Run the neuroLIT CLI.

    This function serves as the main entry point for the Lesion Inpainting Tool
    command-line interface. It parses command-line arguments, validates inputs,
    and initiates the inpainting process.
    """
    parser = argparse.ArgumentParser(
        description="neuroLIT: Neuro Lesion Inpainting Tool",
        add_help=False
    )
    
    # Required/Common arguments (as used in run_lit.sh)
    parser.add_argument("-i", "--input_image", "--t1", help="Input T1w image")
    parser.add_argument("-m", "--lesion_mask", help="Lesion mask")
    parser.add_argument("-o", "--sd", "--out_dir", "--output_dir", "--output_directory", help="Output directory")
    
    # Optional arguments
    parser.add_argument("--dilate", type=int, default=0, help="Number of times to dilate the lesion mask (default: 0)")
    
    # Other arguments
    parser.add_argument("-h", "--help", action="store_true", help="Show help message and exit")
    parser.add_argument("-v", "--version", action="version", version=get_version_with_hash(), help="Print version number and exit")

    args, unknown = parser.parse_known_args()

    if args.help or (not args.input_image and not args.lesion_mask and not args.sd):
        print("Usage: lit-inpainting -i <input_t1w> -m <lesion_mask> -o <output_dir>")
        print("Required arguments:")
        print("  -i, --input_image     : Input T1w image")
        print("  -m, --lesion_mask     : Lesion mask")
        print("  -o, --sd, --out_dir, --output_directory : Output directory")
        print("Optional arguments:")
        print("  --dilate              : Number of times to dilate the lesion mask (default: 0)")
        print("Other arguments:")
        print("  -v, --version         : Print version number and exit")
        print("  -h, --help            : Show help message and exit")
        print("")
        print("If you use neuroLIT for research publications, please cite:")
        print("")
        print("Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole")
        print("  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Accepted for Imaging Neuroscience.")
        sys.exit(0)

    if not args.input_image or not args.sd:
        print("Error: Input image and output directory are required")
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

    # Run inpainting
    if mask_image:
        inpainted_img = out_dir / "inpainting_volumes" / "inpainting_result.nii.gz"
        if not inpainted_img.exists():
            print("Running inpainting...")
            
            inpaint_argv = [
                "--input_image", str(input_image),
                "--mask_image", str(mask_image),
                "--out_dir", str(out_dir),
                "--checkpoint_axial", str(ckpt_axial),
                "--checkpoint_sagittal", str(ckpt_sagittal),
                "--checkpoint_coronal", str(ckpt_coronal),
                "--dilate", str(args.dilate)
            ]
            
            # Forward any unknown arguments
            inpaint_argv.extend(unknown)
            
            inpaint_main(inpaint_argv)
        else:
            print(f"Inpainted image already exists: {inpainted_img}")

    print("Finished inpainting")


if __name__ == "__main__":
    run_lit()
