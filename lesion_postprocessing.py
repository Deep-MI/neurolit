#!/usr/bin/env python3

"""
Script to process volumetric statistics with lesion mapping.
Supports both local and Docker execution modes.
Handles multiple segmentation files and segstats calls dynamically.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from LIT.postprocessing.lesion_to_segmentation import main as lesion_to_segmentation_main
from LIT.postprocessing.lesion_to_surface import main as lesion_to_surface_main


def setup_argparse() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Process volumetric statistics with lesion mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  FASTSURFER_HOME: Path to FastSurfer installation (preferred, checked first)
  FREESURFER_HOME: Path to FreeSurfer installation (fallback if FastSurfer not available)

Note: Either FastSurfer or FreeSurfer is required unless --skip-segstats is used.
      FastSurfer is preferred as it supports all features including the 'measures' subcommand.
        """
    )
    parser.add_argument('-sid', '--subject-id', required=True,
                        help='Subject ID')
    parser.add_argument('-sd', '--subjects-dir', required=True,
                        help='Subjects directory')
    parser.add_argument('--config', type=str,
                        help='Path to segstats configuration file (default: segstats_config.json)')
    parser.add_argument('--freesurfer-home', type=str,
                        help='Path to FreeSurfer installation (for LUT files, overrides FREESURFER_HOME)')
    parser.add_argument('--skip-segstats', action='store_true',
                        help='Skip all segstats calculations (only run lesion mapping and surface masking)')
    parser.add_argument('--skip-surface-masking', action='store_true',
                        help='Skip surface masking step')
    parser.add_argument('--python-cmd', type=str, default='python3',
                        help='Python command to use for running scripts (default: python3)')
    
    return parser


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load segstats configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "segstats_config.json"
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_segstats_installation() -> tuple[Optional[Path], bool]:
    """
    Validate segstats installation and determine which tool to use.
    
    Returns:
        (fastsurfer_path, use_mri_segstats): 
            - fastsurfer_path: Path to FastSurfer if found, None otherwise
            - use_mri_segstats: True if should use mri_segstats, False if segstats.py
    
    Raises:
        SystemExit: If neither FastSurfer nor FreeSurfer is available
    """
    print("=== Validating Segstats Installation ===")
    
    # Try to find FastSurfer (preferred)
    fastsurfer_home = os.environ.get('FASTSURFER_HOME')
    if fastsurfer_home:
        fs_path = Path(fastsurfer_home)
        segstats_script = fs_path / "FastSurferCNN" / "segstats.py"
        
        if fs_path.exists() and segstats_script.exists():
            # Test if segstats.py can be imported/run
            try:
                # Set PYTHONPATH and try to import
                test_env = os.environ.copy()
                test_env['PYTHONPATH'] = f"{fs_path}:{test_env.get('PYTHONPATH', '')}"
                result = subprocess.run(
                    [sys.executable, str(segstats_script), "--help"],
                    capture_output=True,
                    env=test_env,
                    timeout=5
                )
                if result.returncode == 0:
                    print(f"Found FastSurfer segstats.py: {segstats_script}")
                    print(f"FastSurfer path: {fs_path}")
                    return (fs_path, False)
                else:
                    print(f"FastSurfer segstats.py found but not working: {segstats_script}")
                    print(f"Error: {result.stderr.decode()[:200]}")
            except Exception as e:
                print(f"FastSurfer segstats.py found but failed to test: {e}")
    
    # Fall back to FreeSurfer mri_segstats
    freesurfer_home = os.environ.get('FREESURFER_HOME')
    if freesurfer_home or subprocess.run(['which', 'mri_segstats'], 
                                         capture_output=True).returncode == 0:
        try:
            result = subprocess.run(
                ['mri_segstats', '--help'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0 or b'USAGE' in result.stdout or b'USAGE' in result.stderr:
                print("Found FreeSurfer mri_segstats")
                if freesurfer_home:
                    print(f"  FreeSurfer path: {freesurfer_home}")
                print("Note: mri_segstats does not support 'measures' subcommand")
                return (None, True)
        except Exception as e:
            print(f"mri_segstats found but failed to test: {e}")
    
    # Neither found - print helpful error
    print("\n✗ ERROR: Neither FastSurfer nor FreeSurfer found!")
    print("\nTo use LIT postprocessing, you need either:")
    print("  1. FastSurfer:")
    print("     - Clone: git clone https://github.com/Deep-MI/FastSurfer.git")
    print("     - Set: export FASTSURFER_HOME=/path/to/FastSurfer")
    print("\n  2. FreeSurfer:")
    print("     - Install FreeSurfer from: https://surfer.nmr.mgh.harvard.edu/")
    print("     - Set: export FREESURFER_HOME=/path/to/FreeSurfer")
    print("     - Source: source $FREESURFER_HOME/SetUpFreeSurfer.sh")
    print("\n  3. Skip statistics calculations (--skip-segstats)")
    print("\nCurrent environment:")
    print(f"  FASTSURFER_HOME: {fastsurfer_home or '(not set)'}")
    print(f"  FREESURFER_HOME: {freesurfer_home or '(not set)'}")
    sys.exit(1)


def check_required_files(subjects_dir: Path, subject_id: str, config: Dict[str, Any]) -> None:
    """Check if all required input files exist based on config."""
    print("Checking for required input files...")
    
    required_files = [
        subjects_dir / subject_id / "inpainting_volumes" / "inpainting_mask.nii.gz",
        subjects_dir / subject_id / "mri" / "orig_nu.mgz",
        subjects_dir / subject_id / "mri" / "mask.mgz",
    ]
    
    # Check segmentation input files from config
    for seg_map in config.get('segmentation_mappings', []):
        if seg_map.get('map_lesion', False):
            input_file = subjects_dir / subject_id / seg_map['input_file']
            if input_file.exists():
                print(f"  Found: {seg_map['input_file']}")
            else:
                print(f"  Warning: Optional file not found: {seg_map['input_file']}")
    
    # Check critical files
    for file_path in required_files:
        if not file_path.exists():
            print(f"Error: Required file not found: {file_path}")
            sys.exit(1)
    
    print("All critical files found. Proceeding with processing...")


def run_command(cmd: List[str], description: str, env: Optional[Dict[str, str]] = None, 
                fail_ok: bool = False) -> bool:
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        if fail_ok:
            print(f"Warning: {description} failed with exit code {e.returncode}, continuing...")
            return False
        else:
            print(f"Error: {description} failed with exit code {e.returncode}")
            sys.exit(1)


def map_lesion_to_segmentation(subjects_dir: Path, subject_id: str,
                                input_file: str, output_file: str) -> bool:
    """Map lesion label to a segmentation file."""
    input_path = subjects_dir / subject_id / input_file
    output_path = subjects_dir / subject_id / output_file
    mask_path = subjects_dir / subject_id / "inpainting_volumes" / "inpainting_mask.nii.gz"

    lesion_to_segmentation_main(
        image=str(input_path),
        mask=str(mask_path),
        output=str(output_path)
    )


def build_segstats_command(fastsurfer_path: Optional[Path], subjects_dir: Path, subject_id: str,
                            config: Dict[str, Any], fs_home: Optional[Path] = None,
                            python_cmd: str = 'python3', use_mri_segstats: bool = False) -> List[str]:
    """Build segstats command from configuration dynamically."""
    if use_mri_segstats:
        cmd = ["mri_segstats"]
    else:
        if fastsurfer_path is None:
            raise ValueError("fastsurfer_path is required when not using mri_segstats")
        cmd = [python_cmd, str(fastsurfer_path / "FastSurferCNN" / "segstats.py")]
    
    subj_path = subjects_dir / subject_id
    
    # File paths (convert to absolute paths)
    for key, flag in [('segfile', '--segfile'), ('segstatsfile', '--segstatsfile'), 
                      ('normfile', '--normfile'), ('pvfile', '--pvfile')]:
        if key in config:
            cmd.extend([flag, str(subj_path / config[key])])
    
    # Simple value parameters
    cmd.extend(["--threads", "1"])
    for key, flag in [('volume_precision', '--volume_precision')]:
        if key in config:
            cmd.extend([flag, str(config[key])])
    
    # Boolean flags
    for key, flag in [('empty', '--empty'), ('measure_only', '--measure_only')]:
        if config.get(key, False):
            cmd.append(flag)
    
    # List parameters (each item as separate argument)
    for key, flag in [('excludeid', '--excludeid'), ('ids', '--ids')]:
        if key in config:
            cmd.extend([flag] + [str(x) for x in config[key]])
    
    # LUT handling - prefer FastSurfer LUTs, fall back to FreeSurfer
    if 'lut' in config:
        if config.get('lut_absolute', False):
            # For absolute LUTs (like ASegStatsLUT.txt), try multiple locations
            lut_found = False
            search_bases = [fastsurfer_path, fs_home]
            if os.environ.get('FREESURFER_HOME'):
                search_bases.append(Path(os.environ.get('FREESURFER_HOME')))
            
            for base in search_bases:
                if base and (base / config['lut']).exists():
                    lut_path = base / config['lut']
                    lut_found = True
                    break
            if not lut_found:
                # Use FreeSurfer home as fallback even if file doesn't exist (let segstats error)
                fallback = fs_home or (Path(os.environ.get('FREESURFER_HOME')) if os.environ.get('FREESURFER_HOME') else Path('/usr/local/freesurfer'))
                lut_path = fallback / config['lut']
        else:
            # For relative LUTs, use FastSurfer's config directory
            if fastsurfer_path is None:
                raise ValueError("Relative LUT paths require FastSurfer installation")
            lut_path = fastsurfer_path / "FastSurferCNN" / config['lut']
        cmd.extend(["--lut", str(lut_path)])
    
    # Measures subcommand
    if 'measures' in config:
        cmd.append("measures")
        measures = config['measures']
        
        # File and import
        if 'file' in measures:
            cmd.extend(["--file", str(subj_path / measures['file'])])
        if measures.get('import_all', False):
            cmd.extend(["--import", "all"])
        elif 'import' in measures:
            cmd.extend(["--import"] + measures['import'])
        
        # Compute with Mask() path resolution
        if 'compute' in measures:
            cmd.append("--compute")
            for item in measures['compute']:
                if item.startswith("Mask(") and not item.startswith("Mask(/"):
                    item = f"Mask({subj_path / item[5:-1]})"
                cmd.append(item)
    
    return cmd


def run_segstats(fastsurfer_path: Optional[Path], subjects_dir: Path, subject_id: str,
                       config: Dict[str, Any], fs_home: Optional[Path] = None,
                       use_mri_segstats: bool = False, python_cmd: str = 'python3') -> bool:
    """Run segstats locally using either FastSurfer's segstats.py or FreeSurfer's mri_segstats.
    
    Both commands receive exactly the same arguments. The only difference is the command itself:
    - FastSurfer: python3 segstats.py [args]
    - FreeSurfer: mri_segstats [args]
    """
    # Build command with appropriate tool
    cmd = build_segstats_command(fastsurfer_path, subjects_dir, subject_id, config, 
                                 fs_home, python_cmd, use_mri_segstats)
    
    if use_mri_segstats:
        # Note: mri_segstats does not support measures subcommand
        if 'measures' in config: # TODO: check this
            print("  Note: mri_segstats does not support 'measures' subcommand.")
            print("        Measures will not be computed. Use FastSurfer segstats.py for measures support.")
        
        return run_command(cmd, f"Running mri_segstats: {config['name']}", fail_ok=True)
    else:
        # Use FastSurfer's segstats.py - set up environment with PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{fastsurfer_path}:{env.get('PYTHONPATH', '')}"
        
        return run_command(cmd, f"Running segstats: {config['name']}", env=env, fail_ok=True)


def surface_masking(lit_path: Path, subjects_dir: Path, subject_id: str, hemisphere: str) -> bool:
    """Run surface masking for one hemisphere locally."""
    insurf = str(subjects_dir / subject_id / "surf" / f"{hemisphere}.white.preaparc")
    inseg = str(subjects_dir / subject_id / "inpainting_volumes" / "inpainting_mask.nii.gz")
    incort = str(subjects_dir / subject_id / "label" / f"{hemisphere}.cortex.label")
    surflut = str(lit_path / "LIT" / "postprocessing" / "DKTatlaslookup_lesion.txt")
    seglut = str(lit_path / "LIT" / "postprocessing" / "hemi.DKTatlaslookup_lesion.txt")
    out_annot = str(subjects_dir / subject_id / "label" / f"{hemisphere}.lesion.annot")
    to_annot = str(subjects_dir / subject_id / "label" / f"{hemisphere}.aparc.DKTatlas.annot")
    
    # Call the main function with all parameters
    lesion_to_surface_main(
        insurf=insurf,
        inseg=inseg,
        incort=incort,
        surflut=surflut,
        seglut=seglut,
        out_annot=out_annot,
        projmm=0.0,
        radius=None,
        to_annot=to_annot,
        dilation=3
    )



def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Convert to Path objects
    subjects_dir = Path(args.subjects_dir).resolve()
    subject_id = args.subject_id
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # Get FreeSurfer home if provided
    fs_home = Path(args.freesurfer_home) if args.freesurfer_home else Path(os.environ.get('FREESURFER_HOME')) if os.environ.get('FREESURFER_HOME') else None
    
    # Auto-detect LIT path
    lit_path = Path(__file__).parent
    
    # Validate segstats installation (unless skipping segstats)
    if not args.skip_segstats:
        fastsurfer_path, use_mri_segstats = validate_segstats_installation()
        
        # Update fs_home if not provided and using mri_segstats
        if use_mri_segstats and not fs_home:
            fs_home_env = os.environ.get('FREESURFER_HOME')
            if fs_home_env:
                fs_home = Path(fs_home_env)
    else:
        print("=== Skipping segstats validation (--skip-segstats enabled) ===")
        fastsurfer_path = None
        use_mri_segstats = False
    
    # Check required files
    check_required_files(subjects_dir, subject_id, config)
    
    # Create output directories
    (subjects_dir / subject_id / "stats").mkdir(parents=True, exist_ok=True)
    
    # Process all segmentation mappings
    print("\n" + "=" * 60)
    print("STEP 1: Mapping lesions to segmentation files")
    print("=" * 60)
    
    for seg_map in config.get('segmentation_mappings', []):
        if seg_map.get('map_lesion', False):
            print(f"\nProcessing: {seg_map['name']}")
            map_lesion_to_segmentation(subjects_dir, subject_id, seg_map['input_file'], seg_map['output_file'])
    
    # Run all segstats calls (unless skipped)
    if not args.skip_segstats:
        print("\n" + "=" * 60)
        print("STEP 2: Running segstats for all configurations")
        print("=" * 60)
        
        if use_mri_segstats:
            print("Using FreeSurfer mri_segstats (limited features)\n")
        else:
            print("Using FastSurfer segstats.py (full features)\n")
        
        for segstats_config in config.get('segstats_calls', []):
            print(f"\n{'=' * 60}")
            print(f"Running: {segstats_config['name']}")
            print(f"{'=' * 60}")
            
            # Check if segfile exists
            segfile_path = subjects_dir / subject_id / segstats_config['segfile']
            if not segfile_path.exists():
                print(f"  Skipping (segfile not found): {segstats_config['segfile']}")
                continue
            
            run_segstats(fastsurfer_path, subjects_dir, subject_id, segstats_config, 
                        fs_home, use_mri_segstats, args.python_cmd)
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Skipping segstats (--skip-segstats enabled)")
        print("=" * 60)
    
    # Run surface masking
    if not args.skip_surface_masking:
        print("\n" + "=" * 60)
        print("STEP 3: Running surface masking")
        print("=" * 60)
        
        for hemisphere in ["lh", "rh"]:
            surface_masking(lit_path, subjects_dir, subject_id, hemisphere)
    
    print("\n" + "=" * 60)
    print("Processing completed successfully.")
    print("=" * 60)
    
    # List output files
    stats_dir = subjects_dir / subject_id / "stats"
    if stats_dir.exists():
        print("\nGenerated statistics files:")
        for stats_file in sorted(stats_dir.glob("*+lesion*.stats")):
            print(f"  - {stats_file.name}")


if __name__ == "__main__":
    main()

