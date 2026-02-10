#!/usr/bin/env python3

"""
Script to process volumetric statistics with lesion mapping.
Supports both local and Docker execution modes.
Handles multiple segmentation files and segstats calls dynamically.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from neurolit.postprocessing.lesion_to_segmentation import main as lesion_to_segmentation_main
from neurolit.postprocessing.lesion_to_surface import main as lesion_to_surface_main
from neurolit.utils.logging import get_logger

logger = get_logger(__name__)


ADJACENT_LABEL_TARGET = 99


def setup_argparse() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with LIT postprocessing arguments.
    """
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
    parser.add_argument('--surf-config', type=str,
                        help='Path to surface stats configuration file (default: surfstats_config.json)')
    parser.add_argument('--freesurfer-home', type=str,
                        help='Path to FreeSurfer installation (for LUT files, overrides FREESURFER_HOME)')
    parser.add_argument('--skip-segstats', action='store_true',
                        help='Skip all segstats calculations (only run lesion mapping and surface masking)')
    parser.add_argument('--skip-surface-masking', action='store_true',
                        help='Skip surface masking step')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging (shows captured command output)')
    parser.add_argument('--python-cmd', type=str, default='python3',
                        help='Python command to use for running scripts (default: python3)')
    
    return parser


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Parameters
    ----------
    config_path : Path
        Location of the segstats configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration as a dictionary.
    """
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        return json.load(f)


def ensure_backup(file_path: Path) -> Path | None:
    """Ensure a backup of the original file exists before overwriting it.

    Adds '.lit' before the extension(s). Handles symlinks by creating a 
    corresponding .lit symlink.
    """
    if not file_path.exists(follow_symlinks=False):
        return None

    # Construct backup name (e.g., file.mgz -> file.lit.mgz, file.nii.gz -> file.lit.nii.gz)
    if file_path.name.endswith(".nii.gz"):
        ext = ".nii.gz"
    else:
        ext = file_path.suffix

    stem = file_path.name[:-len(ext)] if ext else file_path.name
    backup_name = f"{stem}.lit{ext}"
    backup_path = file_path.with_name(backup_name)

    if not backup_path.exists():
        logger.info(f"  Creating backup: {file_path.name} -> {backup_name}")
        if file_path.is_symlink():
            target = os.readlink(file_path)
            target_path = Path(target)
            # If target is relative and in the same directory, lit-ify it
            if not target_path.is_absolute():
                if target_path.name.endswith(".nii.gz"):
                    t_ext = ".nii.gz"
                else:
                    t_ext = target_path.suffix
                t_stem = target_path.name[:-len(t_ext)] if t_ext else target_path.name
                t_backup_name = f"{t_stem}.lit{t_ext}"
                os.symlink(t_backup_name, backup_path)
            else:
                os.symlink(target, backup_path)
        else:
            shutil.copy2(file_path, backup_path)

    # Also look for other symlinks in the same directory that point to this file
    # and create corresponding .lit symlinks for them.
    try:
        abs_file_path = file_path.resolve()
        for item in file_path.parent.iterdir():
            if not item.is_symlink():
                continue
            if item.name == backup_name or item.name.endswith(".lit" + ext):
                continue

            try:
                if item.resolve() == abs_file_path:
                    # This symlink points to our file! Create a .lit version of it.
                    s_ext = ".nii.gz" if item.name.endswith(".nii.gz") else item.suffix
                    s_stem = item.name[:-len(s_ext)] if s_ext else item.name
                    s_backup_name = f"{s_stem}.lit{s_ext}"
                    s_backup_path = item.with_name(s_backup_name)
                    if not s_backup_path.exists():
                        logger.info(f"  Creating backup symlink: {item.name} -> {s_backup_name}")
                        # Point the new .lit symlink to the .lit backup of the target
                        os.symlink(backup_name, s_backup_path)
            except (OSError, RuntimeError) as e:
                logger.debug(f"Could not resolve symlink {item}: {e}")
                continue
    except (OSError, RuntimeError) as e:
        logger.debug(f"Could not search for related symlinks in {file_path.parent}: {e}")

    return backup_path


def validate_segstats_installation() -> tuple[Path | None, bool]:
    """Validate segstats installation and select the appropriate tool.

    Returns
    -------
    tuple[Optional[Path], bool]
        (`fastsurfer_path`, `use_mri_segstats`): first element is a Path to FastSurfer if
        available, second element is True when FreeSurfer's `mri_segstats` should be used.

    Raises
    ------
    SystemExit
        When neither FastSurfer nor FreeSurfer is installed/configured or working.
    """
    logger.info("=== Validating Segstats Installation ===")
    
    # Try to find FastSurfer (preferred)
    fastsurfer_home = os.environ.get('FASTSURFER_HOME')
    fastsurfer_found = False
    if fastsurfer_home:
        fs_path = Path(fastsurfer_home)
        segstats_script = fs_path / "FastSurferCNN" / "segstats.py"
        
        if fs_path.exists() and segstats_script.exists():
            fastsurfer_found = True
            # Test if segstats.py can be imported/run
            try:
                # Set PYTHONPATH and try to import
                test_env = os.environ.copy()
                test_env['PYTHONPATH'] = f"{fs_path}:{test_env.get('PYTHONPATH', '')}"
                result = subprocess.run(
                    [sys.executable, str(segstats_script), "--help"],
                    capture_output=True,
                    env=test_env,
                    timeout=15
                )
                if result.returncode == 0:
                    logger.info(f"Found FastSurfer segstats.py: {segstats_script}")
                    logger.info(f"FastSurfer path: {fs_path}")
                    return (fs_path, False)
                else:
                    logger.warning(f"FastSurfer segstats.py found but not working: {segstats_script}")
                    stderr = result.stderr.decode()
                    logger.warning(f"Error: {stderr[:500]}")
                    if "ModuleNotFoundError" in stderr and "pandas" in stderr:
                        logger.error("\n[TIP] FastSurfer requires 'pandas'. Try: pip install pandas")
            except subprocess.TimeoutExpired:
                logger.warning("FastSurfer segstats.py test timed out (15s). It might still work, but initialization is slow.")
                logger.info(f"Assuming FastSurfer is available at: {fs_path}")
                return (fs_path, False)
            except Exception as e:
                logger.warning(f"FastSurfer segstats.py found but failed to test: {e}")
    
    # Fall back to FreeSurfer mri_segstats
    freesurfer_home = os.environ.get('FREESURFER_HOME')
    mri_segstats_bin = "mri_segstats"
    
    # Check if mri_segstats is in path or in FREESURFER_HOME/bin
    if freesurfer_home:
        fs_bin = Path(freesurfer_home) / "bin" / "mri_segstats"
        if fs_bin.exists():
            mri_segstats_bin = str(fs_bin)
    
    freesurfer_found = False
    if freesurfer_home or subprocess.run(['which', 'mri_segstats'], 
                                         capture_output=True).returncode == 0:
        freesurfer_found = True
        try:
            result = subprocess.run(
                [mri_segstats_bin, '--help'],
                capture_output=True,
                timeout=15
            )
            if result.returncode == 0 or b'USAGE' in result.stdout or b'USAGE' in result.stderr:
                logger.info(f"Found FreeSurfer mri_segstats: {mri_segstats_bin}")
                if freesurfer_home:
                    logger.info(f"  FreeSurfer path: {freesurfer_home}")
                logger.info("Note: mri_segstats does not support 'measures' subcommand")
                return (None, True)
        except subprocess.TimeoutExpired:
            logger.warning("FreeSurfer mri_segstats test timed out (15s). It might still work, but initialization is slow.")
            return (None, True)
        except Exception as e:
            logger.warning(f"mri_segstats found at {mri_segstats_bin} but failed to test: {e}")
    
    # Neither found or working - print helpful error
    if fastsurfer_found or freesurfer_found:
        logger.error("\n\u2717 ERROR: Tools were found but are not working correctly!")
    else:
        logger.error("\n\u2717 ERROR: Neither FastSurfer nor FreeSurfer found!")
    
    logger.error("\nTo use LIT postprocessing, you need either:")
    logger.error("  1. FastSurfer (preferred):")
    logger.error("     - Clone: git clone https://github.com/Deep-MI/FastSurfer.git")
    logger.error("     - Set: export FASTSURFER_HOME=/path/to/FastSurfer")
    logger.error("     - Note: requires 'pandas' in your python environment")
    logger.error("\n  2. FreeSurfer:")
    logger.error("     - Install FreeSurfer from: https://surfer.nmr.mgh.harvard.edu/")
    logger.error("     - Set: export FREESURFER_HOME=/path/to/FreeSurfer")
    logger.error("     - Source: source $FREESURFER_HOME/SetUpFreeSurfer.sh")
    logger.error("\n  3. Skip statistics calculations (--skip-segstats)")
    logger.error("\nCurrent environment:")
    logger.error(f"  FASTSURFER_HOME: {fastsurfer_home or '(not set)'}")
    logger.error(f"  FREESURFER_HOME: {freesurfer_home or '(not set)'}")
    sys.exit(1)


def check_required_files(subjects_dir: Path, subject_id: str, config: dict[str, Any]) -> None:
    """Verify that the essential files exist for a subject before processing.

    Parameters
    ----------
    subjects_dir : Path
        Subjects directory containing processed data.
    subject_id : str
        Identifier of the subject being processed.
    config : Dict[str, Any]
        Segstats configuration dictionary specifying mappings and files.
    """
    logger.info("Checking for required input files...")
    
    required_files = [
        subjects_dir / subject_id / "inpainting" / "inpainting_mask.nii.gz",
        subjects_dir / subject_id / "mri" / "orig_nu.mgz",
        subjects_dir / subject_id / "mri" / "mask.mgz",
    ]
    
    # Check segmentation input files from config
    for seg_map in config.get('segmentation_mappings', []):
        if seg_map.get('map_lesion', False):
            input_file = subjects_dir / subject_id / seg_map['input_file']
            if input_file.exists():
                logger.info(f"  Found: {seg_map['input_file']}")
            else:
                logger.warning(f"  Optional file not found: {seg_map['input_file']}")
    
    # Check critical files
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Required file not found: {file_path}")
            sys.exit(1)
    
    logger.info("All critical files found. Proceeding with processing...")


def run_command(cmd: list[str], description: str, env: dict[str, str] | None = None,
                fail_ok: bool = False) -> bool:
    """Run a subprocess command and log its output.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments to execute.
    description : str
        Human readable description of the command.
    env : Optional[Dict[str, str]], optional
        Environment variables for the subprocess, by default ``None`` (inherit current env).
    fail_ok : bool, optional
        Whether to treat failures as warnings instead of exiting, by default ``False``.

    Returns
    -------
    bool
        ``True`` when the command succeeded, ``False`` when it failed but ``fail_ok`` was ``True``.
    """
    logger.info(f"{description}...")
    logger.info(f"Command: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.debug(result.stdout.rstrip())
        if result.stderr:
            logger.debug(result.stderr.rstrip())
        return True
    except subprocess.CalledProcessError as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if stdout.strip():
            logger.error(stdout.rstrip())
        if stderr.strip():
            logger.error(stderr.rstrip())
        if fail_ok:
            logger.warning(f"{description} failed with exit code {e.returncode}, continuing...")
            return False
        logger.error(f"{description} failed with exit code {e.returncode}")
        sys.exit(1)


def map_lesion_to_segmentation(subjects_dir: Path, subject_id: str,
                                input_file: str, output_file: str, 
                                report_file: str | None = None, 
                                fs_home: Path | None = None) -> bool:
    """Map lesion labels into a segmentation file via the lesion_to_segmentation script.

    Parameters
    ----------
    subjects_dir : Path
        Root directory containing subject data.
    subject_id : str
        Identifier of the subject to process.
    input_file : str
        Relative path to the input segmentation to be modified.
    output_file : str
        Relative path to the desired output segmentation.
    report_file : str, optional
        Relative path to the anatomy report file.
    fs_home : Path, optional
        FreeSurfer installation path for LUT files.

    Returns
    -------
    bool
        Currently always ``True``; kept for signature compatibility.
    """
    input_path = subjects_dir / subject_id / input_file
    output_path = subjects_dir / subject_id / output_file
    mask_path = subjects_dir / subject_id / "inpainting" / "inpainting_mask.nii.gz"
    
    if not input_path.exists():
        logger.warning(f"  Skipping: {input_file} not found at {input_path}")
        return False

    # Handle backup if input and output are the same
    if input_path == output_path:
        backup_path = ensure_backup(input_path)
        if backup_path:
            # Use backup as input for the mapping
            input_path = backup_path

    report_path = subjects_dir / subject_id / report_file if report_file else None
    if report_path:
        ensure_backup(report_path)
    lut_path = _find_lut_file(fs_home) if report_file else None

    lesion_to_segmentation_main(
        image=str(input_path),
        mask=str(mask_path),
        output=str(output_path),
        report=str(report_path) if report_path else None,
        lut=str(lut_path) if lut_path else None
    )
    return True


def build_segstats_command(fastsurfer_path: Path | None, subjects_dir: Path, subject_id: str,
                            config: dict[str, Any], fs_home: Path | None = None,
                            python_cmd: str = 'python3', use_mri_segstats: bool = False) -> list[str]:
    """Build segstats command from the provided configuration.

    Parameters
    ----------
    fastsurfer_path : Optional[Path]
        Path to the FastSurfer installation (required when not using mri_segstats).
    subjects_dir : Path
        Subjects directory.
    subject_id : str
        Subject identifier.
    config : Dict[str, Any]
        Configuration describing the segstats call.
    fs_home : Optional[Path], optional
        FreeSurfer installation path for LUT fallbacks, by default ``None``.
    python_cmd : str, optional
        Python executable to run FastSurfer's script, by default ``'python3'``.
    use_mri_segstats : bool, optional
        Whether to run FreeSurfer's ``mri_segstats`` instead of FastSurfer's ``segstats.py``, by default ``False``.

    Returns
    -------
    List[str]
        Command line invocation including all configured flags.
    """
    if use_mri_segstats:
        cmd = ["mri_segstats"]
    else:
        if fastsurfer_path is None:
            raise ValueError("fastsurfer_path is required when not using mri_segstats")
        cmd = [python_cmd, str(fastsurfer_path / "FastSurferCNN" / "segstats.py")]
    
    subj_path = subjects_dir / subject_id
    
    # Subject and directory info (needed for surface measures)
    cmd.extend(["--sd", str(subjects_dir)])
    if not use_mri_segstats:
        cmd.extend(["--sid", subject_id])
    
    # File paths (convert to absolute paths)
    for key, flag in [('segfile', '--segfile'), ('in_file', '--in'), 
                      ('segstatsfile', '--segstatsfile'), ('sum', '--sum'),
                      ('normfile', '--normfile'), ('pvfile', '--pvfile')]:
        if key in config:
            cmd.extend([flag, str(subj_path / config[key])])
    
    # Special handling for --annot (takes subject_id, hemi, atlas)
    if 'annot' in config:
        hemi, atlas = config['annot']
        cmd.extend(["--annot", subject_id, hemi, atlas])

    # Simple value parameters
    if not use_mri_segstats:
        cmd.extend(["--threads", "1"])
    for key, flag in [('volume_precision', '--volume_precision')]:
        if key in config:
            cmd.extend([flag, str(config[key])])
    
    # Boolean flags
    for key, flag in [('empty', '--empty'), ('measure_only', '--measure_only'), ('snr', '--snr')]:
        if config.get(key, False):
            cmd.append(flag)
    
    # List parameters (each item as separate argument)
    for key, flag in [('excludeid', '--excludeid'), ('ids', '--ids')]:
        if key in config:
            cmd.extend([flag] + [str(x) for x in config[key]])
    
    # Merged labels (each list as separate --merged_label/--merge call)
    if 'merged_labels' in config:
        flag = "--merge" if use_mri_segstats else "--merged_label"
        for item in config['merged_labels']:
            cmd.extend([flag] + [str(x) for x in item])
    
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
                if fs_home:
                    fallback = fs_home
                elif os.environ.get('FREESURFER_HOME'):
                    fallback = Path(os.environ.get('FREESURFER_HOME'))
                else:
                    fallback = Path('/usr/local/freesurfer')
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


def run_segstats(fastsurfer_path: Path | None, subjects_dir: Path, subject_id: str,
                       config: dict[str, Any], fs_home: Path | None = None,
                       use_mri_segstats: bool = False, python_cmd: str = 'python3') -> bool:
    """Invoke segstats with the configured arguments.

    Parameters
    ----------
    fastsurfer_path : Optional[Path]
        Path to FastSurfer root (ignored when ``use_mri_segstats`` is ``True``).
    subjects_dir : Path
        Directory containing subject data.
    subject_id : str
        Identifier for the subject.
    config : Dict[str, Any]
        Segstats call configuration.
    fs_home : Optional[Path], optional
        Path to FreeSurfer installation, by default ``None``.
    use_mri_segstats : bool, optional
        Whether to use FreeSurfer's ``mri_segstats`` (True) versus FastSurfer's ``segstats.py`` (False),
        by default ``False``.
    python_cmd : str, optional
        Python executable for FastSurfer, by default ``'python3'``.

    Returns
    -------
    bool
        ``True`` when the command succeeded or failure was tolerated via ``fail_ok``.
    """
    # Build command with appropriate tool
    cmd = build_segstats_command(fastsurfer_path, subjects_dir, subject_id, config, 
                                 fs_home, python_cmd, use_mri_segstats)
    
    # Ensure backups for output files
    subj_path = subjects_dir / subject_id
    if 'segstatsfile' in config:
        ensure_backup(subj_path / config['segstatsfile'])
    if 'sum' in config:
        ensure_backup(subj_path / config['sum'])

    if use_mri_segstats:
        # Note: mri_segstats does not support measures subcommand
        if 'measures' in config:
            logger.info("  Note: mri_segstats does not support 'measures' subcommand.")
            logger.info("        Measures will not be computed. Use FastSurfer segstats.py for measures support.")
        
        return run_command(cmd, f"Running mri_segstats: {config['name']}", fail_ok=True)
    else:
        # Use FastSurfer's segstats.py - set up environment with PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{fastsurfer_path}:{env.get('PYTHONPATH', '')}"
        
        return run_command(cmd, f"Running segstats: {config['name']}", env=env, fail_ok=True)


def build_surfstats_command(subjects_dir: Path, subject_id: str, hemisphere: str, 
                            config: dict[str, Any]) -> list[str]:
    """Build mris_anatomical_stats command from configuration."""
    cmd = ["mris_anatomical_stats"]
    subj_path = subjects_dir / subject_id
    
    # Boolean flags
    for key, flag in [('th3', '-th3'), ('mgz', '-mgz'), ('b', '-b')]:
        if config.get(key, False):
            cmd.append(flag)
            
    # File paths with hemisphere interpolation
    if 'cortex' in config:
        cmd.extend(["-cortex", str(subj_path / config['cortex'].format(hemi=hemisphere))])
    
    if 'statsfile' in config:
        cmd.extend(["-f", str(subj_path / config['statsfile'].format(hemi=hemisphere))])
        
    if 'annot' in config:
        cmd.extend(["-a", str(subj_path / config['annot'].format(hemi=hemisphere))])
        
    if 'ctab' in config:
        cmd.extend(["-c", str(subj_path / config['ctab'].format(hemi=hemisphere))])
        
    # Positional arguments: subject hemi surface
    cmd.extend([subject_id, hemisphere, config.get('surface', 'white')])
    
    return cmd


def run_surfstats(subjects_dir: Path, subject_id: str, hemisphere: str, 
                  config: dict[str, Any]) -> bool:
    """Run surface-based anatomical statistics."""
    subj_path = subjects_dir / subject_id
    
    # Check if required files exist before running
    required_files = []
    if 'cortex' in config:
        required_files.append(subj_path / config['cortex'].format(hemi=hemisphere))
    if 'annot' in config:
        required_files.append(subj_path / config['annot'].format(hemi=hemisphere))
    
    # Also check pial/white surface
    surf_name = config.get('surface', 'white')
    required_files.append(subj_path / "surf" / f"{hemisphere}.{surf_name}")

    for f in required_files:
        if not f.exists():
            logger.warning(f"  Skipping surface stats for {hemisphere}: {f.name} not found.")
            return False

    # Ensure backup for statsfile
    if 'statsfile' in config:
        stats_path = subj_path / config['statsfile'].format(hemi=hemisphere)
        ensure_backup(stats_path)

    cmd = build_surfstats_command(subjects_dir, subject_id, hemisphere, config)
    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(subjects_dir)
    return run_command(cmd, f"Running mris_anatomical_stats: {hemisphere}.{config['name']}", env=env, fail_ok=True)


def _find_lut_file(fs_home: Path | None) -> Path | None:
    """Return the FreeSurferColorLUT.txt path if available."""
    candidates = []
    if fs_home:
        candidates.append(fs_home / "FreeSurferColorLUT.txt")
    env_fs = os.environ.get("FREESURFER_HOME")
    if env_fs:
        candidates.append(Path(env_fs) / "FreeSurferColorLUT.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def surface_masking(lit_path: Path, subjects_dir: Path, subject_id: str, 
                    hemisphere: str, input_file: str, output_file: str,
                    report_file: str | None = None,
                    output_ctab: str | None = None) -> bool:
    """Apply surface masking for the hemisphere based on lesion annotations.

    Parameters
    ----------
    lit_path : Path
        Root directory of the LIT repository.
    subjects_dir : Path
        Directory containing subject data.
    subject_id : str
        Subject identifier.
    hemisphere : str
        Hemisphere code (``'lh'`` or ``'rh'``).
    input_file : str
        Relative path to the input annotation file (with {hemi} placeholder).
    output_file : str
        Relative path to the output annotation file (with {hemi} placeholder).
    report_file : str, optional
        Relative path to the surface anatomy report file (with {hemi} placeholder).
    output_ctab : str, optional
        Relative path to the output color table file.

    Returns
    -------
    bool
        Always ``True`` (kept for backwards compatibility).
    """
    insurf = str(subjects_dir / subject_id / "surf" / f"{hemisphere}.pial")
    inseg = str(subjects_dir / subject_id / "inpainting" / "inpainting_mask.nii.gz")
    incort = str(subjects_dir / subject_id / "label" / f"{hemisphere}.cortex.label")
    surflut = str(lit_path / "postprocessing" / "DKTatlaslookup_lesion.txt")
    seglut = str(lit_path / "postprocessing" / "hemi.DKTatlaslookup_lesion.txt")
    
    out_annot_path = subjects_dir / subject_id / output_file.format(hemi=hemisphere)
    out_annot = str(out_annot_path)
    to_annot_path = subjects_dir / subject_id / input_file.format(hemi=hemisphere)
    to_annot = str(to_annot_path)

    if not to_annot_path.exists():
        logger.warning(f"  Skipping: {input_file.format(hemi=hemisphere)} not found at {to_annot_path}")
        return False

    # Handle backup if input and output are the same
    if to_annot_path == out_annot_path:
        backup_path = ensure_backup(to_annot_path)
        if backup_path:
            # Use backup as input for the mapping
            to_annot = str(backup_path)
    
    report_path = subjects_dir / subject_id / report_file.format(hemi=hemisphere) if report_file else None
    if report_path:
        ensure_backup(report_path)
    ctab_path = subjects_dir / subject_id / output_ctab.format(hemi=hemisphere) if output_ctab else None
    if ctab_path:
        ensure_backup(ctab_path)

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
        dilation=3,
        report=str(report_path) if report_path else None,
        out_ctab=str(ctab_path) if ctab_path else None
    )
    return True


def _format_inputs(inputs: list[str]) -> str:
    return " + ".join(inputs)


def build_stats_overview(subject_id: str, config: dict[str, Any], surf_config: dict[str, Any]) -> list[str]:
    """Build human-readable mapping of inputs to output stats files."""
    overview: list[str] = []

    for segstats_call in config.get("segstats_calls", []):
        inputs: list[str] = []
        if "segfile" in segstats_call:
            inputs.append(segstats_call["segfile"])
        if "in_file" in segstats_call:
            inputs.append(segstats_call["in_file"])
        if "normfile" in segstats_call:
            inputs.append(segstats_call["normfile"])
        if "pvfile" in segstats_call:
            inputs.append(segstats_call["pvfile"])

        if "annot" in segstats_call:
            hemi, atlas = segstats_call["annot"]
            inputs.append(f"label/{hemi}.{atlas}.annot")

        output = segstats_call.get("segstatsfile") or segstats_call.get("sum")
        if inputs and output:
            overview.append(f"{_format_inputs(inputs)} => {output}")

    for surfstats_call in surf_config.get("surfstats_calls", []):
        for hemisphere in ["lh", "rh"]:
            inputs = []
            if "cortex" in surfstats_call:
                inputs.append(surfstats_call["cortex"].format(hemi=hemisphere))
            if "annot" in surfstats_call:
                inputs.append(surfstats_call["annot"].format(hemi=hemisphere))
            if "ctab" in surfstats_call:
                inputs.append(surfstats_call["ctab"].format(hemi=hemisphere))
            surface = surfstats_call.get("surface", "white")
            inputs.append(f"surf/{hemisphere}.{surface}")

            output = surfstats_call.get("statsfile")
            if output:
                output = output.format(hemi=hemisphere)

            if inputs and output:
                overview.append(f"{_format_inputs(inputs)} => {output}")

    return overview


def generate_lesion_impact_summary(subjects_dir: Path, subject_id: str, 
                                   mapping_reports: list[tuple[str, str]], 
                                   surface_reports: list[tuple[str, str]]) -> Path | None:
    """Generate a machine-parseable YAML report of lesion impact.

    Parameters
    ----------
    subjects_dir : Path
        Subjects directory.
    subject_id : str
        Subject identifier.
    mapping_reports : List[Tuple[str, str]]
        List of (output_file, report_file) for volumetric mappings.
    surface_reports : List[Tuple[str, str]]
        List of (output_file, report_file) for surface mappings.

    Returns
    -------
    Optional[Path]
        Path to the generated summary report, or None if no reports found.
    """
    logger.info("Generating lesion impact summary (YAML)...")
    
    summary_path = subjects_dir / subject_id / "stats" / "lesion_impact_summary.yaml"
    
    impact = {
        "lh_cortical": False,
        "rh_cortical": False,
        "lh_subcortical": False,
        "rh_subcortical": False,
        "affected_labels": []
    }

    def parse_report(report_path: Path, is_surface: bool = False, hemi: str | None = None):
        if not report_path.exists():
            return
        
        with open(report_path) as f:
            lines = f.readlines()
            
        current_cat = None
        for line in lines:
            line = line.strip()
            if "Replaced Labels" in line or "Reduced Labels" in line:
                current_cat = "overlap"
                continue
            if "Adjacent Labels" in line:
                current_cat = "adjacent"
                continue
            
            if current_cat == "overlap" and line.startswith("#") and "LabelName" not in line and "None found" not in line:
                parts = line[1:].split()
                if not parts: continue
                
                try:
                    lid = int(parts[0])
                    lname = parts[1] if len(parts) > 1 else str(lid)
                except ValueError:
                    continue

                if is_surface:
                    if hemi == "lh": impact["lh_cortical"] = True
                    elif hemi == "rh": impact["rh_cortical"] = True
                else:
                    # Volumetric DKT+aseg logic
                    is_lh = (1000 <= lid < 2000) or lname.startswith("Left-") or lname.startswith("ctx-lh-")
                    is_rh = (2000 <= lid < 3000) or lname.startswith("Right-") or lname.startswith("ctx-rh-")
                    is_cortical = (1000 <= lid < 3000) or lname.startswith("ctx-")
                    
                    if is_lh:
                        if is_cortical: impact["lh_cortical"] = True
                        else: impact["lh_subcortical"] = True
                    if is_rh:
                        if is_cortical: impact["rh_cortical"] = True
                        else: impact["rh_subcortical"] = True
                
                impact["affected_labels"].append(f"{hemi + '.' if hemi else ''}{lname}")

    # Process volumetric reports
    for _, report_file in mapping_reports:
        if "aparc.DKTatlas+aseg" in report_file:
            parse_report(subjects_dir / subject_id / report_file)

    # Process surface reports
    for _, report_file in surface_reports:
        hemi = "lh" if "lh." in report_file else "rh" if "rh." in report_file else None
        parse_report(subjects_dir / subject_id / report_file, is_surface=True, hemi=hemi)

    # Generate YAML content with professional commenting
    yaml_lines = [
        "# Lesion Impact Summary Report",
        "# " + "=" * 60,
        f"# Subject: {subject_id}",
        "# " + "=" * 60,
        "",
        "# General involvement status per hemisphere",
        "hemisphere_impact:",
        "  # True if any cortical or subcortical structure in the left hemisphere is affected",
        f"  left_hemisphere_affected: {str(impact['lh_cortical'] or impact['lh_subcortical']).lower()}",
        "  # True if any cortical or subcortical structure in the right hemisphere is affected",
        f"  right_hemisphere_affected: {str(impact['rh_cortical'] or impact['rh_subcortical']).lower()}",
        "",
        "# Categorized breakdown of affected regions",
        "detailed_involvement:",
        "  # Involvement of left hemisphere cortical structures (from surface/volumetric annotations)",
        f"  left_cortical_affected: {str(impact['lh_cortical']).lower()}",
        "  # Involvement of right hemisphere cortical structures (from surface/volumetric annotations)",
        f"  right_cortical_affected: {str(impact['rh_cortical']).lower()}",
        "  # Involvement of left hemisphere subcortical structures (from volumetric segmentation)",
        f"  left_subcortical_affected: {str(impact['lh_subcortical']).lower()}",
        "  # Involvement of right hemisphere subcortical structures (from volumetric segmentation)",
        f"  right_subcortical_affected: {str(impact['rh_subcortical']).lower()}",
        "",
        "# List of structures showing spatial overlap with the lesion mask",
        "affected_structures:"
    ]
    
    unique_labels = sorted(list(set(impact["affected_labels"])))
    if unique_labels:
        for label in unique_labels:
            yaml_lines.append(f"  - {label}")
    else:
        yaml_lines.append("  []")

    with open(summary_path, 'w') as f:
        f.write("\n".join(yaml_lines) + "\n")

    return summary_path


def main():
    """Main entry point orchestrating LIT postprocessing.

    Parses CLI arguments, validates segstats tooling, runs lesion mapping,
    executes segstats (unless skipped), and applies surface masking as required.
    """
    parser = setup_argparse()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Convert to Path objects
    subjects_dir = Path(args.subjects_dir).resolve()
    subject_id = args.subject_id
    
    # Load configurations
    config_path = Path(args.config) if args.config else Path(__file__).parent.parent / "postprocessing" / "segstats_config.json"
    config = load_config(config_path)
    
    surf_config_path = Path(args.surf_config) if args.surf_config else Path(__file__).parent.parent / "postprocessing" / "surfstats_config.json"
    surf_config = load_config(surf_config_path) if surf_config_path.exists() else {"surfstats_calls": []}
    
    # Get FreeSurfer home if provided
    if args.freesurfer_home:
        fs_home = Path(args.freesurfer_home)
    elif os.environ.get('FREESURFER_HOME'):
        fs_home = Path(os.environ.get('FREESURFER_HOME'))
    else:
        fs_home = None
    
    # Auto-detect LIT path
    lit_path = Path(__file__).parent.parent
    
    # Validate segstats installation (unless skipping segstats)
    if not args.skip_segstats:
        fastsurfer_path, use_mri_segstats = validate_segstats_installation()
        
        # Update fs_home if not provided and using mri_segstats
        if use_mri_segstats and not fs_home:
            fs_home_env = os.environ.get('FREESURFER_HOME')
            if fs_home_env:
                fs_home = Path(fs_home_env)
    else:
        logger.info("=== Skipping segstats validation (--skip-segstats enabled) ===")
        fastsurfer_path = None
        use_mri_segstats = False
    
    # Check required files
    check_required_files(subjects_dir, subject_id, config)
    
    # Create output directories
    (subjects_dir / subject_id / "stats").mkdir(parents=True, exist_ok=True)
    
    # Process all segmentation mappings
    logger.info("=" * 60)
    logger.info("STEP 1: Mapping lesions to segmentation files")
    logger.info("=" * 60)
    
    mapping_reports: list[tuple[str, str]] = []
    for seg_map in config.get('segmentation_mappings', []):
        if seg_map.get('map_lesion', False):
            logger.info(f"Processing: {seg_map['name']}")
            report_file = seg_map.get('report_file')
            success = map_lesion_to_segmentation(subjects_dir, subject_id, seg_map['input_file'], 
                                               seg_map['output_file'], report_file, fs_home)
            if success and report_file:
                mapping_reports.append((seg_map['output_file'], report_file))
    
    # Run all segstats calls (unless skipped)
    surface_segstats_calls: list[dict[str, Any]] = []
    if not args.skip_segstats:
        logger.info("=" * 60)
        logger.info("STEP 2: Running segstats for all configurations")
        logger.info("=" * 60)
        
        if use_mri_segstats:
            logger.info("Using FreeSurfer mri_segstats (limited features)")
        else:
            logger.info("Using FastSurfer segstats.py (full features)")
        
        for segstats_config in config.get('segstats_calls', []):
            # Determine if this specific call needs mri_segstats
            # (Surface-based calls with --in or --annot are FS-only)
            call_use_mri_segstats = use_mri_segstats or \
                                   'annot' in segstats_config or \
                                   'in_file' in segstats_config or \
                                   segstats_config.get('use_mri_segstats', False)

            if call_use_mri_segstats:
                surface_segstats_calls.append(segstats_config)
                continue

            logger.info("-" * 60)
            logger.info(f"Running: {segstats_config['name']}")
            logger.info("-" * 60)
            
            # Check if input file exists
            input_path = None
            input_name = None
            if 'segfile' in segstats_config:
                input_path = subjects_dir / subject_id / segstats_config['segfile']
                input_name = "segfile"
            elif 'in_file' in segstats_config:
                input_path = subjects_dir / subject_id / segstats_config['in_file']
                input_name = "in_file"

            if input_path and not input_path.exists():
                logger.info(f"  Skipping ({input_name} not found): {input_path}")
                continue
            
            run_segstats(fastsurfer_path, subjects_dir, subject_id, segstats_config, 
                        fs_home, call_use_mri_segstats, args.python_cmd)

    else:
        logger.info("=" * 60)
        logger.info("STEP 2: Skipping segstats (--skip-segstats enabled)")
        logger.info("=" * 60)
    
    # Run surface masking (before surface-based statistics)
    surface_reports: list[tuple[str, str]] = []
    if not args.skip_surface_masking:
        logger.info("=" * 60)
        logger.info("STEP 3: Running surface masking")
        logger.info("=" * 60)
        
        for surf_map in surf_config.get('surface_mappings', []):
            if surf_map.get('map_lesion', False):
                logger.info(f"Processing: {surf_map['name']}")
                report_file = surf_map.get('report_file')
                output_ctab = surf_map.get('output_ctab')
                for hemisphere in ["lh", "rh"]:
                    logger.info(f"  Hemisphere: {hemisphere}")
                    success = surface_masking(lit_path, subjects_dir, subject_id, hemisphere, 
                                            surf_map['input_file'], surf_map['output_file'], 
                                            report_file, output_ctab)
                    if success and report_file:
                        surface_reports.append((surf_map['output_file'].format(hemi=hemisphere), 
                                              report_file.format(hemi=hemisphere)))

    # Run surface-related segstats (after surface masking)
    if not args.skip_surface_masking and surface_segstats_calls:
        logger.info("=" * 60)
        logger.info("STEP 4: Running surface segstats (mri_segstats)")
        logger.info("=" * 60)

        for segstats_config in surface_segstats_calls:
            logger.info("-" * 60)
            logger.info(f"Running: {segstats_config['name']}")
            logger.info("-" * 60)

            input_path = None
            input_name = None
            if 'segfile' in segstats_config:
                input_path = subjects_dir / subject_id / segstats_config['segfile']
                input_name = "segfile"
            elif 'in_file' in segstats_config:
                input_path = subjects_dir / subject_id / segstats_config['in_file']
                input_name = "in_file"

            if input_path and not input_path.exists():
                logger.info(f"  Skipping ({input_name} not found): {input_path}")
                continue

            run_segstats(fastsurfer_path, subjects_dir, subject_id, segstats_config,
                        fs_home, True, args.python_cmd)

    # Run surface-based statistics
    if not args.skip_surface_masking and surf_config.get("surfstats_calls"):
        logger.info("=" * 60)
        logger.info("STEP 5: Running surface statistics")
        logger.info("=" * 60)

        for surfstats_call in surf_config.get("surfstats_calls", []):
            for hemisphere in ["lh", "rh"]:
                run_surfstats(subjects_dir, subject_id, hemisphere, surfstats_call)

    logger.info("=" * 60)
    logger.info("Processing completed successfully.")
    logger.info("=" * 60)
    
    # List output files with input overview
    overview = build_stats_overview(subject_id, config, surf_config)
    overview.extend(f"{seg} => {report}" for seg, report in mapping_reports)
    overview.extend(f"{seg} => {report}" for seg, report in surface_reports)
    
    # Generate impact summary
    summary_path = generate_lesion_impact_summary(subjects_dir, subject_id, mapping_reports, surface_reports)
    if summary_path:
        overview.append(f"LESION_IMPACT_SUMMARY => {summary_path.relative_to(subjects_dir / subject_id)}")

    if overview:
        logger.info("Generated statistics files:")
        for line in overview:
            logger.info(f" - {line}")


if __name__ == "__main__":
    main()

