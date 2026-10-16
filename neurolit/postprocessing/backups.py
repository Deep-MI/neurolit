"""Preserve the first pre-lesion snapshot of files and their symlink aliases."""

import os
import shutil
from pathlib import Path

from neurolit.utils.log import get_logger

logger = get_logger(__name__)


def _backup_path(path: Path) -> Path:
    extension = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
    stem = path.name[:-len(extension)] if extension else path.name
    return path.with_name(f"{stem}.lit{extension}")


def ensure_backup(file_path: Path) -> Path | None:
    """Preserve a file's first snapshot, including symlink targets and aliases.

    Parameters
    ----------
    file_path : Path
        Original file or symlink. Broken symlinks are preserved as broken links.

    Returns
    -------
    Path or None
        Snapshot path with ``.lit`` inserted before the extension, or ``None``
        when the original does not exist. Existing snapshots are never replaced.
    """
    # lexists retains broken links and supports Python 3.10 and 3.11.
    if not os.path.lexists(file_path):
        return None

    backup_path = _backup_path(file_path)
    if not os.path.lexists(backup_path):
        if file_path.is_symlink():
            target = Path(os.readlink(file_path))
            target_backup = _backup_path(target)
            try:
                resolved_target = (file_path.parent / target).resolve()
            except (OSError, RuntimeError) as error:
                logger.debug(f"Could not resolve symlink target {target} for {file_path}: {error}")
            else:
                snapshot = ensure_backup(resolved_target)
                if snapshot is not None:
                    target_backup = snapshot if target.is_absolute() else Path(os.path.relpath(snapshot, file_path.parent))

            # Backing up the target can also create this alias's snapshot.
            if not os.path.lexists(backup_path):
                logger.info(f"  Creating backup: {file_path.name} -> {backup_path.name}")
                os.symlink(target_backup, backup_path)
        else:
            logger.info(f"  Creating backup: {file_path.name} -> {backup_path.name}")
            shutil.copy2(file_path, backup_path)

    # Preserve the established same-directory alias behavior without a registry.
    try:
        resolved_file = file_path.resolve()
        for item in file_path.parent.iterdir():
            extension = ".nii.gz" if item.name.endswith(".nii.gz") else item.suffix
            if not item.is_symlink() or item == backup_path or item.name.endswith(f".lit{extension}"):
                continue
            try:
                if item.resolve() == resolved_file:
                    alias_backup = _backup_path(item)
                    if not os.path.lexists(alias_backup):
                        logger.info(f"  Creating backup symlink: {item.name} -> {alias_backup.name}")
                        os.symlink(backup_path.name, alias_backup)
            except (OSError, RuntimeError) as error:
                logger.debug(f"Could not resolve symlink {item}: {error}")
    except (OSError, RuntimeError) as error:
        logger.debug(f"Could not search for related symlinks in {file_path.parent}: {error}")

    return backup_path
