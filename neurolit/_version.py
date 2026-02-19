"""Version number."""

import subprocess
from importlib.metadata import version
from pathlib import Path

__version__ = version(__package__)


def get_version_with_hash():
    """Get the version of the package including git hash if available.

    Returns
    -------
    str
        Version string.
    """
    version_str = __version__
    # Try to get git hash
    try:
        proj_dir = Path(__file__).resolve().parent.parent
        # If .git exists, use git command
        if (proj_dir / ".git").exists():
            hash_val = subprocess.check_output(
                ["git", "-C", str(proj_dir), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            return f"{version_str}+{hash_val}"
        
        # Fallback to git.hash file if it exists
        hash_file = proj_dir / "git.hash"
        if hash_file.exists():
            hash_val = hash_file.read_text().strip()
            return f"{version_str}+{hash_val}"
    # if we fail to get the hash, just return the version
    except Exception:
        pass
    return version_str
