import importlib.resources as resources
import subprocess
from sys import argv


def run_lit():
    """Run the packaged LIT CLI script with forwarded command-line arguments.

    Args
    ----
    None

    Returns
    -------
    None
    """
    with resources.path("neurolit.scripts", "run_lit.sh") as script_path:
        subprocess.run(["bash", str(script_path)] + argv[1:], check=True)


if __name__ == "__main__":
    run_lit()

