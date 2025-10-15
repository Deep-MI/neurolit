import importlib.resources as resources
import subprocess
from sys import argv

def run_lit():
    with resources.path("LIT.scripts", "run_lit.sh") as script_path:
        subprocess.run(["bash", str(script_path)] + argv[1:])

