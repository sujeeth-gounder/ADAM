"""Root launcher entrypoint for ADAM Portable."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORTABLE_LAUNCHER = ROOT / "ADAM_PORTABLE" / "run_adam.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(PORTABLE_LAUNCHER)] + sys.argv[1:]))
