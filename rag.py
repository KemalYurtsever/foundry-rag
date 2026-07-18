"""Run the project from a source checkout without installing build tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from foundry_rag.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
