"""Launch the Foundry RAG desktop application from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from foundry_rag.gui import main


if __name__ == "__main__":
    raise SystemExit(main())
