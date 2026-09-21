"""Step 4: evaluation figures from saved artifacts (ml/artifacts/plots/)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.visualization.eval_plots import make_eval_plots  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    make_eval_plots()
