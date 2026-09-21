"""Step 2: data profile (docs/data_profile.json) + EDA figures (docs/figures/eda/)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.config import PROJECT_ROOT  # noqa: E402
from ml.src.visualization.eda_plots import make_figures, write_data_profile  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    skip_profile = "--skip-profile" in sys.argv
    if not skip_profile:
        write_data_profile(PROJECT_ROOT / "docs" / "data_profile.json")
    print(json.dumps(make_figures(), indent=2))
