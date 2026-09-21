"""Step 3: build features, train + compare models, persist artifacts to ml/artifacts/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.models.train import run_training  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run_training()
    m = res["metrics"]
    print("selected:", m["selected_model"])
    print("final test:", {k: round(v, 4) for k, v in m["final_model_test"].items()})
