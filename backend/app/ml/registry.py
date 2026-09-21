"""Loads the trained model exactly once at startup and exposes it to request handlers.

The API never trains: it only loads artifacts written by ``ml/scripts/03_train_models.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_settings
from app.core.errors import ModelUnavailableError

if str(PROJECT_ROOT) not in sys.path:  # make the shared `ml` package importable
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.forecaster import Forecaster, ModelNotAvailableError  # noqa: E402

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self.forecaster: Forecaster | None = None
        self.error: str | None = None
        self.feature_importance: dict[str, float] = {}

    def load(self, artifacts_dir: Path | None = None) -> None:
        d = artifacts_dir or get_settings().model_artifacts_dir
        try:
            self.forecaster = Forecaster(d)
            fi = d / "feature_importance.json"
            self.feature_importance = json.loads(fi.read_text()) if fi.exists() else {}
            self.error = None
        except ModelNotAvailableError as exc:
            self.forecaster = None
            self.error = str(exc)
            log.error("Model not loaded: %s", exc)

    @property
    def loaded(self) -> bool:
        return self.forecaster is not None

    def require(self) -> Forecaster:
        if self.forecaster is None:
            raise ModelUnavailableError(
                "The prediction model is not available. Train it with `make pipeline` and restart."
            )
        return self.forecaster
