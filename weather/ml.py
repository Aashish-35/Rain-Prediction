import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings

# Must match the training column names and order
FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]


@lru_cache(maxsize=1)
def _load():
    """Load the pipeline once per process."""
    model = joblib.load(settings.ML_MODEL_PATH)
    meta = {"threshold": 0.5}
    meta_path = Path(settings.ML_META_PATH)
    if meta_path.exists():
        meta.update(json.loads(meta_path.read_text()))
    return model, meta


def predict_rain(temperature, humidity, wind_speed, cloud_cover, pressure):
    """Returns (probability_of_rain, will_rain, threshold_used)."""
    model, meta = _load()
    row = pd.DataFrame(
        [[temperature, humidity, wind_speed, cloud_cover, pressure]],
        columns=FEATURES,
    )
    prob = float(model.predict_proba(row)[0, 1])   # SMOTE is skipped automatically at predict time
    threshold = float(meta["threshold"])
    return prob, prob >= threshold, threshold