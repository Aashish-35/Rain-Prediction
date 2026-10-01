import numpy as np

FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]

LABELS = {
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Wind_Speed": "Wind speed",
    "Cloud_Cover": "Cloud cover",
    "Pressure": "Pressure",
}
UNITS = {
    "Temperature": "°C",
    "Humidity": "%",
    "Wind_Speed": "km/h",
    "Cloud_Cover": "%",
    "Pressure": "hPa",
}


def rain_shap(explainer, X):
    """
    Return (shap_values_for_rain_class, base_value) for any shap version.
    For a Random Forest, values are in probability units, so
    base_value + sum(shap_values) ≈ predicted probability of rain.
    """
    sv = explainer.shap_values(X, check_additivity=False)
    ev = np.atleast_1d(explainer.expected_value)

    if isinstance(sv, list):                 # older shap: list of arrays, one per class
        return np.asarray(sv[1]), float(ev[1])

    sv = np.asarray(sv)
    if sv.ndim == 3:                         # newer shap: (rows, features, classes)
        return sv[:, :, 1], float(ev[1])
    return sv, float(ev[-1])