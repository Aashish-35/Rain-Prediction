import requests
from django.conf import settings


class PredictionAPIError(Exception):
    """Raised when the FastAPI service is unreachable or returns an error."""


def _request(method: str, path: str, **kwargs):
    url = f"{settings.FASTAPI_URL}{path}"
    try:
        r = requests.request(
            method, url,
            headers={"X-API-Key": settings.FASTAPI_API_KEY},
            timeout=settings.FASTAPI_TIMEOUT,
            **kwargs,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise PredictionAPIError(
            f"Prediction service returned {e.response.status_code}: {e.response.text[:200]}"
        ) from e
    except requests.RequestException as e:
        raise PredictionAPIError(
            "Prediction service is unreachable. Start it with: "
            "uvicorn api.main:app --port 8001"
        ) from e


def predict(data: dict) -> dict:
    return _request("POST", "/predict", json=data)


def model_info() -> dict:
    return _request("GET", "/model/info")


def health() -> dict:
    return _request("GET", "/health")