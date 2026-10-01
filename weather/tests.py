from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .api_client import PredictionAPIError
from .models import Prediction

VALID = {"temperature": 20, "humidity": 85, "wind_speed": 8, "cloud_cover": 80, "pressure": 1010}
FAKE = {
    "probability": 0.72, "will_rain": True, "threshold": 0.4, "risk_level": "High", "base_value": 35.0,
    "contributions": [{"feature": "Humidity", "label": "Humidity", "value": 85.0, "unit": "%", "impact": 20.5}],
}


class PredictFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("tester", password="pass12345!")

    def test_login_required(self):
        self.assertEqual(self.client.get(reverse("predict")).status_code, 302)

    @patch("weather.views.api_client.predict", return_value=FAKE)
    def test_prediction_saved_with_explanation(self, _m):
        self.client.login(username="tester", password="pass12345!")
        r = self.client.post(reverse("predict"), VALID)
        self.assertEqual(r.status_code, 302)
        obj = Prediction.objects.get(user=self.user)
        self.assertEqual(obj.explanation["contributions"][0]["feature"], "Humidity")

    @patch("weather.views.api_client.predict", side_effect=PredictionAPIError("down"))
    def test_api_down_shows_error(self, _m):
        self.client.login(username="tester", password="pass12345!")
        r = self.client.post(reverse("predict"), VALID)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Prediction.objects.count(), 0)

    def test_out_of_range_rejected(self):
        self.client.login(username="tester", password="pass12345!")
        r = self.client.post(reverse("predict"), {**VALID, "humidity": 500})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Prediction.objects.count(), 0)