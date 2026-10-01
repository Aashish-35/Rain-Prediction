from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .models import Prediction

VALID = {"temperature": 20, "humidity": 85, "wind_speed": 8,
         "cloud_cover": 80, "pressure": 1010}


class PredictFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("tester", password="pass12345!")

    def test_login_required(self):
        r = self.client.get(reverse("predict"))
        self.assertEqual(r.status_code, 302)

    @patch("weather.views.predict_rain", return_value=(0.72, True, 0.4))
    def test_prediction_saved(self, _mock):
        self.client.login(username="tester", password="pass12345!")
        r = self.client.post(reverse("predict"), VALID)
        self.assertEqual(r.status_code, 302)
        self.assertEqual(Prediction.objects.filter(user=self.user).count(), 1)

    def test_out_of_range_rejected(self):
        self.client.login(username="tester", password="pass12345!")
        r = self.client.post(reverse("predict"), {**VALID, "humidity": 500})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Prediction.objects.count(), 0)