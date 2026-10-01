from django.conf import settings
from django.db import models


class Prediction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="predictions")
    temperature = models.FloatField()
    humidity = models.FloatField()
    wind_speed = models.FloatField()
    cloud_cover = models.FloatField()
    pressure = models.FloatField()

    probability = models.FloatField()
    threshold = models.FloatField()
    will_rain = models.BooleanField()
    explanation = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def percent(self):
        return round(self.probability * 100, 1)

    @property
    def risk_level(self):
        if self.probability >= 0.6:
            return "High"
        if self.probability >= self.threshold:
            return "Moderate"
        return "Low"

    def __str__(self):
        return f"{self.user} | {self.created_at:%Y-%m-%d %H:%M} | {self.percent}%"