from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "probability", "will_rain")
    list_filter = ("will_rain", "created_at")
    search_fields = ("user__username",)