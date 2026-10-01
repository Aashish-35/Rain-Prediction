from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User


class BootstrapMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-control")


class LoginForm(BootstrapMixin, AuthenticationForm):
    pass


class SignUpForm(BootstrapMixin, UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


def _num(label, lo, hi, help_text=""):
    return forms.FloatField(
        label=label, min_value=lo, max_value=hi, help_text=help_text,
        widget=forms.NumberInput(attrs={"step": "any"}),
    )


class PredictionForm(BootstrapMixin, forms.Form):
    # Ranges match the training data, so predictions stay in-distribution
    temperature = _num("Temperature (°C)", 10, 35, "10 to 35")
    humidity = _num("Humidity (%)", 30, 100, "30 to 100")
    wind_speed = _num("Wind speed (km/h)", 0, 20, "0 to 20")
    cloud_cover = _num("Cloud cover (%)", 0, 100, "0 to 100")
    pressure = _num("Pressure (hPa)", 980, 1050, "980 to 1050")