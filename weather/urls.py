from django.contrib.auth import views as auth_views
from django.urls import path

from . import views
from .forms import LoginForm

urlpatterns = [
    path("", views.predict_view, name="predict"),
    path("result/<int:pk>/", views.result_view, name="result"),
    path("history/", views.history_view, name="history"),
    path("history/<int:pk>/delete/", views.delete_prediction, name="delete_prediction"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("api/predict/", views.api_predict, name="api_predict"),

    path("signup/", views.signup, name="signup"),
    path("login/", auth_views.LoginView.as_view(authentication_form=LoginForm), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]