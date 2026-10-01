import json

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import PredictionForm, SignUpForm
from .ml import predict_rain
from .models import Prediction


def signup(request):
    if request.user.is_authenticated:
        return redirect("predict")
    form = SignUpForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Account created. Welcome!")
        return redirect("predict")
    return render(request, "registration/signup.html", {"form": form})


@login_required
def predict_view(request):
    form = PredictionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        prob, will_rain, threshold = predict_rain(**data)
        obj = Prediction.objects.create(
            user=request.user, **data,
            probability=prob, threshold=threshold, will_rain=will_rain,
        )
        return redirect("result", pk=obj.pk)
    return render(request, "predict.html", {"form": form})


@login_required
def result_view(request, pk):
    obj = get_object_or_404(Prediction, pk=pk, user=request.user)
    return render(request, "result.html", {"p": obj})


@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)
    page = Paginator(qs, 10).get_page(request.GET.get("page"))
    return render(request, "history.html", {"page": page})


@login_required
@require_POST
def delete_prediction(request, pk):
    get_object_or_404(Prediction, pk=pk, user=request.user).delete()
    messages.info(request, "Prediction deleted.")
    return redirect("history")


@login_required
def dashboard_view(request):
    qs = Prediction.objects.filter(user=request.user)
    stats = qs.aggregate(
        total=Count("id"),
        rain=Count("id", filter=Q(will_rain=True)),
        avg_prob=Avg("probability"),
    )
    recent = list(qs[:20])[::-1]   # oldest to newest for the chart
    chart = {
        "labels": [p.created_at.strftime("%d %b %H:%M") for p in recent],
        "probs": [p.percent for p in recent],
    }
    ctx = {
        "total": stats["total"],
        "rain": stats["rain"],
        "no_rain": stats["total"] - stats["rain"],
        "avg_prob": round((stats["avg_prob"] or 0) * 100, 1),
        "chart": chart,
    }
    return render(request, "dashboard.html", ctx)


@login_required
@require_POST
def api_predict(request):
    """JSON endpoint. Session-authenticated, so send the CSRF token with fetch()."""
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    form = PredictionForm(payload)
    if not form.is_valid():
        return JsonResponse({"errors": form.errors.get_json_data()}, status=400)

    data = form.cleaned_data
    prob, will_rain, threshold = predict_rain(**data)
    obj = Prediction.objects.create(
        user=request.user, **data,
        probability=prob, threshold=threshold, will_rain=will_rain,
    )
    return JsonResponse({
        "id": obj.pk,
        "probability": round(prob, 4),
        "will_rain": will_rain,
        "risk_level": obj.risk_level,
    })