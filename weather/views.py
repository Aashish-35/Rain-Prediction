from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import api_client
from .api_client import PredictionAPIError
from .forms import PredictionForm, SignUpForm
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
        try:
            result = api_client.predict(data)
        except PredictionAPIError as e:
            messages.error(request, str(e))
            return render(request, "predict.html", {"form": form})

        obj = Prediction.objects.create(
            user=request.user,
            **data,
            probability=result["probability"],
            threshold=result["threshold"],
            will_rain=result["will_rain"],
            explanation={
                "base_value": result["base_value"],
                "contributions": result["contributions"],
            },
        )
        return redirect("result", pk=obj.pk)
    return render(request, "predict.html", {"form": form})


@login_required
def result_view(request, pk):
    obj = get_object_or_404(Prediction, pk=pk, user=request.user)

    explain = None
    exp = obj.explanation or {}
    contribs = exp.get("contributions") or []
    if contribs:
        explain = {
            "base": round(exp.get("base_value", 0), 1),
            "final": obj.percent,
            "labels": [c["label"] for c in contribs],
            "values": [round(c["impact"], 1) for c in contribs],
            "items": contribs,
        }
    return render(request, "result.html", {"p": obj, "explain": explain})


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
    recent = list(qs[:20])[::-1]
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
def model_performance(request):
    info, cards = None, []
    try:
        info = api_client.model_info()
        t = info["test"]
        cards = [
            ("Accuracy", t["accuracy"], "bi-bullseye", "g1"),
            ("Precision", t["precision"], "bi-crosshair", "g2"),
            ("Recall", t["recall"], "bi-search", "g3"),
            ("F1 score", t["f1"], "bi-award", "g4"),
            ("ROC-AUC", t["roc_auc"], "bi-graph-up", "g1"),
            ("PR-AUC", t["pr_auc"], "bi-bar-chart-steps", "g2"),
        ]
    except PredictionAPIError as e:
        messages.error(request, str(e))
    return render(request, "model_performance.html", {"info": info, "cards": cards})