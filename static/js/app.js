// ---------- theme ----------
(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("theme");
  const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  const apply = (t) => { root.setAttribute("data-theme", t); root.setAttribute("data-bs-theme", t); };
  apply(saved || preferred);

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeToggle");
    const icon = btn && btn.querySelector("i");
    const sync = () => { if (icon) icon.className = root.getAttribute("data-theme") === "dark" ? "bi bi-sun" : "bi bi-moon-stars"; };
    sync();
    if (btn) btn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      localStorage.setItem("theme", next);
      apply(next); sync();
      document.dispatchEvent(new Event("themechange"));
    });
  });
})();

document.addEventListener("DOMContentLoaded", () => {
  // ---------- slider <-> number sync ----------
  document.querySelectorAll(".metric").forEach((m) => {
    const range = m.querySelector("input[type=range]");
    const num = m.querySelector("input[type=number]");
    if (!range || !num) return;
    range.value = num.value;
    range.addEventListener("input", () => (num.value = range.value));
    num.addEventListener("input", () => { if (num.value !== "") range.value = num.value; });
  });

  // ---------- presets ----------
  const presets = {
    sunny:  { temperature: 28, humidity: 45, wind_speed: 6,  cloud_cover: 15, pressure: 1022 },
    cloudy: { temperature: 21, humidity: 68, wind_speed: 9,  cloud_cover: 62, pressure: 1010 },
    stormy: { temperature: 16, humidity: 92, wind_speed: 14, cloud_cover: 90, pressure: 996 },
  };
  document.querySelectorAll("[data-preset]").forEach((b) => {
    b.addEventListener("click", () => {
      const p = presets[b.dataset.preset];
      Object.entries(p).forEach(([name, val]) => {
        const num = document.querySelector(`input[type=number][name=${name}]`);
        if (!num) return;
        num.value = val;
        num.dispatchEvent(new Event("input"));
      });
    });
  });

  // ---------- loading state on submit ----------
  document.querySelectorAll("form[data-loading]").forEach((f) => {
    f.addEventListener("submit", () => {
      const b = f.querySelector("button[type=submit]");
      if (b && f.checkValidity()) {
        b.disabled = true;
        b.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analysing…';
      }
    });
  });

  // ---------- animated gauge ----------
  const bar = document.querySelector(".gauge .bar");
  if (bar) {
    const pct = parseFloat(bar.dataset.percent) || 0;
    const full = 339.292;
    requestAnimationFrame(() => setTimeout(() => (bar.style.strokeDashoffset = full * (1 - pct / 100)), 100));
    const label = document.getElementById("gaugeNum");
    if (label) {
      let cur = 0; const step = pct / 45;
      const t = setInterval(() => {
        cur = Math.min(pct, cur + step);
        label.textContent = cur.toFixed(1) + "%";
        if (cur >= pct) clearInterval(t);
      }, 25);
    }
  }

  // ---------- history filter ----------
  const filters = document.querySelectorAll("[data-filter]");
  filters.forEach((f) => f.addEventListener("click", () => {
    filters.forEach((x) => x.classList.remove("active"));
    f.classList.add("active");
    const v = f.dataset.filter;
    document.querySelectorAll("#histBody tr[data-result]").forEach((r) => {
      r.style.display = v === "all" || r.dataset.result === v ? "" : "none";
    });
  }));
});