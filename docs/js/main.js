"use strict";

/* ============ generic tab groups ============ */
function activate(group, btn){
  group.querySelectorAll(".tab").forEach(b => b.classList.toggle("is-active", b === btn));
}

/* comparison model toggle (FE vs P-DivGNN / FE vs GNN) */
(function () {
  const bottom = document.getElementById("compareBottom");
  const tagRight = document.getElementById("compareTagRight");
  document.querySelectorAll('.tabs[data-group="cmp"] .tab').forEach(btn => {
    btn.addEventListener("click", () => {
      activate(btn.parentElement, btn);
      bottom.src = btn.dataset.cmp;
      tagRight.textContent = btn.dataset.label;
    });
  });
})();

/* refinement-image tabs */
const refineImg = document.getElementById("refineImg");
document.querySelectorAll('.tabs[data-group="refine"] .tab').forEach(btn => {
  btn.addEventListener("click", () => {
    activate(btn.parentElement, btn);
    refineImg.src = btn.dataset.img;
  });
});

/* ============ comparison divider ============ */
(function () {
  const root = document.getElementById("compare");
  if (!root) return;
  const top = document.getElementById("compareTop");
  const divider = document.getElementById("compareDivider");
  const range = document.getElementById("compareRange");

  function setPct(p) {
    p = Math.max(0, Math.min(100, p));
    top.style.clipPath = "inset(0 " + (100 - p) + "% 0 0)";
    divider.style.left = p + "%";
    range.value = String(p);
  }
  range.addEventListener("input", () => setPct(parseFloat(range.value)));

  // pointer drag anywhere on the image
  let dragging = false;
  function pctFromEvent(e) {
    const rect = root.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    return (x / rect.width) * 100;
  }
  root.addEventListener("pointerdown", e => { dragging = true; setPct(pctFromEvent(e)); });
  window.addEventListener("pointermove", e => { if (dragging) setPct(pctFromEvent(e)); });
  window.addEventListener("pointerup", () => { dragging = false; });

  setPct(50);
})();

/* ============ benchmark chart ============ */
(function () {
  const canvas = document.getElementById("benchChart");
  if (!canvas || typeof Chart === "undefined") return;

  const nodes = [100, 4196, 12342, 28805, 40494, 56206];
  const fe    = [2.379, 35.769, 98.149, 262.384, 326.295, 428.071];
  const gnn   = [0.012, 0.027, 0.058, 0.127, 0.176, 0.121];

  const pt = (xs, ys) => xs.map((x, i) => ({ x, y: ys[i] }));
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const bordeaux = css("--bordeaux") || "#8a1f3f";
  const tubaf = css("--tubaf") || "#0a4f63";

  const chart = new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        { label: "Finite elements (CPU)", data: pt(nodes, fe),
          borderColor: bordeaux, backgroundColor: bordeaux, tension: .2, pointRadius: 4 },
        { label: "LSTM-GNN (GPU)", data: pt(nodes, gnn),
          borderColor: tubaf, backgroundColor: tubaf, tension: .2, pointRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: { type: "logarithmic", title: { display: true, text: "Number of mesh nodes" } },
        y: { type: "logarithmic", title: { display: true, text: "Inference time (s)" } }
      },
      plugins: {
        legend: { labels: { boxWidth: 14, font: { size: 13 } } },
        tooltip: {
          callbacks: {
            label: c => `${c.dataset.label}: ${c.parsed.y < 1 ? c.parsed.y.toFixed(3) : c.parsed.y.toFixed(1)} s`,
            title: items => `${items[0].parsed.x.toLocaleString()} nodes`
          }
        }
      }
    }
  });

  document.querySelectorAll('.tabs[data-group="scale"] .tab').forEach(btn => {
    btn.addEventListener("click", () => {
      activate(btn.parentElement, btn);
      const t = btn.dataset.scale;            // "logarithmic" | "linear"
      chart.options.scales.x.type = t;
      chart.options.scales.y.type = t;
      chart.update();
    });
  });
})();

/* ============ GNN vs P-DivGNN validation-loss chart ============ */
(function () {
  const canvas = document.getElementById("lossChart");
  if (!canvas || typeof Chart === "undefined") return;
  const GNN_VAL=[0.050285, 0.020872, 0.019758, 0.020159, 0.015248, 0.019546, 0.013427, 0.013642, 0.016549, 0.012652, 0.01294, 0.016431, 0.013883, 0.01228, 0.012032, 0.012292, 0.011594, 0.011251, 0.011001, 0.011166, 0.010743, 0.010901, 0.010881, 0.010814, 0.011073, 0.011023, 0.010261, 0.011475, 0.010592, 0.010435, 0.010582, 0.011076, 0.01057, 0.01069, 0.010342, 0.010634, 0.012154, 0.010688, 0.010113, 0.010724, 0.010486, 0.01028, 0.010371, 0.010217, 0.009769, 0.010005, 0.010277, 0.010527, 0.010163, 0.010054, 0.010286, 0.0102, 0.010068, 0.009529, 0.010283, 0.009809, 0.009976, 0.009872, 0.009816, 0.009577, 0.010347, 0.010265, 0.009664, 0.009753, 0.009743, 0.00983, 0.010159, 0.010698, 0.009646, 0.009875, 0.009991, 0.01028, 0.009701, 0.009595, 0.00966, 0.009959, 0.00958, 0.009492, 0.009601, 0.009775, 0.009658, 0.009899, 0.009514, 0.00973, 0.009709, 0.00984, 0.01021, 0.009571, 0.010318, 0.009938, 0.009926, 0.009523, 0.009628, 0.01041, 0.009574, 0.00959, 0.010444, 0.009795, 0.009632, 0.009396];
  const PDIV_VAL=[0.028141, 0.019068, 0.034525, 0.014902, 0.036296, 0.02286, 0.012329, 0.016253, 0.01194, 0.012624, 0.011162, 0.010978, 0.009778, 0.024974, 0.009358, 0.009271, 0.009178, 0.008881, 0.008478, 0.009472, 0.007948, 0.011008, 0.009239, 0.008025, 0.008223, 0.007252, 0.007203, 0.007256, 0.008917, 0.006761, 0.006766, 0.007253, 0.007633, 0.006397, 0.006398, 0.006752, 0.005766, 0.006951, 0.005957, 0.008111, 0.005749, 0.005847, 0.00753, 0.005355, 0.005924, 0.005699, 0.005801, 0.005221, 0.005543, 0.004969, 0.005251, 0.005143, 0.005083, 0.004825, 0.004636, 0.005211, 0.005246, 0.005768, 0.004731, 0.004881, 0.004762, 0.004898, 0.004966, 0.004962, 0.00431, 0.004315, 0.004739, 0.004682, 0.004412, 0.005119, 0.004634, 0.005185, 0.004332, 0.004976, 0.004398, 0.004719, 0.004987, 0.004462, 0.005045, 0.003913, 0.0044, 0.004344, 0.005319, 0.004741, 0.005097, 0.004708, 0.004124, 0.004413, 0.00428, 0.00429, 0.004229, 0.004042, 0.004083, 0.004035, 0.004261, 0.003926, 0.004479, 0.003924, 0.004247, 0.004163];
  const epochs = GNN_VAL.map((_, i) => i + 1);
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const bordeaux = css("--bordeaux") || "#8a1f3f", tubaf = css("--tubaf") || "#0a4f63";
  new Chart(canvas, {
    type: "line",
    data: { labels: epochs, datasets: [
      { label: "GNN (baseline)", data: GNN_VAL, borderColor: tubaf, backgroundColor: tubaf, pointRadius: 0, borderWidth: 2, tension: .2 },
      { label: "P-DivGNN", data: PDIV_VAL, borderColor: bordeaux, backgroundColor: bordeaux, pointRadius: 0, borderWidth: 2, tension: .2 }
    ]},
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Epoch" }, ticks: { maxTicksLimit: 11 } },
        y: { type: "logarithmic", title: { display: true, text: "Validation NMSE" } }
      },
      plugins: {
        legend: { labels: { boxWidth: 14, font: { size: 13 } } },
        tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } }
      }
    }
  });
})();

/* ============ LSTM hidden-state linked viz ============ */
(function () {
  const D = window.HIDDEN_DATA;
  const curveEl = document.getElementById("hsCurve");
  if (!D || !curveEl || typeof Chart === "undefined") return;
  const n = D.steps;

  // viridis-like colormap: dark purple -> blue -> teal -> green -> yellow
  const stops = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function cmap(f) {
    f = Math.max(0, Math.min(1, f));
    const x = f * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), t = x - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
  }
  const colors = Array.from({ length: n }, (_, i) => cmap(i / (n - 1)));
  const COL = ["#1f77b4", "#ff7f0e", "#2ca02c"];
  // strain vs time-step
  const exx = D.strain.map((s, i) => ({ x: i, y: s[0] }));
  const eyy = D.strain.map((s, i) => ({ x: i, y: s[1] }));
  const exy = D.strain.map((s, i) => ({ x: i, y: s[2] }));
  const allE = D.strain.flat(), eMin = Math.min(...allE), eMax = Math.max(...allE);
  const pad = (eMax - eMin) * 0.06, Y0 = eMin - pad, Y1 = eMax + pad;
  const marker = i => [{ x: i, y: Y0 }, { x: i, y: Y1 }];
  // stress-strain hysteresis (σ_c vs ε_c), shown one component at a time
  const NM = ["xx", "yy", "xy"];
  const ss = c => D.strain.map((s, i) => ({ x: s[c], y: D.stress[i][c] }));
  const pcaPts = D.pca.map(p => ({ x: p[0], y: p[1] }));
  const tsnePts = D.tsne.map(p => ({ x: p[0], y: p[1] }));

  const HL = { pointBackgroundColor: "#fff", pointBorderColor: "#8a1f3f", pointBorderWidth: 3, pointRadius: 8 };
  const noAxis = { ticks: { display: false }, grid: { display: true, color: "rgba(0,0,0,0.07)" }, border: { display: false } };
  const base = { responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } } };

  function scatter(el, pts) {
    return new Chart(el, {
      type: "scatter",
      data: { datasets: [
        { data: pts, pointBackgroundColor: colors, pointRadius: 3, pointBorderWidth: 0 },
        { data: [pts[0]], ...HL }
      ]},
      options: { ...base, scales: { x: noAxis, y: noAxis } }
    });
  }

  let mode = "strain";
  const lineSet = (label, data, color, dash) => ({ label, data, borderColor: color, borderWidth: 2, borderDash: dash, pointRadius: 0 });
  const strainDatasets = () => [
    lineSet("εxx", exx, COL[0], [6, 3]), lineSet("εyy", eyy, COL[1], [6, 3]), lineSet("εxy", exy, COL[2], [6, 3]),
    { label: "current", data: marker(0), borderColor: "#8a1f3f", borderWidth: 2, pointRadius: 0 }
  ];
  const stressDatasets = c => [
    lineSet("σ" + NM[c], ss(c), COL[c]),
    { label: "current", type: "scatter", data: [{ x: D.strain[0][c], y: D.stress[0][c] }], pointBackgroundColor: "#8a1f3f", pointBorderColor: "#fff", pointBorderWidth: 1.5, pointRadius: 6, showLine: false }
  ];
  const curveOpts = (xt, yt) => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: true, labels: { boxWidth: 14, font: { size: 12 }, filter: it => it.text !== "current" } }, tooltip: { enabled: false } },
    scales: { x: { type: "linear", title: { display: true, text: xt }, ticks: { maxTicksLimit: 9 } },
              y: { title: { display: true, text: yt } } }
  });
  const curveChart = new Chart(curveEl, { type: "line", data: { datasets: strainDatasets() }, options: curveOpts("Time step", "Total strain ε") });
  const pcaChart = scatter(document.getElementById("hsPca"), pcaPts);
  const tsneChart = scatter(document.getElementById("hsTsne"), tsnePts);

  const slider = document.getElementById("hsSlider");
  const label = document.getElementById("hsLabel");
  const playBtn = document.getElementById("hsPlay");
  slider.max = String(n - 1);

  function setStep(i) {
    i = Math.max(0, Math.min(n - 1, i | 0));
    const last = curveChart.data.datasets.length - 1;
    curveChart.data.datasets[last].data = mode === "strain"
      ? marker(i) : [{ x: D.strain[i][+mode.slice(2)], y: D.stress[i][+mode.slice(2)] }];
    curveChart.update("none");
    pcaChart.data.datasets[1].data = [pcaPts[i]]; pcaChart.update("none");
    tsneChart.data.datasets[1].data = [tsnePts[i]]; tsneChart.update("none");
    slider.value = String(i); label.textContent = "step " + i;
  }
  slider.addEventListener("input", () => setStep(+slider.value));

  const cap = document.getElementById("hsCurveCap");
  document.querySelectorAll('.tabs[data-group="hs-curve"] .tab').forEach(btn => {
    btn.addEventListener("click", () => {
      activate(btn.parentElement, btn);
      mode = btn.dataset.hsmode;
      if (mode === "strain") {
        curveChart.data.datasets = strainDatasets();
        curveChart.options.scales.x.title.text = "Time step";
        curveChart.options.scales.y.title.text = "Total strain ε";
        if (cap) cap.innerHTML = "Eight-segment strain path (&epsilon;<sub>xx</sub>, &epsilon;<sub>yy</sub>, &epsilon;<sub>xy</sub>) &mdash; red line = current step";
      } else {
        const c = +mode.slice(2);
        curveChart.data.datasets = stressDatasets(c);
        curveChart.options.scales.x.title.text = "Strain ε" + NM[c];
        curveChart.options.scales.y.title.text = "Stress σ" + NM[c] + " [MPa]";
        if (cap) cap.innerHTML = "Stress-strain loop &sigma;<sub>" + NM[c] + "</sub> vs. &epsilon;<sub>" + NM[c] + "</sub> &mdash; red dot = current step";
      }
      curveChart.update();
      setStep(+slider.value);
    });
  });

  let timer = null;
  const stop = () => { if (timer) { clearInterval(timer); timer = null; playBtn.innerHTML = "&#9654;"; } };
  playBtn.addEventListener("click", () => {
    if (timer) { stop(); return; }
    playBtn.innerHTML = "&#10073;&#10073;";
    timer = setInterval(() => setStep((+slider.value + 1) % n), 55);
  });

  setStep(0);
})();

/* ============ copy BibTeX ============ */
(function () {
  const btn = document.getElementById("copyBib");
  const pre = document.getElementById("bibtex");
  if (!btn || !pre) return;
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      btn.textContent = "Copied";
      btn.classList.add("copied");
      setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600);
    } catch (e) {
      const r = document.createRange(); r.selectNode(pre);
      const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    }
  });
})();
