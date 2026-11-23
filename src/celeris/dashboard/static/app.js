/* Celeris planning control tower: single-page UI over the FastAPI backend. */
"use strict";

/* ------------------------------------------------------------------ config */
const ICONS = {
  overview: "M3 13h8V3H3zm0 8h8v-6H3zm10 0h8V11h-8zm0-18v6h8V3z",
  health: "M3 12h4l3-8 4 16 3-8h4",
  explorer: "M10 2a8 8 0 105.3 14l5.4 5.3 1.4-1.4-5.3-5.4A8 8 0 0010 2zm0 2a6 6 0 110 12 6 6 0 010-12z",
  anomalies: "M12 2L1 21h22L12 2zm0 5l7.5 12h-15L12 7zm-1 4v4h2v-4zm0 5v2h2v-2z",
  stockouts: "M3 3h18v4H3zm1 6h16v12H4zm5 3v2h6v-2z",
  drivers: "M4 20V10m6 10V4m6 16v-7m6 7V8",
  forecast: "M3 17l6-6 4 4 8-8M15 7h6v6",
  exceptions: "M4 4h16v2H4zm0 7h16v2H4zm0 7h10v2H4z",
  procurement: "M3 4h2l2.6 11.2a2 2 0 002 1.8h8.8a2 2 0 002-1.6L22 8H6",
  whatif: "M4 6h10M4 12h16M4 18h7M17 4v4M14 10v4M14 16v4",
  lifecycle: "M12 4a8 8 0 108 8h-2a6 6 0 11-6-6V2l4 3-4 3z",
  downloads: "M12 3v12m0 0l-5-5m5 5l5-5M4 21h16",
};
const VIEWS = [
  { id: "overview", group: "Overview", title: "Executive overview", part: "History + forecast", sub: "Units and revenue from history into the 19-month forecast, mix and headline numbers" },
  { id: "health", group: "What happened", title: "Data health", part: "Training data", sub: "Issues found in each file, rows affected and the data before vs after cleaning" },
  { id: "explorer", group: "What happened", title: "Demand explorer", part: "Training data", sub: "Drill down family → SKU → region → customer: trend, seasonality, zero months, lifecycle" },
  { id: "anomalies", group: "What happened", title: "Anomaly monitor", part: "Training data", sub: "Flagged points on the time series with type, reason and the action taken" },
  { id: "stockouts", group: "What happened", title: "Stockouts & lost sales", part: "Training data", opt: true, sub: "Supply-constrained months, unserved demand and why it happened" },
  { id: "drivers", group: "What happened", title: "Macro drivers", part: "Training data", opt: true, sub: "Which indicator leads which family, at what lag, and the shift to EV" },
  { id: "forecast", group: "What will happen", title: "Forecast & accuracy", part: "Prediction", sub: "Forecast vs history with P10–P90 ranges; backtest WAPE and bias by family, region and horizon" },
  { id: "exceptions", group: "What will happen", title: "Exception list", part: "Prediction", opt: true, sub: "Series a planner should look at first, ranked by revenue at stake" },
  { id: "procurement", group: "What will happen", title: "Procurement plan (P90)", part: "Prediction", opt: true, sub: "Order quantities that cover demand 9 times out of 10, and the cost of the buffer" },
  { id: "whatif", group: "What will happen", title: "What-if scenarios", part: "Prediction", opt: true, sub: "Shock a macro indicator and see the demand response by family" },
  { id: "lifecycle", group: "What will happen", title: "Lifecycle: launches & EOL", part: "Prediction", opt: true, sub: "End-of-life run-downs, new launches and the SKUs they replace" },
  { id: "downloads", group: "Data", title: "Data download", part: "Both", opt: true, sub: "Export what you see, with the current filters applied" },
];
const DIMS = ["family", "sku", "region", "channel", "customer"];
const DIM_LABEL = { family: "Family", sku: "SKU", region: "Region", channel: "Channel", customer: "Customer" };

const S = { view: "overview", f: { family: [], sku: [], region: [], channel: [], customer: [] }, p: {} };
let META = null;
let charts = [];
let token = 0;
const cache = new Map();

/* ------------------------------------------------------------------ utils */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const isNum = (v) => v !== null && v !== undefined && !Number.isNaN(v);
function fmtN(v, digits) {
  if (!isNum(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(digits ?? 2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(digits ?? 1) + "k";
  return Math.round(v).toLocaleString("en-US");
}
const fmtUSD = (v) => (isNum(v) ? (v < 0 ? "−$" : "$") + fmtN(Math.abs(v)) : "–");
const fmtInt = (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "–");
const fmtPct = (v, d = 1, sign = false) => (isNum(v) ? (sign && v > 0 ? "+" : "") + (v * 100).toFixed(d) + "%" : "–");
const fmtAcc = (w, d = 0) => (!isNum(w) ? "–" : w >= 1 ? "WAPE " + fmtPct(w, 0) : fmtPct(1 - w, d));
const delta = (v, d = 1) => (isNum(v) ? `<span class="${v >= 0 ? "delta-up" : "delta-down"}">${v >= 0 ? "▲" : "▼"} ${fmtPct(Math.abs(v), d)}</span>` : "–");
const pretty = (s) => String(s ?? "").replace(/_/g, " ");
const series = (i) => css(`--series-${i}`);
const sum = (a) => a.reduce((x, y) => x + (isNum(y) ? y : 0), 0);
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

function qs(extra = {}) {
  const q = new URLSearchParams();
  for (const d of DIMS) for (const v of S.f[d]) q.append(d, v);
  for (const [k, v] of Object.entries(extra)) if (v !== undefined && v !== null && v !== "") q.set(k, v);
  return q.toString();
}
async function api(path, extra = {}, withFilters = true) {
  const url = `/api/${path}?${withFilters ? qs(extra) : new URLSearchParams(extra).toString()}`;
  if (cache.has(url)) return cache.get(url);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const j = await r.json();
  if (cache.size > 200) cache.clear();
  cache.set(url, j);
  return j;
}

/* ------------------------------------------------------------------ state <-> URL */
function hashFor(view = S.view) {
  const q = new URLSearchParams(qs());
  for (const [k, v] of Object.entries(S.p)) if (v !== undefined && v !== null && v !== "") q.set("p_" + k, v);
  const s = q.toString();
  return `#/${view}${s ? "?" + s : ""}`;
}
function readHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [view, query] = h.split("?");
  S.view = VIEWS.some((v) => v.id === view) ? view : "overview";
  const q = new URLSearchParams(query || "");
  for (const d of DIMS) S.f[d] = q.getAll(d);
  S.p = {};
  for (const [k, v] of q.entries()) if (k.startsWith("p_")) S.p[k.slice(2)] = v;
}
function go(view = S.view) {
  const h = hashFor(view);
  if (location.hash === h) render(); else location.hash = h;
}
function setFilter(dim, values) { S.f[dim] = values; go(); }
function setParam(k, v) { S.p[k] = v; go(); }

/* ------------------------------------------------------------------ charts */
function disposeCharts() { charts.forEach((c) => c.dispose()); charts = []; }
function baseOption() {
  const tx = css("--text-secondary"), mu = css("--text-muted");
  return {
    backgroundColor: "transparent",
    color: [1, 2, 3, 4, 5, 6, 7, 8].map(series),
    textStyle: { color: tx, fontFamily: getComputedStyle(document.body).fontFamily },
    animationDuration: 350,
    tooltip: {
      backgroundColor: css("--surface-1"), borderColor: css("--border"), borderWidth: 1,
      textStyle: { color: css("--text-primary"), fontSize: 12 }, extraCssText: "box-shadow:0 6px 20px rgba(0,0,0,.18);border-radius:8px;",
      confine: true,
    },
    legend: { textStyle: { color: tx, fontSize: 12 }, icon: "roundRect", itemWidth: 14, itemHeight: 8, top: 0, inactiveColor: mu },
    grid: { left: 8, right: 16, top: 34, bottom: 8, containLabel: true },
  };
}
function axisCat(data, extra = {}) {
  return {
    type: "category", data, boundaryGap: true,
    axisLine: { lineStyle: { color: css("--border") } }, axisTick: { show: false },
    axisLabel: { color: css("--text-muted"), fontSize: 11 }, ...extra,
  };
}
function axisVal(extra = {}) {
  return {
    type: "value", axisLine: { show: false }, axisTick: { show: false },
    axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtN(v) },
    splitLine: { lineStyle: { color: css("--grid") } }, ...extra,
  };
}
function mount(id, option, onClick) {
  const el = document.getElementById(id);
  if (!el) return null;
  const c = echarts.init(el, null, { renderer: "canvas" });
  const b = baseOption();
  c.setOption({ ...b, ...option, tooltip: { ...b.tooltip, ...(option.tooltip || {}) }, legend: option.legend === false ? { show: false } : { ...b.legend, ...(option.legend || {}) }, grid: Array.isArray(option.grid) ? option.grid : { ...b.grid, ...(option.grid || {}) } });
  if (onClick) c.on("click", onClick);
  charts.push(c);
  return c;
}
window.addEventListener("resize", debounce(() => charts.forEach((c) => c.resize()), 120));

function ttRow(color, label, value) {
  return `<div style="display:flex;gap:8px;align-items:center;justify-content:space-between;min-width:180px">
    <span style="display:flex;gap:6px;align-items:center">${color ? `<i style="width:8px;height:8px;border-radius:2px;background:${color};display:inline-block"></i>` : ""}${esc(label)}</span><b>${value}</b></div>`;
}
const ttHead = (t) => `<div style="font-weight:600;margin-bottom:4px">${esc(t)}</div>`;

/* One continuous history + forecast timeline with the forecast zone clearly marked. */
function timelineOption(tl, metric = "units", opts = {}) {
  const H = tl.hist_months, F = tl.fc_months, nH = H.length, x = [...H, ...F];
  const rev = metric === "revenue";
  const hist = rev ? tl.revenue : tl.demand;
  const p50 = rev ? tl.rev_p50 : tl.p50, p10 = rev ? tl.rev_p10 : tl.p10, p90 = rev ? tl.rev_p90 : tl.p90;
  const nulF = Array(F.length).fill(null), nulH = Array(nH).fill(null);
  const c1 = series(1), c2 = series(2), mu = css("--muted-line");
  const fmt = rev ? fmtUSD : fmtN;
  const lastIdx = (() => { for (let i = nH - 1; i >= 0; i--) if (isNum(hist[i])) return i; return nH - 1; })();
  const p50S = [...Array(nH).fill(null), ...p50];
  p50S[lastIdx] = hist[lastIdx];
  const s = [
    { name: rev ? "Revenue (actual)" : "Demand (unconstrained)", type: "line", data: [...hist, ...nulF], showSymbol: false, lineStyle: { width: 2, color: c1 }, itemStyle: { color: c1 }, z: 3 },
  ];
  if (!rev && opts.shipped !== false) s.push({ name: "Shipped units", type: "line", data: [...tl.shipped, ...nulF], showSymbol: false, lineStyle: { width: 1.2, color: mu }, itemStyle: { color: mu }, z: 2 });
  s.push(
    { name: "_p10", type: "line", data: [...nulH, ...p10], stack: "band", lineStyle: { opacity: 0 }, showSymbol: false, silent: true, tooltip: { show: false } },
    { name: "P10–P90 range", type: "line", data: [...nulH, ...p90.map((v, i) => v - p10[i])], stack: "band", lineStyle: { opacity: 0 }, showSymbol: false, areaStyle: { color: c2, opacity: 0.2 }, itemStyle: { color: c2 }, silent: true },
    {
      name: "Forecast P50", type: "line", data: p50S, showSymbol: false, lineStyle: { width: 2.2, type: [6, 4], color: c2 }, itemStyle: { color: c2 }, z: 4,
      markArea: { silent: true, itemStyle: { color: css("--forecast-zone") }, label: { show: true, position: "insideTop", color: css("--text-muted"), fontSize: 11, fontWeight: 600, formatter: "FORECAST" }, data: [[{ xAxis: F[0] }, { xAxis: F[F.length - 1] }]] },
      markLine: { silent: true, symbol: "none", lineStyle: { color: css("--text-muted"), type: "solid", width: 1 }, label: { show: false }, data: [{ xAxis: H[nH - 1] }] },
    },
  );
  return {
    legend: { data: s.filter((z) => !z.name.startsWith("_")).map((z) => z.name) },
    grid: { bottom: opts.zoom === false ? 8 : 44 },
    tooltip: {
      trigger: "axis", axisPointer: { type: "line", lineStyle: { color: css("--text-muted") } },
      formatter: (ps) => {
        const i = ps[0].dataIndex;
        if (i < nH) {
          let out = ttHead(H[i] + " · actual") + ttRow(c1, rev ? "Revenue" : "Demand", fmt(hist[i]));
          if (!rev) out += ttRow(mu, "Shipped", fmt(tl.shipped[i]));
          return out;
        }
        const j = i - nH;
        return ttHead(F[j] + " · forecast") + ttRow(c2, "P50", fmt(p50[j])) + ttRow(null, "P10–P90", `${fmt(p10[j])} – ${fmt(p90[j])}`);
      },
    },
    xAxis: axisCat(x, { boundaryGap: false }),
    yAxis: axisVal({ scale: true, axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmt(v) } }),
    dataZoom: opts.zoom === false ? [] : [
      { type: "inside", startValue: opts.start || x[0] },
      { type: "slider", height: 18, bottom: 6, startValue: opts.start || x[0], borderColor: css("--border"), fillerColor: css("--accent-soft"), textStyle: { color: css("--text-muted") }, dataBackground: { lineStyle: { color: css("--muted-line") }, areaStyle: { opacity: 0 } } },
    ],
    series: s,
  };
}

function barH(names, seriesDefs, opts = {}) {
  return {
    legend: seriesDefs.length > 1 ? {} : false,
    grid: { left: 8, right: 24, top: seriesDefs.length > 1 ? 30 : 6, bottom: 4 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, ...(opts.tooltip || {}) },
    xAxis: axisVal({ splitNumber: 3, ...(opts.xAxis || {}), axisLabel: { color: css("--text-muted"), fontSize: 11, hideOverlap: true, formatter: (v) => fmtN(v), ...((opts.xAxis || {}).axisLabel || {}) } }),
    yAxis: axisCat(names, { inverse: true, axisLabel: { color: css("--text-secondary"), fontSize: 11.5, width: opts.labelWidth || 150, overflow: "truncate" } }),
    series: seriesDefs.map((s) => ({ type: "bar", barMaxWidth: 14, barGap: "20%", itemStyle: { borderRadius: [0, 4, 4, 0] }, ...s })),
  };
}

/* ------------------------------------------------------------------ html helpers */
function kpi(label, value, sub = "", title = "") {
  return `<div class="kpi" ${title ? `title="${esc(title)}"` : ""}><div class="k">${esc(label)}</div><div class="v">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
}
function card(title, body, { sub = "", tools = "", cls = "" } = {}) {
  return `<div class="card ${cls}"><div class="card-h"><h3 class="card-title">${esc(title)}</h3>${sub ? `<span class="card-sub">${sub}</span>` : ""}${tools ? `<div class="card-tools">${tools}</div>` : ""}</div>${body}</div>`;
}
const chartDiv = (id, cls = "") => `<div class="chart ${cls}" id="${id}"></div>`;
function seg(key, options, current) {
  return `<div class="seg" data-seg="${key}">${options.map(([v, l]) => `<button data-v="${esc(v)}" class="${String(v) === String(current) ? "on" : ""}">${esc(l)}</button>`).join("")}</div>`;
}
function bindSeg(root, handlers) {
  $$("[data-seg]", root).forEach((el) => {
    el.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b && handlers[el.dataset.seg]) handlers[el.dataset.seg](b.dataset.v);
    });
  });
}
function dlLink(kind, label = "Download CSV", extra = {}) {
  return `<a class="btn" href="/api/download/${kind}.csv?${qs(extra)}" download>⤓ ${esc(label)}</a>`;
}

/* Sortable table. cols: {k, label, fmt, num, html} */
function table(el, cols, rows, opts = {}) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el) return;
  let sortK = opts.sort || null, dir = opts.dir || -1;
  const draw = () => {
    let data = rows.slice();
    if (sortK) {
      data.sort((a, b) => {
        const x = a[sortK], y = b[sortK];
        if (x === y) return 0;
        if (x === null || x === undefined || x === "") return 1;
        if (y === null || y === undefined || y === "") return -1;
        return (x > y ? 1 : -1) * dir;
      });
    }
    const max = opts.max || 500;
    el.innerHTML = `<div class="tbl-wrap" style="${opts.height ? `max-height:${opts.height}px` : ""}"><table class="tbl"><thead><tr>${cols.map((c) => `<th data-k="${c.k}" class="${c.num ? "num" : ""}">${esc(c.label)} <span class="arrow">${sortK === c.k ? (dir > 0 ? "▲" : "▼") : ""}</span></th>`).join("")}</tr></thead><tbody>${
      data.slice(0, max).map((r, i) => `<tr data-i="${i}" class="${opts.onRow ? "click" : ""}">${cols.map((c) => `<td class="${c.num ? "num" : ""}${c.wrap ? " wrap" : ""}">${c.html ? c.html(r) : esc(c.fmt ? c.fmt(r[c.k], r) : r[c.k])}</td>`).join("")}</tr>`).join("")
    }</tbody></table></div>${rows.length === 0 ? `<div class="empty">Nothing in the current selection.</div>` : ""}${opts.foot ? `<div class="tbl-foot">${opts.foot}</div>` : ""}`;
    $$("th", el).forEach((th) => th.addEventListener("click", () => {
      const k = th.dataset.k;
      if (sortK === k) dir = -dir; else { sortK = k; dir = -1; }
      draw();
    }));
    if (opts.onRow) $$("tbody tr", el).forEach((tr) => tr.addEventListener("click", () => opts.onRow(data[+tr.dataset.i])));
  };
  draw();
}

/* ------------------------------------------------------------------ filters */
function optionsFor(dim) {
  const di = DIMS.indexOf(dim);
  const others = DIMS.map((d, i) => [i, new Set(S.f[d].map((v) => META.values[d].indexOf(v)))]).filter(([i, s]) => i !== di && s.size);
  const set = new Set();
  for (const r of META.rows) if (others.every(([i, s]) => s.has(r[i]))) set.add(r[di]);
  for (const v of S.f[dim]) set.add(META.values[dim].indexOf(v));
  return [...set].filter((i) => i >= 0).sort((a, b) => a - b).map((i) => META.values[dim][i]);
}
function filterSummary(dim) {
  const v = S.f[dim];
  if (!v.length) return "All";
  if (v.length === 1) return v[0];
  return `${v.length} selected`;
}
const applyFilters = debounce(() => go(), 450);
function renderFilters() {
  const el = $("#filters");
  const any = DIMS.some((d) => S.f[d].length);
  el.innerHTML = `<span class="lbl">Filters</span>` + DIMS.map((d) => `
    <div class="ms" data-dim="${d}">
      <button class="ms-btn ${S.f[d].length ? "on" : ""}" title="${esc(S.f[d].join(", "))}"><span class="k">${DIM_LABEL[d]}</span><span class="v">${esc(filterSummary(d))}</span><span class="caret">▾</span></button>
      <div class="ms-pop"></div>
    </div>`).join("") + (any ? `<button class="reset-btn" id="resetF">Reset all</button>` : "");
  $$(".ms", el).forEach((ms) => {
    const dim = ms.dataset.dim;
    $(".ms-btn", ms).addEventListener("click", (e) => {
      e.stopPropagation();
      const open = ms.classList.contains("open");
      closePops();
      if (!open) { ms.classList.add("open"); fillPop(ms, dim); }
    });
  });
  const r = $("#resetF");
  if (r) r.addEventListener("click", () => { DIMS.forEach((d) => (S.f[d] = [])); go(); });
}
function fillPop(ms, dim) {
  const pop = $(".ms-pop", ms);
  const opts = optionsFor(dim);
  const desc = META.sku_desc || {};
  pop.innerHTML = `<input type="search" placeholder="Search ${DIM_LABEL[dim].toLowerCase()}…" aria-label="Search">
    <div class="ms-tools"><span class="cnt"></span><span><button data-a="all">Select shown</button> · <button data-a="none">Clear</button></span></div>
    <div class="ms-list"></div>`;
  const list = $(".ms-list", pop), input = $("input", pop);
  pop.addEventListener("click", (e) => e.stopPropagation());
  let shown = [];
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    shown = opts.filter((o) => !q || o.toLowerCase().includes(q) || (dim === "sku" && (desc[o] || "").toLowerCase().includes(q)));
    const sel = new Set(S.f[dim]);
    list.innerHTML = shown.slice(0, 400).map((o) => `<label class="ms-opt"><input type="checkbox" value="${esc(o)}" ${sel.has(o) ? "checked" : ""}><span>${esc(o)}${dim === "sku" && desc[o] ? `<small>${esc(desc[o])}</small>` : ""}</span></label>`).join("") +
      (shown.length > 400 ? `<div class="ms-more">${shown.length - 400} more, type to narrow</div>` : "") + (shown.length ? "" : `<div class="ms-more">No match</div>`);
    $(".cnt", pop).textContent = `${shown.length} of ${opts.length}`;
  };
  draw();
  input.addEventListener("input", draw);
  setTimeout(() => input.focus(), 10);
  list.addEventListener("change", (e) => {
    const v = e.target.value;
    const s = new Set(S.f[dim]);
    e.target.checked ? s.add(v) : s.delete(v);
    S.f[dim] = [...s];
    updateMsBtn(ms, dim);
    applyFilters();
  });
  $$(".ms-tools button", pop).forEach((b) => b.addEventListener("click", () => {
    S.f[dim] = b.dataset.a === "all" ? [...new Set([...S.f[dim], ...shown])] : [];
    draw(); updateMsBtn(ms, dim); applyFilters();
  }));
}
function updateMsBtn(ms, dim) {
  const b = $(".ms-btn", ms);
  b.classList.toggle("on", S.f[dim].length > 0);
  $(".v", b).textContent = filterSummary(dim);
}
function closePops() { $$(".ms.open").forEach((m) => m.classList.remove("open")); }
document.addEventListener("click", () => closePops());

/* ------------------------------------------------------------------ nav + shell */
function renderNav() {
  let g = null, html = "";
  VIEWS.forEach((v, i) => {
    if (v.group !== g) { g = v.group; html += `<div class="nav-group">${esc(g)}</div>`; }
    html += `<button class="nav-item ${S.view === v.id ? "active" : ""}" data-v="${v.id}" title="${esc(v.sub)} (${i + 1 <= 9 ? "key " + (i + 1) : ""})">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${ICONS[v.id]}"/></svg>${esc(v.title)}${v.opt ? "" : `<span class="nav-tag">core</span>`}</button>`;
  });
  $("#nav").innerHTML = html;
  $$(".nav-item").forEach((b) => b.addEventListener("click", () => { document.body.classList.remove("nav-open"); go(b.dataset.v); }));
}
function effectiveDark() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
$("#themeBtn").addEventListener("click", () => {
  const next = effectiveDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("celeris-theme", next); } catch (e) { /* storage unavailable */ }
  render();
});
$("#menuBtn").addEventListener("click", (e) => { e.stopPropagation(); document.body.classList.toggle("nav-open"); });
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "/") { e.preventDefault(); $(".ms-btn")?.click(); return; }
  if (e.key === "Escape") closePops();
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= 9 && VIEWS[n - 1]) go(VIEWS[n - 1].id);
});
window.addEventListener("hashchange", () => { readHash(); render(); });

async function render() {
  const my = ++token;
  const v = VIEWS.find((x) => x.id === S.view);
  renderNav();
  renderFilters();
  $("#pageTitle").textContent = v.title;
  $("#pageSub").innerHTML = `<b style="color:var(--text-secondary)">${esc(v.part)}</b> · ${esc(v.sub)}`;
  document.title = `${v.title} · Celeris Control Tower`;
  disposeCharts();
  const C = $("#content");
  C.innerHTML = `<div class="kpis">${'<div class="skel" style="height:86px"></div>'.repeat(5)}</div><div class="skel" style="height:360px"></div>`;
  try {
    await RENDER[S.view](C, () => my === token);
  } catch (err) {
    if (my !== token) return;
    console.error(err);
    C.innerHTML = `<div class="card"><div class="empty">Could not load this view: ${esc(err.message)}</div></div>`;
  }
}

/* ------------------------------------------------------------------ 1. executive overview */
async function vOverview(C, alive) {
  const d = await api("overview");
  if (!alive()) return;
  const k = d.kpi, metric = S.p.metric || "units";
  const fam = d.mix.family.filter((r) => isNum(r.growth) && r.hist12 > 0);
  const up = fam.slice().sort((a, b) => b.growth - a.growth)[0], down = fam.slice().sort((a, b) => a.growth - b.growth)[0];
  const reg = d.mix.region.filter((r) => isNum(r.growth) && r.hist12 > 0).sort((a, b) => b.growth - a.growth)[0];
  const insight = `Over the next 19 months (${META.fc_months[0]} → ${META.fc_months.at(-1)}) the selection is forecast to need <b>${fmtN(k.fc19_units)} units</b>
    (<b>${fmtUSD(k.fc19_rev)}</b> revenue), ${delta(k.growth19)} vs the previous 19 months.
    ${up && fam.length > 1 ? `Fastest-growing family: <b>${esc(up.name)}</b> (${fmtPct(up.growth, 1, true)} next 12 vs last 12 months)${down && down.name !== up.name ? `; slowest: <b>${esc(down.name)}</b> (${fmtPct(down.growth, 1, true)})` : ""}.` : ""}
    ${reg && d.mix.region.length > 1 ? `Strongest region: <b>${esc(reg.name)}</b> (${fmtPct(reg.growth, 1, true)}).` : ""}
    In the backtest this model was <b>${fmtAcc(k.wape)}</b> accurate per SKU-month and <b>${fmtAcc(k.agg_wape, 1)}</b> for the selection total.`;
  C.innerHTML = `
    <div class="kpis">
      ${kpi("19-month demand forecast", fmtN(k.fc19_units), `P10–P90 ${fmtN(k.fc19_p10)} – ${fmtN(k.fc19_p90)}`, "Sum of monthly P50; range is the sum of monthly P10 / P90")}
      ${kpi("19-month revenue (P50)", fmtUSD(k.fc19_rev), `${delta(k.growth19)} units vs prior 19 months`)}
      ${kpi("Next 12 vs last 12 months", fmtPct(k.growth12, 1, true), `${fmtN(k.next12_units)} vs ${fmtN(k.last12_units)} units`)}
      ${kpi("Forecast accuracy (backtest)", fmtAcc(k.wape), `SKU-month · ${fmtAcc(k.agg_wape, 1)} at selection total · naive ${fmtAcc(k.naive_wape)}`, "1 − WAPE over 4 rolling origins, 19-month horizon")}
      ${kpi("Bias · P10–P90 coverage", fmtPct(k.bias, 1, true), `coverage ${fmtPct(k.coverage, 1)} (target 80%)`)}
      ${kpi("Active series", fmtInt(k.series), `${fmtInt(k.skus)} SKUs · ${k.new_launch_skus} launches · ${k.eol_skus} EOL`)}
    </div>
    <div class="insight">${insight}</div>
    ${card(metric === "revenue" ? "Revenue: history into forecast" : "Units: history into forecast", chartDiv("ov-tl", "tall") + `<div class="note">History is <b>unconstrained demand</b> (stockout months corrected to what customers asked for); shipped units shown for reference. Band: ${esc(d.timeline.band)}. Drag the slider to zoom.</div>`,
      { sub: "one continuous timeline, forecast shaded", tools: seg("metric", [["units", "Units"], ["revenue", "Revenue"]], metric) })}
    <div class="grid g3">
      ${card("Mix by family", chartDiv("ov-fam"), { sub: "last 12 months vs next 12 · click to filter" })}
      ${card("Mix by region", chartDiv("ov-reg"), { sub: "last 12 months vs next 12" })}
      ${card("Mix by channel", chartDiv("ov-ch"), { sub: "last 12 months vs next 12" })}
    </div>`;
  bindSeg(C, { metric: (v) => setParam("metric", v) });
  mount("ov-tl", timelineOption(d.timeline, metric, { start: "2021-01" }));
  const mixChart = (id, rows, dim) => {
    const revM = metric === "revenue";
    const names = rows.map((r) => r.name);
    mount(id, barH(names, [
      { name: "Last 12 months", data: rows.map((r) => (revM ? r.hist12_rev : r.hist12)), itemStyle: { color: series(1), borderRadius: [0, 4, 4, 0] } },
      { name: "Next 12 months (P50)", data: rows.map((r) => (revM ? r.fc12_rev : r.fc12)), itemStyle: { color: series(2), borderRadius: [0, 4, 4, 0] } },
    ], {
      labelWidth: 120,
      tooltip: { formatter: (ps) => { const r = rows[ps[0].dataIndex]; const f = revM ? fmtUSD : fmtN; return ttHead(r.name) + ttRow(series(1), "Last 12 months", f(revM ? r.hist12_rev : r.hist12)) + ttRow(series(2), "Next 12 months", f(revM ? r.fc12_rev : r.fc12)) + ttRow(null, "Change", fmtPct(r.growth, 1, true)) + ttRow(null, "Full 19 months", f(revM ? r.fc19_rev : r.fc19)); } },
      xAxis: revM ? { axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: fmtUSD } } : {},
    }), (p) => setFilter(dim, [names[p.dataIndex]]));
  };
  mixChart("ov-fam", d.mix.family, "family");
  mixChart("ov-reg", d.mix.region, "region");
  mixChart("ov-ch", d.mix.channel, "channel");
}

/* ------------------------------------------------------------------ 2. data health */
const FILE_ORDER = ["train.csv", "test.csv", "sku_master.csv", "inventory_stockouts.csv", "macro_indicators.csv", "events_calendar.csv"];
const fileColor = (f) => series([1, 2, 3, 4, 5, 7][Math.max(0, FILE_ORDER.indexOf(f))]);
async function vHealth(C, alive) {
  const d = await api("health");
  if (!alive()) return;
  const k = d.kpi;
  const issues = d.issues.slice().sort((a, b) => b.rows_affected - a.rows_affected);
  C.innerHTML = `
    <div class="kpis">
      ${kpi("Clean transaction rows", fmtInt(k.rows), "after de-duplication")}
      ${kpi("Rows corrected", fmtInt(k.rows_flagged), `${fmtPct(k.rows_flagged / Math.max(1, k.rows), 1)} of rows in selection`)}
      ${kpi("Issue types found", d.n_issue_types, "across 6 input files")}
      ${kpi("Missing months filled", fmtInt(k.missing_months), "imputed from the local level")}
      ${kpi("Raw vs clean units", fmtPct(k.raw_total / Math.max(1, k.clean_total) - 1, 1, true), `raw ${fmtN(k.raw_total)} → clean ${fmtN(k.clean_total)}`)}
      ${kpi("Demand vs shipped", fmtPct(k.demand_total / Math.max(1, k.clean_total) - 1, 1, true), "stockout correction adds back unmet demand")}
    </div>
    ${card("Before vs after cleaning", chartDiv("h-ba", "tall") + `<div class="note">Raw units include unit-of-measure errors (×10), sign flips, duplicated reloads and missing values. <b>Clean</b> = shipped units after fixes. <b>Demand</b> = clean units with stockout months corrected to requested units.</div>`, { sub: "monthly units for the current selection" })}
    <div class="grid g-2-1">
      ${card("Issues by type and file", chartDiv("h-iss", "tall"), { sub: "rows affected (whole dataset)" })}
      ${card("Rows per file", `<div id="h-rc"></div>`, { sub: "raw → clean" })}
    </div>
    ${card("Issue log", `<div id="h-tbl"></div>`, { sub: "every check, what it found and what was done" })}
    <div class="grid g2">
      ${card("Row-level corrections in selection", chartDiv("h-flags"), { sub: "rows changed, by fix" })}
      ${card("Where the demand numbers came from", chartDiv("h-src"), { sub: "rows by demand source, selection" })}
    </div>`;
  const m = d.months, c1 = series(1), c2 = series(2), mu = css("--muted-line");
  mount("h-ba", {
    legend: {},
    tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
    xAxis: axisCat(m, { boundaryGap: false }), yAxis: axisVal(),
    grid: { bottom: 44 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6, borderColor: css("--border"), fillerColor: css("--accent-soft") }],
    series: [
      { name: "Raw (as loaded)", type: "line", data: d.raw_units, showSymbol: false, lineStyle: { width: 1.2, color: mu }, itemStyle: { color: mu } },
      { name: "Clean shipped", type: "line", data: d.clean_units, showSymbol: false, lineStyle: { width: 2, color: c1 }, itemStyle: { color: c1 } },
      { name: "Unconstrained demand", type: "line", data: d.demand, showSymbol: false, lineStyle: { width: 2, color: c2, type: [6, 3] }, itemStyle: { color: c2 } },
    ],
  });
  const top = issues.slice(0, 22);
  const files = [...new Set(top.map((r) => r.file))];
  mount("h-iss", {
    ...barH(top.map((r) => pretty(r.issue_type)), [{ name: "Rows affected", data: top.map((r) => ({ value: r.rows_affected, itemStyle: { color: fileColor(r.file), borderRadius: [0, 4, 4, 0] } })) }], { labelWidth: 190 }),
    legend: false,
    tooltip: { trigger: "item", formatter: (p) => { const r = top[p.dataIndex]; return ttHead(pretty(r.issue_type)) + ttRow(fileColor(r.file), r.file, fmtInt(r.rows_affected) + " rows") + `<div style="max-width:280px;white-space:normal;margin-top:4px;color:${css("--text-secondary")}">${esc(r.action)}</div>`; } },
  });
  // coloured legend for files (identity not by colour alone: file is also in the tooltip and table)
  $("#h-iss").insertAdjacentHTML("afterend", `<div class="legend-inline">${files.map((f) => `<span><i class="sw box" style="background:${fileColor(f)}"></i>${esc(f)}</span>`).join("")}</div>`);
  table("h-rc", [
    { k: "file", label: "File" },
    { k: "raw_rows", label: "Raw", num: true, fmt: fmtInt },
    { k: "after_dedupe", label: "Deduped", num: true, fmt: fmtInt },
    { k: "clean_rows", label: "Clean", num: true, fmt: fmtInt },
  ], d.row_counts, { height: 360 });
  table("h-tbl", [
    { k: "file", label: "File" },
    { k: "issue_type", label: "Issue", fmt: pretty },
    { k: "category", label: "Category" },
    { k: "rows_affected", label: "Rows", num: true, fmt: fmtInt },
    { wrap: true, k: "action", label: "Action taken" },
    { wrap: true, k: "detail", label: "Detail" },
  ], issues, { sort: "rows_affected", height: 380 });
  const fl = d.row_flags.slice(0, 15);
  mount("h-flags", barH(fl.map((r) => pretty(r.flag)), [{ name: "Rows", data: fl.map((r) => r.rows), itemStyle: { color: c1, borderRadius: [0, 4, 4, 0] } }], { labelWidth: 160 }));
  const ds = d.demand_source;
  mount("h-src", barH(ds.map((r) => pretty(r.source)), [{ name: "Rows", data: ds.map((r) => r.rows), itemStyle: { color: series(3), borderRadius: [0, 4, 4, 0] } }], { labelWidth: 170, xAxis: { type: "log", axisLabel: { color: css("--text-muted"), formatter: (v) => fmtN(v) } } }));
}

/* ------------------------------------------------------------------ 3. demand explorer */
const DRILL = ["family", "sku", "region", "customer"];
async function vExplorer(C, alive) {
  const d = await api("explorer");
  if (!alive()) return;
  const lvlDim = d.level;
  const crumbs = [`<button data-up="0">All families</button>`];
  DRILL.forEach((dim, i) => {
    if (S.f[dim].length) crumbs.push(`<span class="sep">›</span>` + (i < DRILL.length - 1 ? `<button data-up="${i + 1}">${esc(S.f[dim].join(", "))}</button>` : `<b>${esc(S.f[dim].join(", "))}</b>`));
  });
  const k = d.kpi;
  const yoy = k.prev12 > 0 ? k.last12 / k.prev12 - 1 : null;
  const br = d.breakdown.slice(0, 25);
  C.innerHTML = `
    <div class="card"><div class="crumbs"><span style="color:var(--text-muted)">Drill path:</span> ${crumbs.join(" ")} <span class="sep">›</span> <span style="color:var(--text-muted)">${esc(DIM_LABEL[lvlDim])} (click a bar)</span></div></div>
    <div class="kpis">
      ${kpi("Demand, last 12 months", fmtN(k.last12), `${delta(yoy)} vs prior 12`)}
      ${kpi("Series (SKU × region)", fmtInt(k.series), `${fmtInt(k.skus)} SKUs`)}
      ${kpi("Zero-demand months", fmtPct(k.zero_share, 1), "share of series-months with no demand")}
      ${kpi(`${DIM_LABEL[lvlDim]} items`, fmtInt(d.n_items), "at the next drill level")}
    </div>
    <div class="grid g-1-2">
      ${card(`Demand by ${lvlDim === "sku" ? "SKU" : DIM_LABEL[lvlDim].toLowerCase()}`, chartDiv("e-br", "tall"), { sub: "last 12 months · click to drill down" })}
      ${card("Trend", chartDiv("e-tr", "tall"), { sub: "monthly demand with 12-month average" })}
    </div>
    <div class="grid g-2-1">
      ${card("Seasonality", chartDiv("e-heat", "tall"), { sub: "demand by year × month" })}
      ${card("Seasonal index", chartDiv("e-si", "tall"), { sub: "month ÷ year average (COVID years left out)" })}
    </div>
    <div class="grid g3">
      ${card("Zero months over time", chartDiv("e-zero"), { sub: "share of series with zero demand" })}
      ${card("Demand pattern", chartDiv("e-int"), { sub: "series by intermittency class (ADI / CV²)" })}
      ${card("Lifecycle", chartDiv("e-life"), { sub: "SKUs by status" })}
    </div>
    ${card(`${DIM_LABEL[lvlDim]} detail`, `<div id="e-tbl"></div>`)}
    ${d.lifecycle_rows.length ? card("Lifecycle events in selection", `<div id="e-lc"></div>`, { sub: "discontinued, EOL announced, planned launches and replacements" }) : ""}`;
  $$(".crumbs [data-up]", C).forEach((b) => b.addEventListener("click", () => {
    const n = +b.dataset.up;
    DRILL.forEach((dim, i) => { if (i >= n) S.f[dim] = []; });
    go();
  }));
  const drill = (name) => { if (lvlDim) setFilter(lvlDim, [name]); };
  mount("e-br", barH(br.map((r) => (lvlDim === "sku" && r.desc ? `${r.name} · ${r.desc}` : r.name)), [{ name: "Last 12 months", data: br.map((r) => r.last12), itemStyle: { color: series(1), borderRadius: [0, 4, 4, 0] } }], {
    labelWidth: 170,
    tooltip: { formatter: (ps) => { const r = br[ps[0].dataIndex]; return ttHead(r.name) + (r.desc ? `<div style="color:${css("--text-muted")};margin-bottom:4px">${esc(r.desc)}</div>` : "") + ttRow(series(1), "Last 12 months", fmtN(r.last12)) + ttRow(null, "vs prior 12", fmtPct(r.yoy, 1, true)) + ttRow(null, "Zero months", fmtPct(r.zero_share, 0)); } },
  }), (p) => drill(br[p.dataIndex].name));
  mount("e-tr", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
    xAxis: axisCat(d.months, { boundaryGap: false }), yAxis: axisVal(), grid: { bottom: 44 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6, borderColor: css("--border"), fillerColor: css("--accent-soft") }],
    series: [
      { name: "Demand", type: "line", data: d.demand, showSymbol: false, lineStyle: { width: 2, color: series(1) }, itemStyle: { color: series(1) } },
      { name: "Shipped", type: "line", data: d.shipped, showSymbol: false, lineStyle: { width: 1.2, color: css("--muted-line") }, itemStyle: { color: css("--muted-line") } },
      { name: "12-month average", type: "line", data: d.ma12, showSymbol: false, lineStyle: { width: 2, color: series(2) }, itemStyle: { color: series(2) } },
    ],
  });
  const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const hv = d.heat.filter((r) => isNum(r[2])).map((r) => r[2]);
  mount("e-heat", {
    legend: false,
    tooltip: { trigger: "item", formatter: (p) => ttHead(`${mon[p.value[0]]} ${d.years[p.value[1]]}`) + ttRow(null, "Demand", fmtN(p.value[2])) },
    grid: { left: 8, right: 70, top: 8, bottom: 8, containLabel: true },
    xAxis: axisCat(mon, { splitArea: { show: false } }), yAxis: axisCat(d.years, { inverse: true }),
    visualMap: { min: Math.min(...hv), max: Math.max(...hv), calculable: true, orient: "vertical", right: 0, top: "middle", itemHeight: 140, textStyle: { color: css("--text-muted"), fontSize: 10 }, formatter: (v) => fmtN(v), inRange: { color: [css("--seq-lo"), css("--seq-hi")] } },
    series: [{ type: "heatmap", data: d.heat.filter((r) => isNum(r[2])), itemStyle: { borderColor: css("--surface-1"), borderWidth: 2, borderRadius: 3 }, emphasis: { itemStyle: { borderColor: css("--text-primary") } } }],
  });
  mount("e-si", {
    legend: false, tooltip: { trigger: "axis", valueFormatter: (v) => (isNum(v) ? v.toFixed(2) : "–") },
    xAxis: axisCat(mon), yAxis: axisVal({ min: (v) => Math.min(0.6, Math.floor(v.min * 10) / 10), axisLabel: { color: css("--text-muted"), fontSize: 11 } }),
    grid: { top: 10 },
    series: [{ type: "bar", data: d.season_idx.map((v) => ({ value: v, itemStyle: { color: v >= 1 ? series(1) : css("--muted-line"), borderRadius: [4, 4, 0, 0] } })), barMaxWidth: 18,
      markLine: { silent: true, symbol: "none", data: [{ yAxis: 1 }], lineStyle: { color: css("--text-muted"), type: "dashed" }, label: { formatter: "avg", color: css("--text-muted") } } }],
  });
  mount("e-zero", {
    legend: false, tooltip: { trigger: "axis", valueFormatter: (v) => fmtPct(v, 1) },
    xAxis: axisCat(d.months, { boundaryGap: false }), yAxis: axisVal({ axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } }), grid: { top: 10 },
    series: [{ name: "Zero share", type: "line", data: d.zero_share_month, showSymbol: false, lineStyle: { width: 2, color: series(1) }, areaStyle: { color: series(1), opacity: 0.1 } }],
  });
  const ic = d.intermittency;
  mount("e-int", barH(ic.map((r) => r.cls), [{ name: "Series", data: ic.map((r) => r.n), itemStyle: { color: series(3), borderRadius: [0, 4, 4, 0] } }], { labelWidth: 90 }));
  const ls = d.lifecycle_status;
  mount("e-life", barH(ls.map((r) => r.status), [{ name: "SKUs", data: ls.map((r) => r.n), itemStyle: { color: series(7), borderRadius: [0, 4, 4, 0] } }], { labelWidth: 110, xAxis: { type: "log", axisLabel: { color: css("--text-muted"), formatter: (v) => fmtN(v) } } }));
  table("e-tbl", [
    { k: "name", label: DIM_LABEL[lvlDim] },
    ...(lvlDim === "sku" ? [{ k: "desc", label: "Description" }] : []),
    { k: "last12", label: "Last 12 mo", num: true, fmt: fmtInt },
    { k: "prev12", label: "Prior 12 mo", num: true, fmt: fmtInt },
    { k: "yoy", label: "Change", num: true, fmt: (v) => fmtPct(v, 1, true) },
    { k: "zero_share", label: "Zero months", num: true, fmt: (v) => fmtPct(v, 0) },
  ], d.breakdown, { sort: "last12", height: 340, onRow: (r) => drill(r.name), foot: `<span>Click a row to drill down.</span>` });
  if (d.lifecycle_rows.length) table("e-lc", [
    { k: "SKU_ID", label: "SKU" }, { wrap: true, k: "SKU_Description", label: "Description" }, { k: "Powertrain_Application", label: "Powertrain" },
    { k: "Launch_Date", label: "Launch" }, { k: "EOL_Date", label: "EOL" }, { k: "Replaces_SKU", label: "Replaces" }, { k: "Lifecycle_Status", label: "Status" },
  ], d.lifecycle_rows, { height: 260 });
}

/* ------------------------------------------------------------------ 4. anomaly monitor */
const CAT_ORDER = ["stockout", "error", "real_event", "suspected", "data_gap"];
const CAT_STYLE = {
  stockout: { slot: 2, symbol: "triangle", label: "Stockout" },
  error: { slot: 8, symbol: "rect", label: "Data error (fixed)" },
  real_event: { slot: 3, symbol: "circle", label: "Real event" },
  suspected: { slot: 4, symbol: "diamond", label: "Suspected (review)" },
  data_gap: { slot: 7, symbol: "roundRect", label: "Data gap (imputed)" },
};
async function vAnomalies(C, alive) {
  const cat = S.p.acat || "", atype = S.p.atype || "";
  const d = await api("anomalies", { category: cat, atype });
  if (!alive()) return;
  const counts = Object.fromEntries(d.categories.map((r) => [r.category, r.n]));
  const total = sum(d.categories.map((r) => r.n));
  const types = d.types.filter((t) => !cat || t.category === cat);
  C.innerHTML = `
    <div class="card"><div class="card-h"><h3 class="card-title">Categories</h3><span class="card-sub">${fmtInt(total)} flags in selection · click to filter the chart and table</span></div>
      <div class="chips">
        <button class="chip ${!cat ? "on" : ""}" data-c="">All <span class="n">${fmtInt(total)}</span></button>
        ${CAT_ORDER.filter((c) => counts[c]).map((c) => `<button class="chip ${cat === c ? "on" : ""}" data-c="${c}"><i class="dot" style="background:${series(CAT_STYLE[c].slot)}"></i>${CAT_STYLE[c].label} <span class="n">${fmtInt(counts[c])}</span></button>`).join("")}
      </div></div>
    ${card("Flagged points on the time series", chartDiv("a-ts", "xtall") + `<div class="note">Top: demand and shipped units. Bottom: one lane per flag category at the month it was flagged; marker size grows with the number of flags. Dashed verticals mark family-level structural breaks. Hover any month for counts and the main type.</div>`, { sub: "demand and shipped units, selection" })}
    <div class="grid g-2-1">
      ${card("Flag register", `<div id="a-tbl"></div>`, { sub: `${fmtInt(d.table_total)} rows${cat ? " · " + esc(CAT_STYLE[cat]?.label || cat) : ""}${atype ? " · " + esc(pretty(atype)) : ""}`,
        tools: `<select class="sel" id="a-type"><option value="">All types</option>${types.map((t) => `<option value="${esc(t.anomaly_type)}" ${t.anomaly_type === atype ? "selected" : ""}>${esc(pretty(t.anomaly_type))} (${t.n})</option>`).join("")}</select> ${dlLink("anomalies")}` })}
      ${card("By type", chartDiv("a-types", "tall"), { sub: "flags in selection" })}
    </div>
    ${card("Events calendar in scope", `<div id="a-ev"></div>`, { sub: "global events plus those for the selected regions and families" })}`;
  $$(".chip[data-c]", C).forEach((b) => b.addEventListener("click", () => { S.p.atype = ""; setParam("acat", b.dataset.c); }));
  $("#a-type").addEventListener("change", (e) => setParam("atype", e.target.value));
  const lanes = CAT_ORDER.filter((c) => (!cat || c === cat) && counts[c]);
  const laneLabels = lanes.map((c) => CAT_STYLE[c].label);
  const sc = lanes.map((c) => ({
    name: CAT_STYLE[c].label, type: "scatter", symbol: CAT_STYLE[c].symbol, xAxisIndex: 1, yAxisIndex: 1, z: 5,
    data: d.marks.filter((m) => m.category === c).map((m) => ({ value: [m.Date, CAT_STYLE[c].label], n: m.n, top: m.top })),
    symbolSize: (v, p) => 5 + 2 * Math.log2(1 + p.data.n),
    itemStyle: { color: series(CAT_STYLE[c].slot), borderColor: css("--surface-1"), borderWidth: 1 },
  }));
  const L = 150;
  mount("a-ts", {
    legend: { data: ["Demand", "Shipped"] },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    grid: [{ left: L, right: 16, top: 30, height: "48%" }, { left: L, right: 16, top: "62%", bottom: 46 }],
    tooltip: {
      trigger: "axis", axisPointer: { type: "line", lineStyle: { color: css("--text-muted") } },
      formatter: (ps) => {
        const i = ps[0].dataIndex, m = ps[0].axisValue;
        let out = ttHead(m);
        const mi = d.months.indexOf(m);
        out += ttRow(series(1), "Demand", fmtN(d.demand[mi])) + ttRow(css("--muted-line"), "Shipped", fmtN(d.shipped[mi]));
        d.marks.filter((x) => x.Date === m && lanes.includes(x.category)).forEach((x) => { out += ttRow(series(CAT_STYLE[x.category].slot), CAT_STYLE[x.category].label, `${x.n} · ${pretty(x.top)}`); });
        const br = d.breaks.filter((b) => b.Date === m);
        br.forEach((b) => { out += ttRow(null, "Break: " + b.Product_Family, (b.change_pct > 0 ? "+" : "") + b.change_pct + "%"); });
        return i === undefined ? "" : out;
      },
    },
    xAxis: [axisCat(d.months, { gridIndex: 0, boundaryGap: false, axisLabel: { show: false } }), axisCat(d.months, { gridIndex: 1, boundaryGap: false })],
    yAxis: [axisVal({ gridIndex: 0 }), axisCat(laneLabels, { gridIndex: 1, inverse: true, boundaryGap: true, splitLine: { show: true, lineStyle: { color: css("--grid") } }, axisLabel: { color: css("--text-secondary"), fontSize: 11.5 } })],
    dataZoom: [{ type: "inside", xAxisIndex: [0, 1] }, { type: "slider", xAxisIndex: [0, 1], height: 18, bottom: 6, borderColor: css("--border"), fillerColor: css("--accent-soft") }],
    series: [
      { name: "Demand", type: "line", data: d.demand, showSymbol: false, lineStyle: { width: 2, color: series(1) }, itemStyle: { color: series(1) }, z: 2,
        markLine: { silent: true, symbol: "none", lineStyle: { color: css("--text-muted"), type: "dashed", width: 1 }, label: { show: false }, data: d.breaks.map((b) => ({ xAxis: b.Date })) } },
      { name: "Shipped", type: "line", data: d.shipped, showSymbol: false, lineStyle: { width: 1.2, color: css("--muted-line") }, itemStyle: { color: css("--muted-line") }, z: 1 },
      ...sc,
    ],
  });
  const catTag = (c) => `<span class="prio"><i style="background:${series(CAT_STYLE[c]?.slot || 1)}"></i>${esc(CAT_STYLE[c]?.label || c)}</span>`;
  table("a-tbl", [
    { k: "Date", label: "Month" }, { k: "SKU_ID", label: "SKU" }, { k: "Region", label: "Region" }, { k: "Customer_Name", label: "Customer" },
    { k: "anomaly_type", label: "Type", fmt: pretty }, { k: "category", label: "Category", html: (r) => catTag(r.category) },
    { k: "value", label: "Value", num: true, fmt: fmtInt }, { k: "expected", label: "Expected", num: true, fmt: fmtInt },
    { wrap: true, k: "reason", label: "Reason" }, { wrap: true, k: "action", label: "Action" },
  ], d.table, { height: 430, foot: d.table_total > d.table.length ? `<span>Showing latest ${d.table.length} of ${fmtInt(d.table_total)}; download for all.</span>` : "" });
  const tt = types.slice(0, 18);
  mount("a-types", barH(tt.map((t) => pretty(t.anomaly_type)), [{ name: "Flags", data: tt.map((t) => ({ value: t.n, itemStyle: { color: series(CAT_STYLE[t.category]?.slot || 1), borderRadius: [0, 4, 4, 0] } })) }], { labelWidth: 170 }),
    (p) => setParam("atype", tt[p.dataIndex].anomaly_type));
  table("a-ev", [
    { k: "Event_ID", label: "ID" }, { k: "Start_Month", label: "From" }, { k: "End_Month", label: "To" }, { k: "Event_Type", label: "Type" },
    { k: "Scope_Type", label: "Scope" }, { k: "Scope_Value", label: "Applies to" }, { wrap: true, k: "Description", label: "Description" },
  ], d.events, { height: 260, sort: "Start_Month" });
}

/* ------------------------------------------------------------------ 5. stockouts */
async function vStockouts(C, alive) {
  const d = await api("stockouts");
  if (!alive()) return;
  const k = d.kpi;
  C.innerHTML = `
    <div class="kpis">
      ${kpi("Stockout months", fmtInt(k.events), `${fmtInt(k.last12_events)} in the last 12 months`)}
      ${kpi("Lost sales (est.)", fmtN(k.lost_units) + " units", `≈ ${fmtUSD(k.lost_value)} revenue`)}
      ${kpi("Fill rate", fmtPct(k.fill_rate, 1), "shipped ÷ unconstrained demand")}
      ${kpi("Demand above shipments", fmtPct(k.uplift, 2, true), "what the forecast adds back")}
    </div>
    <div class="grid g2">
      ${card("Lost and back-ordered units", chartDiv("s-lost"), { sub: "monthly, selection" })}
      ${card("Fill rate", chartDiv("s-fill"), { sub: "shipped ÷ demand, monthly" })}
    </div>
    ${card("Stockout months: listed vs inferred", chartDiv("s-ev", "short"), { sub: "listed in the inventory file, or inferred from a shipment collapse with no explanation" })}
    <div class="grid g3">
      ${card("By reason", chartDiv("s-reason"), { sub: "lost units, listed stockouts" })}
      ${card("By family", chartDiv("s-fam"), { sub: "lost units" })}
      ${card("By region", chartDiv("s-reg"), { sub: "lost units" })}
    </div>
    ${card("Most affected SKU × region", `<div id="s-top"></div>`, { sub: "top 50 by lost units" })}`;
  mount("s-lost", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) }, xAxis: axisCat(d.months), yAxis: axisVal(), grid: { bottom: 44 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6, borderColor: css("--border"), fillerColor: css("--accent-soft") }],
    series: [
      { name: "Lost (est.)", type: "bar", stack: "u", data: d.lost, itemStyle: { color: series(2) } },
      { name: "Back-ordered", type: "bar", stack: "u", data: d.back, itemStyle: { color: series(4) } },
    ],
  });
  mount("s-fill", {
    legend: false, tooltip: { trigger: "axis", valueFormatter: (v) => fmtPct(v, 1) }, xAxis: axisCat(d.months, { boundaryGap: false }),
    yAxis: axisVal({ min: (v) => Math.max(0, Math.floor(v.min * 50) / 50), max: 1, axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } }), grid: { bottom: 44, top: 10 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6, borderColor: css("--border"), fillerColor: css("--accent-soft") }],
    series: [{ name: "Fill rate", type: "line", data: d.fill_rate, showSymbol: false, lineStyle: { width: 2, color: series(1) } }],
  });
  mount("s-ev", {
    legend: {}, tooltip: { trigger: "axis" }, xAxis: axisCat(d.months), yAxis: axisVal(),
    series: [
      { name: "Listed", type: "bar", stack: "e", data: d.listed, itemStyle: { color: series(1) } },
      { name: "Inferred", type: "bar", stack: "e", data: d.inferred, itemStyle: { color: series(3) } },
    ],
  });
  const hb = (id, rows, key, slot) => mount(id, barH(rows.map((r) => r[key]), [{ name: "Lost units", data: rows.map((r) => r.lost), itemStyle: { color: series(slot), borderRadius: [0, 4, 4, 0] } }], {
    labelWidth: 150, tooltip: { formatter: (ps) => { const r = rows[ps[0].dataIndex]; return ttHead(r[key]) + ttRow(series(slot), "Lost units", fmtN(r.lost)) + ttRow(null, "Stockout months", fmtInt(r.events)) + (isNum(r.value) ? ttRow(null, "Lost revenue", fmtUSD(r.value)) : ""); } },
  }));
  hb("s-reason", d.reason, "Reason", 2);
  hb("s-fam", d.family.sort((a, b) => b.lost - a.lost), "Product_Family", 1);
  hb("s-reg", d.region.sort((a, b) => b.lost - a.lost), "Region", 3);
  table("s-top", [
    { k: "SKU_ID", label: "SKU" }, { k: "Region", label: "Region" }, { k: "Product_Family", label: "Family" },
    { k: "months", label: "Stockout months", num: true }, { k: "lost", label: "Lost units", num: true, fmt: fmtInt },
    { k: "value", label: "Lost revenue", num: true, fmt: fmtUSD }, { k: "last", label: "Last stockout" }, { k: "reason", label: "Main reason" },
  ], d.top, { sort: "lost", height: 360, onRow: (r) => { S.f.sku = [r.SKU_ID]; S.f.region = [r.Region]; go("anomalies"); }, foot: "<span>Click a row to open it in the anomaly monitor.</span>" });
}

/* ------------------------------------------------------------------ 6. macro drivers */
async function vDrivers(C, alive) {
  const d = await api("drivers", { family: S.p.dfam || "", indicator: S.p.dind || "" }, false);
  if (!alive()) return;
  const ind = d.indicators, fams = d.families;
  C.innerHTML = `
    <div class="grid g-2-1">
      ${card("Lead / lag heatmap", chartDiv("d-best", "tall") + `<div class="note">Correlation of family demand growth (YoY, 3-month average) with each indicator at its <b>best lead</b> (label = months the indicator leads). COVID months (2020–21) left out. Click a cell to inspect.</div>`, { sub: "family × indicator" })}
      ${card("Strongest driver per family", `<div id="d-top"></div>`)}
    </div>
    <div class="grid g2">
      ${card(`${d.family}: correlation by lead`, chartDiv("d-ll", "tall"), { sub: "indicator × months of lead", tools: `<select class="sel" id="d-fam">${fams.map((f) => `<option ${f === d.family ? "selected" : ""}>${esc(f)}</option>`).join("")}</select>` })}
      ${card(`${pretty(d.indicator)} leads ${d.family} by ${d.lag} months`, chartDiv("d-line", "tall") + `<div class="note">Both series standardised (z-score) to share one axis. Indicator shifted forward by ${d.lag} months; transform: ${esc(d.transform)}.</div>`,
        { tools: `<select class="sel" id="d-ind">${ind.map((x) => `<option value="${esc(x)}" ${x === d.indicator ? "selected" : ""}>${esc(pretty(x))}</option>`).join("")}</select>` })}
    </div>
    <div class="grid g2">
      ${card("Shift to electric vehicles", chartDiv("d-ev"), { sub: "share of demand by powertrain application" })}
      ${card(`Is the lead stable? ${d.family}`, `<div id="d-stab"></div>`, { sub: "best lead in three rolling windows" })}
    </div>`;
  $("#d-fam").addEventListener("change", (e) => { S.p.dind = ""; setParam("dfam", e.target.value); });
  $("#d-ind").addEventListener("change", (e) => setParam("dind", e.target.value));
  const div = { dimension: 2, min: -1, max: 1, calculable: false, orient: "horizontal", left: "center", bottom: 0, itemWidth: 12, itemHeight: 160, textStyle: { color: css("--text-muted"), fontSize: 10 }, inRange: { color: [css("--div-neg"), css("--div-mid"), css("--div-pos")] }, text: ["+1", "−1"] };
  mount("d-best", {
    legend: false,
    tooltip: { trigger: "item", formatter: (p) => ttHead(`${fams[p.value[1]]} × ${pretty(ind[p.value[0]])}`) + ttRow(null, "Correlation", p.value[2].toFixed(2)) + ttRow(null, "Best lead", p.value[3] + " months") },
    grid: { left: 8, right: 8, top: 8, bottom: 44, containLabel: true },
    xAxis: axisCat(ind.map(pretty), { axisLabel: { color: css("--text-muted"), fontSize: 10.5, interval: 0, rotate: 30, width: 110, overflow: "truncate" } }),
    yAxis: axisCat(fams, { inverse: true, axisLabel: { color: css("--text-secondary"), fontSize: 11.5 } }),
    visualMap: div,
    series: [{ type: "heatmap", data: d.best_matrix, label: { show: true, formatter: (p) => `${p.value[2].toFixed(2)}\n${p.value[3]}m`, fontSize: 10, color: css("--text-primary") },
      itemStyle: { borderColor: css("--surface-1"), borderWidth: 2, borderRadius: 3 } }],
  }, (p) => { S.p.dfam = fams[p.value[1]]; setParam("dind", ind[p.value[0]]); });
  table("d-top", [
    { k: "Product_Family", label: "Family" }, { k: "indicator", label: "Indicator", fmt: pretty },
    { k: "lag", label: "Lead (m)", num: true }, { k: "corr", label: "Corr.", num: true, fmt: (v) => v.toFixed(2) },
  ], d.top, { height: 380 });
  const lags = [...Array(13).keys()];
  mount("d-ll", {
    legend: false,
    tooltip: { trigger: "item", formatter: (p) => ttHead(`${pretty(ind[p.value[1]])}, lead ${p.value[0]}m`) + ttRow(null, "Correlation", p.value[2].toFixed(2)) },
    grid: { left: 8, right: 8, top: 8, bottom: 44, containLabel: true },
    xAxis: axisCat(lags.map(String), { name: "lead (months)", nameLocation: "middle", nameGap: 24, nameTextStyle: { color: css("--text-muted"), fontSize: 11 } }),
    yAxis: axisCat(ind.map(pretty), { inverse: true, axisLabel: { color: css("--text-secondary"), fontSize: 11 } }),
    visualMap: { ...div, show: false },
    series: [{ type: "heatmap", data: d.leadlag, itemStyle: { borderColor: css("--surface-1"), borderWidth: 1.5, borderRadius: 2 } }],
  }, (p) => setParam("dind", ind[p.value[1]]));
  const z = (a) => { const v = a.filter(isNum); const m = v.reduce((x, y) => x + y, 0) / v.length; const s = Math.sqrt(v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length) || 1; return a.map((x) => (isNum(x) ? (x - m) / s : null)); };
  const im = Object.fromEntries(d.ind_months.map((m, i) => [m, d.ind[i]]));
  mount("d-line", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => (isNum(v) ? v.toFixed(2) + " σ" : "–") },
    xAxis: axisCat(d.months, { boundaryGap: false }), yAxis: axisVal({ axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => v + "σ" } }),
    series: [
      { name: `${d.family} demand growth`, type: "line", data: z(d.growth), showSymbol: false, lineStyle: { width: 2, color: series(1) }, itemStyle: { color: series(1) } },
      { name: `${pretty(d.indicator)} (lead ${d.lag}m)`, type: "line", data: z(d.months.map((m) => im[m] ?? null)), showSymbol: false, lineStyle: { width: 2, color: series(2), type: [6, 3] }, itemStyle: { color: series(2) } },
    ],
  });
  const ev = d.ev;
  mount("d-ev", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtPct(v, 1) },
    xAxis: axisCat(ev.map((r) => r.Date), { boundaryGap: false }),
    yAxis: axisVal({ max: 1, axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } }),
    series: [["ICE_share", "ICE", 2], ["Universal_share", "Universal", 1], ["EV_share", "EV", 3]].map(([k, n, s]) => ({
      name: n, type: "line", stack: "sh", data: ev.map((r) => r[k] ?? 0), showSymbol: false, lineStyle: { width: 0 }, areaStyle: { color: series(s), opacity: 0.85 }, itemStyle: { color: series(s) },
    })),
  });
  table("d-stab", [
    { k: "indicator", label: "Indicator", fmt: pretty }, { k: "window", label: "Window" },
    { k: "lag", label: "Lead (m)", num: true }, { k: "corr", label: "Corr.", num: true, fmt: (v) => v.toFixed(2) },
  ], d.stability, { height: 300 });
}

/* ------------------------------------------------------------------ 7. forecast & accuracy */
async function vForecast(C, alive) {
  const d = await api("forecast");
  if (!alive()) return;
  const k = d.kpi;
  const metric = S.p.metric || "units";
  const src = Object.fromEntries(d.sources.map((r) => [r.source, r.units]));
  const tot = sum(d.sources.map((r) => r.units));
  C.innerHTML = `
    <div class="kpis">
      ${kpi("SKU-level accuracy", fmtAcc(k.wape), `1 − WAPE · seasonal naive ${fmtAcc(k.naive_wape)}`, "SKU × region × month, 4 rolling origins, 1–19 months ahead")}
      ${kpi("Selection-total accuracy", fmtAcc(k.agg_wape, 1), "errors net out when series are added")}
      ${kpi("Bias", fmtPct(k.bias, 1, true), "forecast − actual, ÷ actual")}
      ${kpi("P10–P90 coverage", fmtPct(k.coverage, 1), "share of actuals inside the range (target 80%)")}
      ${kpi("Forecast source", fmtPct((src.gbm || 0) / Math.max(1, tot), 1), `LightGBM · ${fmtPct((src.new_launch_rule || 0) / Math.max(1, tot), 1)} launch rule · ${fmtPct((src.baseline_short_history || 0) / Math.max(1, tot), 1)} short-history baseline`)}
    </div>
    ${card("Forecast vs history", chartDiv("f-tl", "tall") + `<div class="note">Band: ${esc(d.timeline.band)}.</div>`, { sub: "P50 with P10–P90 range", tools: seg("metric", [["units", "Units"], ["revenue", "Revenue"]], metric) })}
    <div class="grid g2">
      ${card("Backtest: what the model said at each origin", chartDiv("f-fan", "tall") + `<div class="note">Each dashed line is a 19-month forecast made at the origin shown, using only data before it. Solid = actual demand.</div>`, { sub: "selection total" })}
      ${card("Model leaderboard", `<div id="f-board"></div><div class="note">WAPE = Σ|actual − forecast| ÷ Σ actual. <b>SKU</b>: per SKU-region-month. <b>Total</b>: on the selection total per month.</div>`, { sub: "backtest, selection" })}
    </div>
    <div class="grid g3">
      ${card("WAPE by horizon", chartDiv("f-h"), { sub: "months ahead" })}
      ${card("WAPE by family", chartDiv("f-fam"), { sub: "final model vs seasonal naive" })}
      ${card("WAPE by region", chartDiv("f-reg"), { sub: "final model vs seasonal naive" })}
    </div>
    <div class="grid g3">
      ${card("Bias by family and region", chartDiv("f-bias"), { sub: "negative = under-forecast" })}
      ${card("P10–P90 coverage by origin", chartDiv("f-cov"), { sub: "target 80%" })}
      ${card("What drives the model", chartDiv("f-imp"), { sub: "LightGBM gain, top 15 features" })}
    </div>`;
  bindSeg(C, { metric: (v) => setParam("metric", v) });
  mount("f-tl", timelineOption(d.timeline, metric, { start: "2023-01" }));
  const tl = d.timeline, all = [...tl.hist_months];
  const startI = all.indexOf("2023-01");
  const xs = all.slice(startI);
  const fanColors = [series(7), series(3), series(5), series(2)];
  mount("f-fan", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
    xAxis: axisCat(xs, { boundaryGap: false }), yAxis: axisVal({ scale: true }),
    series: [
      { name: "Actual demand", type: "line", data: tl.demand.slice(startI), showSymbol: false, lineStyle: { width: 2.4, color: series(1) }, itemStyle: { color: series(1) }, z: 5 },
      ...d.fan.map((f, i) => {
        const m = Object.fromEntries(f.dates.map((x, j) => [x, f.pred[j]]));
        return { name: "Origin " + f.cutoff, type: "line", data: xs.map((x) => m[x] ?? null), showSymbol: false, lineStyle: { width: 1.6, type: [5, 3], color: fanColors[i % 4] }, itemStyle: { color: fanColors[i % 4] } };
      }),
    ],
  });
  table("f-board", [
    { k: "label", label: "Model", html: (r) => (r.final ? `<b>${esc(r.label)}</b>` : esc(r.label)) },
    { k: "wape", label: "WAPE (SKU)", num: true, fmt: (v) => fmtPct(v, 1) },
    { k: "agg_wape", label: "WAPE (total)", num: true, fmt: (v) => fmtPct(v, 1) },
    { k: "bias", label: "Bias", num: true, fmt: (v) => fmtPct(v, 1, true) },
  ], d.board, { sort: "wape", dir: 1, height: 330 });
  const h = d.by_horizon.sort((a, b) => a.name - b.name);
  mount("f-h", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtPct(v, 1) },
    xAxis: axisCat(h.map((r) => String(r.name))), yAxis: axisVal({ scale: true, axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } }),
    series: [
      { name: "LightGBM (final)", type: "line", data: h.map((r) => r.wape), symbolSize: 5, lineStyle: { width: 2, color: series(1) }, itemStyle: { color: series(1) } },
      { name: "Seasonal naive", type: "line", data: h.map((r) => r.naive_wape), symbolSize: 5, lineStyle: { width: 1.5, color: css("--muted-line"), type: [5, 3] }, itemStyle: { color: css("--muted-line") } },
    ],
  });
  const wb = (id, rows, dim) => {
    rows = rows.slice().sort((a, b) => a.wape - b.wape);
    mount(id, barH(rows.map((r) => r.name), [
      { name: "LightGBM (final)", data: rows.map((r) => r.wape), itemStyle: { color: series(1), borderRadius: [0, 4, 4, 0] } },
      { name: "Seasonal naive", data: rows.map((r) => r.naive_wape), itemStyle: { color: css("--muted-line"), borderRadius: [0, 4, 4, 0] } },
    ], { labelWidth: 130, tooltip: { valueFormatter: (v) => fmtPct(v, 1) }, xAxis: { axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } } }),
    (p) => setFilter(dim, [rows[p.dataIndex].name]));
  };
  wb("f-fam", d.by_family, "family");
  wb("f-reg", d.by_region, "region");
  const bi = [...d.by_family.map((r) => ({ ...r, g: "Family" })), ...d.by_region.map((r) => ({ ...r, g: "Region" }))];
  mount("f-bias", barH(bi.map((r) => r.name), [{ name: "Bias", data: bi.map((r) => ({ value: r.bias, itemStyle: { color: r.bias < 0 ? css("--div-neg") : css("--div-pos"), borderRadius: 3 } })) }],
    { labelWidth: 130, tooltip: { valueFormatter: (v) => fmtPct(v, 1, true) }, xAxis: { axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } } }));
  mount("f-cov", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtPct(v, 1) },
    xAxis: axisCat(d.coverage.map((r) => r.cutoff)), yAxis: axisVal({ max: 1, axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } }),
    series: [
      { name: "Inside P10–P90", type: "bar", stack: "c", data: d.coverage.map((r) => r.coverage), itemStyle: { color: series(3) }, barMaxWidth: 34,
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 0.8 }], lineStyle: { color: css("--text-muted"), type: "dashed" }, label: { formatter: "80%", color: css("--text-muted"), fontSize: 10 } } },
      { name: "Below P10", type: "bar", stack: "c", data: d.coverage.map((r) => r.below), itemStyle: { color: series(1) } },
      { name: "Above P90", type: "bar", stack: "c", data: d.coverage.map((r) => r.above), itemStyle: { color: series(2) } },
    ],
  });
  const imp = d.importance, gtot = sum(imp.map((x) => x.gain));
  mount("f-imp", barH(imp.map((r) => pretty(r.feature)), [{ name: "Gain share", data: imp.map((r) => r.gain / gtot), itemStyle: { color: series(7), borderRadius: [0, 4, 4, 0] } }],
    { labelWidth: 150, tooltip: { valueFormatter: (v) => fmtPct(v, 1) }, xAxis: { axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: (v) => fmtPct(v, 0) } } }));
}

/* ------------------------------------------------------------------ 8. exceptions */
const FLAG_INFO = {
  "Forecast jump": "next 12 months ≥35% above last 12",
  "Forecast drop": "next 12 months ≥35% below last 12",
  "Wide range": "P10–P90 wider than the P50",
  "Low past accuracy": "backtest WAPE ≥ 60%",
  "Recent stockouts": "≥2 stockout months in the last 6",
  "Unexplained dips": "≥2 unexplained dips in the last 6 months",
  "Level shift": "level shift, customer switch or lost account in the last 12 months",
  "EOL run-down": "end of life inside the horizon",
  "New launch": "no history; forecast from the replaced SKU",
};
async function vExceptions(C, alive) {
  const flag = S.p.flag || "";
  const d = await api("exceptions", { flag });
  if (!alive()) return;
  const prioColor = { High: css("--critical"), Medium: css("--warning"), Low: css("--good") };
  const prioIcon = { High: "▲", Medium: "■", Low: "●" };
  const nHigh = d.table.filter((r) => r.priority === "High").length;
  C.innerHTML = `
    <div class="kpis">
      ${kpi("Series needing review", fmtInt(d.total), flag ? `flag: ${esc(flag)}` : "out of " + fmtInt(META.rows.length) + " combinations")}
      ${kpi("High priority", fmtInt(nHigh), "top 15% by flags × revenue")}
      ${kpi("Revenue at stake (12 mo)", fmtUSD(d.rev_at_stake), "P50 revenue of flagged series")}
    </div>
    <div class="card"><div class="card-h"><h3 class="card-title">Why a series is flagged</h3><span class="card-sub">click to filter</span></div>
      <div class="chips"><button class="chip ${!flag ? "on" : ""}" data-f="">All</button>
      ${d.counts.map((c) => `<button class="chip ${flag === c.flag ? "on" : ""}" data-f="${esc(c.flag)}" title="${esc(FLAG_INFO[c.flag] || "")}">${esc(c.flag)} <span class="n">${c.n}</span></button>`).join("")}</div>
    </div>
    ${card("Exception list", `<div id="x-tbl"></div>`, { sub: "ranked by number of flags × 12-month revenue · click a row to open its forecast", tools: dlLink("exceptions") })}`;
  $$(".chip[data-f]", C).forEach((b) => b.addEventListener("click", () => setParam("flag", b.dataset.f)));
  table("x-tbl", [
    { k: "priority_score", label: "Priority", html: (r) => `<span class="prio" style="color:${prioColor[r.priority]}"><span>${prioIcon[r.priority]}</span><span style="color:var(--text-primary)">${r.priority}</span></span>` },
    { k: "SKU_ID", label: "SKU" }, { k: "Region", label: "Region" }, { k: "Customer_Name", label: "Customer" }, { k: "Sales_Channel", label: "Channel" },
    { k: "last12", label: "Last 12 mo", num: true, fmt: fmtInt }, { k: "next12", label: "Next 12 mo", num: true, fmt: fmtInt },
    { k: "change", label: "Change", num: true, fmt: (v) => fmtPct(v, 0, true) }, { k: "band_width", label: "Range ÷ P50", num: true, fmt: (v) => fmtPct(v, 0) },
    { k: "bt_wape", label: "Past WAPE", num: true, fmt: (v) => fmtPct(v, 0) }, { k: "rev12", label: "Revenue 12 mo", num: true, fmt: fmtUSD },
    { wrap: true, k: "flags", label: "Flags", html: (r) => r.flags.map((f) => `<span class="tag" title="${esc(FLAG_INFO[f] || "")}">${esc(f)}</span>`).join("") },
    { wrap: true, k: "action", label: "Suggested action" },
  ], d.table, { height: 560, onRow: (r) => { DIMS.forEach((x) => (S.f[x] = [])); S.f.sku = [r.SKU_ID]; S.f.region = [r.Region]; go("forecast"); },
    foot: d.total > d.table.length ? `<span>Top ${d.table.length} of ${fmtInt(d.total)} shown; download for all.</span>` : "" });
}

/* ------------------------------------------------------------------ 9. procurement */
async function vProcurement(C, alive) {
  const w = +(S.p.window || 6);
  const d = await api("procurement", { window: w });
  if (!alive()) return;
  const k = d.kpi;
  C.innerHTML = `
    <div class="card"><div class="card-h"><h3 class="card-title">Planning window</h3><span class="card-sub">order cover from ${META.fc_months[0]} for the next</span>
      <div class="card-tools">${seg("window", [[3, "3 months"], [6, "6 months"], [12, "12 months"], [19, "19 months"]], w)}</div></div>
      <div class="note" style="margin:0">P90 = a quantity demand stays below 9 months out of 10, per SKU and region (each region stocks separately). The <b>buffer</b> is what P90 adds over the P50 forecast.</div></div>
    <div class="kpis">
      ${kpi("Plan at P90", fmtN(k.p90_units) + " units", `vs ${fmtN(k.p50_units)} at P50`)}
      ${kpi("Safety buffer", fmtN(k.p90_units - k.p50_units) + " units", fmtPct((k.p90_units - k.p50_units) / Math.max(1, k.p50_units), 1, true) + " over P50")}
      ${kpi("Plan value (P90)", fmtUSD(k.p90_value), "at forecast selling price")}
      ${kpi("Buffer value", fmtUSD(k.buffer_value), "cost of cover above P50")}
      ${kpi("SKUs in plan", fmtInt(k.skus))}
    </div>
    <div class="grid g-2-1">
      ${card("Monthly P50 vs P90", chartDiv("p-m"), { sub: "sum of series quantities · window shaded" })}
      ${card("Buffer value by family", chartDiv("p-fam"))}
    </div>
    ${card("Purchase plan by SKU", `<div id="p-tbl"></div>`, { sub: `next ${w} months · sorted by P90 value`, tools: dlLink("procurement", "Download plan", { window: w }) })}`;
  bindSeg(C, { window: (v) => setParam("window", v) });
  mount("p-m", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
    xAxis: axisCat(d.months), yAxis: axisVal(),
    series: [
      { name: "P50", type: "bar", data: d.p50, itemStyle: { color: series(1), borderRadius: [4, 4, 0, 0] }, barGap: "10%", barMaxWidth: 16,
        markArea: { silent: true, itemStyle: { color: css("--accent-soft") }, data: [[{ xAxis: d.months[0] }, { xAxis: d.months[w - 1] }]] } },
      { name: "P90", type: "bar", data: d.p90, itemStyle: { color: series(2), borderRadius: [4, 4, 0, 0] }, barMaxWidth: 16 },
    ],
  });
  const f = d.by_family.sort((a, b) => b.buffer_value - a.buffer_value);
  mount("p-fam", barH(f.map((r) => r.Product_Family), [{ name: "Buffer value", data: f.map((r) => r.buffer_value), itemStyle: { color: series(2), borderRadius: [0, 4, 4, 0] } }],
    { labelWidth: 140, tooltip: { valueFormatter: fmtUSD }, xAxis: { axisLabel: { color: css("--text-muted"), fontSize: 11, formatter: fmtUSD } } }));
  table("p-tbl", [
    { k: "SKU_ID", label: "SKU" }, { wrap: true, k: "SKU_Description", label: "Description" }, { k: "Product_Family", label: "Family" },
    { k: "regions", label: "Regions", num: true }, { k: "p50", label: "P50 units", num: true, fmt: fmtInt }, { k: "p90", label: "P90 units", num: true, fmt: fmtInt },
    { k: "buffer", label: "Buffer", num: true, fmt: fmtInt }, { k: "buffer_pct", label: "Buffer %", num: true, fmt: (v) => fmtPct(v, 0) },
    { k: "unit_cost_proxy", label: "Unit price", num: true, fmt: (v) => (isNum(v) ? "$" + v.toFixed(0) : "–") },
    { k: "p90_value", label: "P90 value", num: true, fmt: fmtUSD }, { k: "buffer_value", label: "Buffer value", num: true, fmt: fmtUSD },
    { k: "Lifecycle_Status", label: "Status" },
  ], d.table, { sort: "p90_value", height: 480, foot: d.n > d.table.length ? `<span>Top ${d.table.length} of ${d.n} SKUs; download for all.</span>` : "" });
}

/* ------------------------------------------------------------------ 10. what-if */
const PRESETS = [
  ["ISM_Manufacturing_PMI", -3, "PMI −3 pts"], ["ISM_Manufacturing_PMI", 3, "PMI +3 pts"],
  ["Raw_Material_Cost_Index", 10, "Raw materials +10%"], ["New_Vehicle_Orders_Index", -10, "Vehicle orders −10%"],
  ["Auto_Production_YoY_Pct", -5, "Auto production −5 pts"], ["Fuel_Price_Index", 15, "Fuel +15%"],
];
async function vWhatif(C, alive) {
  const ind = S.p.wind || "ISM_Manufacturing_PMI";
  const shock = isNum(parseFloat(S.p.wshock)) ? parseFloat(S.p.wshock) : -3;
  const d = await api("whatif", { indicator: ind, shock });
  if (!alive()) return;
  const lvl = d.level_type;
  const rng = lvl ? 10 : 30;
  const dU = d.scen_total - d.base_total, dR = d.scen_rev - d.base_rev;
  const rel = (r2) => (r2 >= 0.4 ? ["strong", css("--good")] : r2 >= 0.2 ? ["moderate", css("--warning")] : ["weak", css("--critical")]);
  C.innerHTML = `
    <div class="card">
      <div class="card-h"><h3 class="card-title">Scenario</h3><span class="card-sub">sustained change from ${META.fc_months[0]} onward</span></div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
        <label>Indicator <select class="sel" id="w-ind">${d.indicators.map((x) => `<option value="${esc(x)}" ${x === ind ? "selected" : ""}>${esc(pretty(x))}</option>`).join("")}</select></label>
        <label style="display:flex;gap:8px;align-items:center">Shock <input type="range" id="w-rng" min="${-rng}" max="${rng}" step="${lvl ? 0.5 : 1}" value="${shock}" style="width:220px">
          <input class="num-in" id="w-num" type="number" step="${lvl ? 0.5 : 1}" value="${shock}" style="width:80px"> ${esc(d.unit)}</label>
        <span class="card-sub">latest value: ${isNum(d.current) ? d.current.toFixed(1) : "–"}</span>
      </div>
      <div class="chips" style="margin-top:10px">${PRESETS.map(([i, s, l]) => `<button class="chip ${i === ind && s === shock ? "on" : ""}" data-i="${i}" data-s="${s}">${esc(l)}</button>`).join("")}</div>
    </div>
    <div class="kpis">
      ${kpi("Baseline (19 mo, P50)", fmtN(d.base_total) + " units", fmtUSD(d.base_rev))}
      ${kpi("Scenario", fmtN(d.scen_total) + " units", fmtUSD(d.scen_rev))}
      ${kpi("Change in units", (dU >= 0 ? "+" : "") + fmtN(dU), delta(dU / Math.max(1, d.base_total), 2))}
      ${kpi("Change in revenue", (dR >= 0 ? "+" : "−") + fmtUSD(Math.abs(dR)), delta(dR / Math.max(1, d.base_rev), 2))}
    </div>
    ${card("Baseline vs scenario", chartDiv("w-ch", "tall"), { sub: "last 24 months of history, then the 19-month forecast" })}
    ${card("Response by family", `<div id="w-tbl"></div><div class="note">Each family responds with the slope of its demand growth (YoY) on the indicator at the indicator's best lead, fitted on 2019–2026 excluding COVID. ${lvl ? "For level indicators a sustained shift keeps growth lower or higher for as long as it lasts, so the effect builds after 12 months." : "The index is shifted by the % given, which moves the demand level once the lead has passed."} Sensitivities are historical associations, not causal effects; treat <b>weak</b> fits with care.</div>`)}`;
  const upd = debounce((i, s) => { S.p.wind = i; setParam("wshock", s); }, 300);
  $("#w-ind").addEventListener("change", (e) => { S.p.wind = e.target.value; setParam("wshock", LEVEL_DEFAULT(e.target.value)); });
  $("#w-rng").addEventListener("input", (e) => { $("#w-num").value = e.target.value; upd(ind, e.target.value); });
  $("#w-num").addEventListener("change", (e) => upd(ind, e.target.value));
  $$(".chip[data-i]", C).forEach((b) => b.addEventListener("click", () => { S.p.wind = b.dataset.i; setParam("wshock", b.dataset.s); }));
  const x = [...d.hist_months, ...d.months], nH = d.hist_months.length, nul = Array(nH).fill(null);
  const base = [...nul, ...d.base], scen = [...nul, ...d.scen];
  base[nH - 1] = d.hist[nH - 1]; scen[nH - 1] = d.hist[nH - 1];
  mount("w-ch", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
    xAxis: axisCat(x, { boundaryGap: false }), yAxis: axisVal({ scale: true }),
    series: [
      { name: "History", type: "line", data: [...d.hist, ...Array(d.months.length).fill(null)], showSymbol: false, lineStyle: { width: 2, color: series(1) }, itemStyle: { color: series(1) } },
      { name: "Baseline P50", type: "line", data: base, showSymbol: false, lineStyle: { width: 2, type: [6, 4], color: series(2) }, itemStyle: { color: series(2) },
        markArea: { silent: true, itemStyle: { color: css("--forecast-zone") }, data: [[{ xAxis: d.months[0] }, { xAxis: d.months.at(-1) }]] } },
      { name: "Scenario", type: "line", data: scen, showSymbol: false, lineStyle: { width: 2.4, color: series(7) }, itemStyle: { color: series(7) } },
    ],
  });
  table("w-tbl", [
    { k: "family", label: "Family" }, { k: "lag", label: "Lead (m)", num: true },
    { k: "beta", label: "Sensitivity", num: true, fmt: (v) => v.toFixed(4) }, { k: "corr", label: "Corr.", num: true, fmt: (v) => v.toFixed(2) },
    { k: "r2", label: "Fit", html: (r) => { const [l, c] = rel(r.r2); return `<span class="prio"><i style="background:${c}"></i>${l} (R² ${r.r2.toFixed(2)})</span>`; } },
    { k: "base", label: "Baseline", num: true, fmt: fmtInt }, { k: "scen", label: "Scenario", num: true, fmt: fmtInt },
    { k: "pct", label: "Change", num: true, html: (r) => delta(r.scen / Math.max(1, r.base) - 1, 2) },
    { k: "drev", label: "Revenue change", num: true, html: (r) => fmtUSD(r.scen_rev - r.base_rev) },
  ], d.families.map((r) => ({ ...r, pct: r.scen / Math.max(1, r.base) - 1, drev: r.scen_rev - r.base_rev })), { sort: "pct", height: 340 });
}
const LEVEL_DEFAULT = (i) => (["ISM_Manufacturing_PMI", "Industrial_Util_Rate_Pct", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days"].includes(i) ? -3 : -10);

/* ------------------------------------------------------------------ 11. lifecycle */
async function vLifecycle(C, alive) {
  const d = await api("lifecycle");
  if (!alive()) return;
  const k = d.kpi;
  const rows = d.table;
  const pairs = rows.filter((r) => r.Replaces_SKU);
  C.innerHTML = `
    <div class="kpis">
      ${kpi("SKUs in selection", fmtInt(k.skus))}
      ${kpi("End of life in horizon", fmtInt(k.eol_in_horizon), "forecast runs down, then 0 after EOL")}
      ${kpi("Launches in horizon", fmtInt(k.launch_in_horizon), "forecast from the SKU they replace")}
      ${kpi("Discontinued", fmtInt(k.discontinued), "no forecast")}
    </div>
    <div class="grid g-1-2">
      ${card("SKUs by status", chartDiv("l-st"))}
      ${card("Forecast for run-downs and launches", chartDiv("l-stack"), { sub: "P50 units, active SKUs excluded" })}
    </div>
    ${card("Launches, run-downs and replacements", `<div class="mini-grid" id="l-mini"></div>`, { sub: "last 36 months of demand (solid) and forecast (dashed) per SKU" })}
    ${card("Lifecycle register", `<div id="l-tbl"></div>`, { sub: `${pairs.length} replacement pairs` })}`;
  mount("l-st", barH(d.status.map((r) => r.status), [{ name: "SKUs", data: d.status.map((r) => r.n), itemStyle: { color: series(7), borderRadius: [0, 4, 4, 0] } }], { labelWidth: 110, xAxis: { type: "log", axisLabel: { color: css("--text-muted"), formatter: (v) => fmtN(v) } } }));
  const groups = Object.keys(d.stack).filter((g) => g !== "Active");
  mount("l-stack", {
    legend: {}, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
    xAxis: axisCat(d.months), yAxis: axisVal(),
    series: groups.map((g) => ({ name: g, type: "bar", stack: "g", data: d.stack[g], itemStyle: { color: g === "New launch" ? series(3) : series(2) }, barMaxWidth: 22 })),
  });
  const mini = $("#l-mini");
  const shown = rows.filter((r) => d.lines[r.SKU_ID] && (sum(d.lines[r.SKU_ID].hist) > 0 || sum(d.lines[r.SKU_ID].fc) > 0)).slice(0, 16);
  mini.innerHTML = shown.map((r) => `<div class="mini"><div class="t">${esc(r.SKU_ID)} · ${esc(r.Lifecycle_Status)}</div><div class="d">${esc(r.SKU_Description)}${r.Replaces_SKU ? ` · replaces ${esc(r.Replaces_SKU)}` : ""}${r.EOL_Date ? ` · EOL ${esc(r.EOL_Date)}` : ""}${r.Launch_Date >= "2024-01" ? ` · launch ${esc(r.Launch_Date)}` : ""}</div><div class="chart" id="lm-${esc(r.SKU_ID)}"></div></div>`).join("") || `<div class="empty">No lifecycle events in this selection.</div>`;
  const x = [...d.hist_months, ...d.months], nH = d.hist_months.length;
  shown.forEach((r) => {
    const L = d.lines[r.SKU_ID];
    mount("lm-" + r.SKU_ID, {
      legend: false, grid: { left: 4, right: 8, top: 8, bottom: 4 }, tooltip: { trigger: "axis", valueFormatter: (v) => fmtN(v) },
      xAxis: axisCat(x, { boundaryGap: false, axisLabel: { show: false } }), yAxis: axisVal({ splitNumber: 2 }),
      series: [
        { name: "Demand", type: "line", data: [...L.hist, ...Array(d.months.length).fill(null)], showSymbol: false, lineStyle: { width: 1.6, color: series(1) }, itemStyle: { color: series(1) } },
        { name: "Forecast", type: "line", data: [...Array(nH).fill(null), ...L.fc], showSymbol: false, lineStyle: { width: 1.6, type: [4, 3], color: series(2) }, itemStyle: { color: series(2) },
          markArea: { silent: true, itemStyle: { color: css("--forecast-zone") }, data: [[{ xAxis: d.months[0] }, { xAxis: d.months.at(-1) }]] } },
      ],
    });
  });
  table("l-tbl", [
    { k: "SKU_ID", label: "SKU" }, { wrap: true, k: "SKU_Description", label: "Description" }, { k: "Product_Family", label: "Family" },
    { k: "Powertrain_Application", label: "Powertrain" }, { k: "Launch_Date", label: "Launch" }, { k: "EOL_Date", label: "EOL" },
    { k: "Replaces_SKU", label: "Replaces" }, { k: "Lifecycle_Status", label: "Status" },
    { k: "last12", label: "Last 12 mo", num: true, fmt: fmtInt }, { k: "fc19", label: "Next 19 mo", num: true, fmt: fmtInt },
    { k: "rule", label: "Forecast rule", fmt: pretty },
  ], rows, { height: 420, onRow: (r) => { DIMS.forEach((x) => (S.f[x] = [])); S.f.sku = [r.SKU_ID]; go("forecast"); }, foot: "<span>Click a row to open the SKU's forecast.</span>" });
}

/* ------------------------------------------------------------------ 12. downloads */
async function vDownloads(C) {
  const any = DIMS.some((d) => S.f[d].length);
  const items = [
    ["forecast", "Forecast", "SKU × region × customer × month, P10 / P50 / P90, revenue, source and lifecycle rule."],
    ["history", "Cleaned history", "Monthly shipped units, unconstrained demand, revenue and stockout flags."],
    ["anomalies", "Anomaly register", "Every flag with type, category, value, expected value, reason and action."],
    ["procurement", "Procurement plan (6 months)", "P50 / P90 quantities, buffer and value per SKU."],
    ["exceptions", "Exception list", "Flagged series with reasons and suggested actions."],
    ["backtest", "Backtest predictions", "All models at 4 rolling origins with actuals (large)."],
    ["submission", "Submission file", "outputs/submission.csv exactly as submitted (filters not applied)."],
  ];
  C.innerHTML = `
    <div class="insight">${any ? `Downloads use the current filters: <b>${esc(DIMS.filter((d) => S.f[d].length).map((d) => `${DIM_LABEL[d]} = ${S.f[d].join(", ")}`).join(" · "))}</b>.` : "No filters set: downloads contain the whole dataset."}</div>
    <div class="grid g3">${items.map(([k, t, desc]) => card(t, `<div class="dl-card"><div class="card-sub">${esc(desc)}</div><div>${k === "submission" ? `<a class="btn primary" href="/api/download/submission.csv" download>⤓ Download CSV</a>` : dlLink(k).replace('class="btn"', 'class="btn primary"')}</div></div>`)).join("")}</div>
    ${card("API", `<div class="card-sub">Every view is backed by a JSON endpoint that takes the same filters, e.g. <code>/api/forecast?family=Hydraulic%20Systems&amp;region=Europe</code>. Interactive docs: <a href="/api/docs" target="_blank" rel="noopener">/api/docs</a>.</div>`)}`;
}

const RENDER = {
  overview: vOverview, health: vHealth, explorer: vExplorer, anomalies: vAnomalies, stockouts: vStockouts, drivers: vDrivers,
  forecast: vForecast, exceptions: vExceptions, procurement: vProcurement, whatif: vWhatif, lifecycle: vLifecycle, downloads: vDownloads,
};

/* ------------------------------------------------------------------ boot */
(async function boot() {
  try {
    META = await api("meta", {}, false);
  } catch (e) {
    $("#content").innerHTML = `<div class="card"><div class="empty">Could not reach the API: ${esc(e.message)}</div></div>`;
    return;
  }
  $("#stamp").innerHTML = `History ${META.hist_months[0]} → ${META.train_end} · <b>Forecast ${META.test_start} → ${META.test_end}</b>`;
  readHash();
  render();
})();
