# Celeris Automobili — Demand Forecasting & Planning Control Tower

End-to-end pipeline for the Celeris Automobili Industrial case: clean a messy ERP extract,
separate data errors from real demand events, estimate unconstrained demand, forecast
`Units_Sold` for every row of `test.csv` (2026-11 → 2028-05, 19 months) and serve a
planning dashboard from the pipeline outputs.

**Demo video (2:40, every dashboard view):** [docs/demo.mp4](docs/demo.mp4)

## Quick start

```bash
# 1. environment (Python 3.12, uv)
make setup                       # or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock -e ".[dev]"

# 2. data: put the organiser's zip at data/raw/student_data.zip, then
make data                        # extracts the CSVs into data/raw/

# 3. whole pipeline (clean -> anomalies -> features -> forecast), ~25 min on 16 cores
make all                         # = python -m celeris all

# 4. tests
make test

# 5. dashboard: http://127.0.0.1:8501
make app                         # = python -m celeris.dashboard [--host 0.0.0.0 --port 8501]
```

Run single stages with `python -m celeris clean|anomalies|features|forecast`.
Set `CELERIS_SKIP_BACKTEST=1` to skip the rolling-origin backtest and only refit the
final model. LightGBM uses `CELERIS_THREADS` threads (default 4); raise it on a machine
with good cooling. `requirements.lock` pins the exact library versions the results were
produced with.

## Outputs

| File | What it is |
|---|---|
| `outputs/submission.csv` | Final forecast, one row per `test.csv` row (`Date, SKU_ID, Region, Units_Sold_Forecast`) |
| `outputs/anomaly_register.csv` | Every flagged point / series / family event with type, category, reason, action |
| `reports/dq_report.json`, `reports/dq_issues.csv` | Machine-readable data-quality report: issue, file, rows affected, action, before/after counts |
| `reports/anomaly_summary.json` | Register counts, demand vs shipments uplift, price-rise buying |
| `reports/drivers_summary.json` | Leading indicator and lag per family, lag stability, EV shift |
| `reports/backtest_metrics.json` | Rolling-origin WAPE / bias by model, family, region, horizon |
| `data/processed/forecast_aggregates.parquet` | Family / region / total forecast with calibrated P10 / P90 bands |
| `data/processed/*.parquet` | Clean tables the dashboard reads (train_clean, panel, demand, forecast, backtest, drivers…) |

## Dashboard: planning control tower

`make app` starts a FastAPI backend with a single-page UI (ECharts, vendored, so it works
offline). It is one app in two parts, **what happened** (cleaned history) and **what will
happen** (forecast). All views share the same filters (family, SKU, region, channel,
customer), and the filter state is kept in the URL, so any view can be shared as a link.
Every number is computed from `data/processed`, `reports/` and `outputs/` at start-up;
nothing is typed in by hand.

| View | Part | Shows |
|---|---|---|
| Executive overview | both | Units / revenue from history into the forecast on one timeline (forecast shaded, P10–P90 band), mix by family, region and channel, headline 19-month forecast, auto-written summary |
| Data health | training | Issues by type and file, rows affected, raw vs clean vs unconstrained demand, rows per file, full issue log |
| Demand explorer | training | Click-to-drill family → SKU → region → customer; trend, year × month seasonality heatmap, seasonal index, zero months, intermittency class, lifecycle |
| Anomaly monitor | training | Demand line with one lane per flag category (stockout, data error, real event, suspected, data gap), structural breaks, register with type / reason / action, events in scope |
| Forecast & accuracy | prediction | Forecast vs history with P10–P90, backtest fan by origin, model leaderboard, WAPE by horizon / family / region, bias, interval coverage, feature importance, all recomputed for the selection |
| Stockouts & lost sales | optional | Lost and back-ordered units, fill rate, listed vs inferred stockouts, reasons, worst SKU × region |
| Macro drivers | optional | Family × indicator lead/lag heatmap, correlation by lead, standardised overlay, lag stability, EV shift |
| Exception list | optional | Series to review first (forecast jump/drop, wide range, low past accuracy, recent stockouts, level shifts, EOL, launches) with a suggested action, ranked by revenue at stake |
| Procurement plan (P90) | optional | P50 vs P90 quantities per SKU for a 3 / 6 / 12 / 19-month window, buffer units and value |
| What-if scenarios | optional | Sustained shock to any macro indicator (e.g. PMI −3); response per family from historical elasticities at the best lead, with fit quality |
| Lifecycle | optional | EOL run-downs, launches and the SKUs they replace, per-SKU history + forecast |
| Data download | optional | CSV of forecast, history, anomalies, procurement plan, exceptions, backtest, submission, all filtered |

Forecast bands for a selection: whole company, one family or one region use the
calibrated aggregate bands from the pipeline; a single series uses its own quantiles;
any other subset combines the series bands in quadrature, floored at the family-level
backtest spread. The JSON API behind every view is documented at `/api/docs`.

## Results (rolling-origin backtest)

4 forecast origins (2023-10, 2024-04, 2024-10, 2025-03), each scored on the full
19-month horizon against unconstrained demand, SKU × Region × month level.

| Model | WAPE | Bias |
|---|---|---|
| Seasonal naive | 0.356 | −3.4 % |
| 12-month moving average | 0.494 | −3.1 % |
| Seasonal index × level | 0.316 | −2.9 % |
| TSB (intermittent) | 0.493 | −6.0 % |
| LightGBM Tweedie, history + calendar + events only | 0.2745 | −4.0 % |
| + macro change-from-origin features | 0.2710 | −4.7 % |
| + 3-seed bagging | 0.2700 | −5.2 % |
| + one-off uplift | 0.2701 | −4.2 % |
| **+ EV-shift features, 36 / 48-month level (final)** | **0.2671** | **−4.2 %** |

* Summed bottom-up: family-level WAPE 6.7 %, region-level 6.0 % (seasonal naive 9.8 % /
  8.3 %; 8.2 % / 7.6 % before the macro change features).
* SKU-level P10–P90 coverage 80.8 % over all 4 origins (80.4 / 80.2 / 81.5 / 80.9 %;
  target 80 %).
* By horizon: WAPE 0.251 at month 1 → 0.265 at month 19 (peak 0.279). By region: Europe
  and North America 0.239 → Middle East & Africa 0.341 (shortest history, entered 2022).
  By family: Powertrain 0.260 → Electrical Assemblies 0.277.
* EV shift: EV-only SKUs were the worst segment (WAPE 0.324, bias −20 % at the 2023-10
  origin and +24 % at 2025-03): the model extrapolated EV growth momentum, while EV
  demand per series follows the EV adoption index, which flattened at ~90 in 2025.
  Giving EV and ICE series their own change in the (known) index between origin and
  target month cut EV WAPE to 0.285 and SKU-level WAPE by 0.003, better or equal at
  every origin (checked with two seeds before the full run). The 36 / 48-month means and
  12-month median are neutral at SKU level and lower family-level WAPE.
* Bias by origin: −5.3 %, −7.2 %, −3.8 %, −0.5 %, shrinking towards zero for recent
  origins. Roughly 1 point is the capped one-off orders, which the uplift puts back;
  no further blanket correction is applied (a leave-one-origin-out correction made the
  two most recent origins worse).
* Headroom: an oracle that knows each series' true centred 13-month level and seasonal
  profile (future included) scores WAPE 0.215 on the same rows, so roughly 0.2 is
  month-to-month noise that no model can predict. Ordering cycles are near-deterministic
  (98–99 % of gaps between orders equal the series' cycle) and the model already places
  order months correctly; zero-demand months are 1.4 % of the error.
* Tried and rejected (no gain beyond noise, within ±0.001 WAPE or worse at family level):
  aggregate family / region / customer growth features, seasonally clean momentum,
  pooled family × region seasonal index with deseasonalised level, recency weighting,
  wider macro lag sets, 127 / 31 leaves, lower learning rate, Tweedie power 1.1 / 1.5,
  blends with the seasonal-index baseline or the L1 model, EV index *level* (instead of
  change) per powertrain, a ridge regression of family totals on macro changes used
  top-down (family WAPE 13.4 % vs 7.1 % bottom-up), customer effects for the 10 series
  that switch customer at the forecast start (no customer-level volume effect exists in
  the history; post-switch level ratios of 0.5–2.2× are unpredictable).

## Repository layout

```
src/celeris/
  clean/       labels, train cleaning, reference files, SKU x Region panel, DQ log
  anomalies/   unconstrained demand, outlier detection + event attribution, structural checks
  features/    direct multi-horizon feature matrix, macro lead/lag drivers, intermittency
  models/      baselines, LightGBM, rolling-origin backtest, lifecycle rules, final forecast
tests/         unit tests (labels), invariants on every output (business rules 1-7), dashboard checks
  dashboard/   FastAPI backend (data.py: all view logic) + static single-page UI
```

## Method (short)

### 1. Cleaning
* **Keys & labels**: SKU IDs normalised (`thm008`, `HYD_006`, `BSS-38` → `AAA-NNN`); ~30 region
  spellings and ~17 channel spellings mapped; 208 customer-name variants (case, spacing,
  `Inc.` suffix, typos) mapped to the 12 customers in `test.csv`; 7 date formats parsed.
* **Duplicates**: 3,682 repeated `Record_ID`s; 732 were real revisions (different units) →
  latest `Load_Timestamp` kept (rule 2).
* **Dates**: `Date` is authoritative (rule 1); 30 impossible dates (`2062-01`, a digit swap)
  repaired from the load timestamp (every row loads the month after its delivery month).
* **Price / units / revenue** (rule 4: one list price per SKU per year, same in all regions,
  plus the 2022 H2 6 % surcharge):
  * price ×100 (cents) — revenue correct → price fixed;
  * price ×10 with units ÷10 — packs-of-10 unit-of-measure error → units = revenue / price
    (confirmed against `inventory_stockouts.Shipped_Units`);
  * Europe rows booked in EUR (price and revenue ×0.89) → converted;
  * negative units with negative revenue of normal magnitude → sign error;
  * units ×10 vs revenue → the side closer to the series' local level is trusted.
* **Missing vs zero** (rule 3): a full monthly SKU × Region panel marks missing rows as NaN,
  never 0. The LatAm ERP migration (EV-033) shows up as 379 rows with blank units *and*
  revenue — treated as a data gap, not zero demand.
* **Lifecycle** (rule 5): rows before launch dropped; master data error fixed (PMP-004 had the
  wrong business unit — each family belongs to one BU).
* **Macro**: duplicate 2022-06 row, PMI 4.79 (decimal shift → 47.9), 2019 auto-production YoY
  stored as fractions (×100), 5 missing values interpolated.

### 2. Anomalies & unconstrained demand
* Listed stockouts → demand = `Requested_Units`; back-orders removed from the following month.
* The stockout file is incomplete: unlisted Electrical Assemblies months inside the
  semiconductor allocation (EV-019) show the same ~0.65 shipment ratio → demand = shipments ÷
  family fill rate of the listed rows that month.
* Spikes/dips are scored against local level × seasonal index and attributed to the events
  calendar (holidays, COVID, strikes, logistics, price rises). One-off orders stay in history
  but are capped in the model target; COVID months are down-weighted.
* Structural breaks (ruptures on family totals), series level shifts, 23 customer switches,
  ordering-cycle changes, lost accounts.

### 3. Drivers
Family demand growth vs each indicator at lags 0–12 (COVID excluded) → leading indicator per
family; the same analysis in three windows tests lag stability; demand by powertrain shows
the EV shift.

### 4. Forecasting
* Two baselines (seasonal naive, 12-month moving average), seasonal-index × level, TSB for
  intermittent demand, and a **global LightGBM, direct multi-horizon**: one row per series ×
  origin × horizon (1–19), features computed only from data up to the origin; calendar,
  macro (available for the forecast window), events and lifecycle at the target month.
  Target scaled by the series level with scale weights so the loss tracks WAPE.
* Macro change-from-origin features: for every indicator and leading lag (0, 2, 3, 5
  months, from the drivers analysis), the change between the value that drives the
  target month and the value that drove the origin (log change for index-type
  indicators). This lets the model carry the macro outlook into the 19-month horizon
  instead of extrapolating recent history.
* EV-shift features: the log change in the EV adoption index between origin and target
  month (lags 0, 3, 6), given only to EV-only series and, as a separate feature, only to
  ICE-only series (universal parts get 0). Plus 36 / 48-month means and a 12-month
  median as longer, more robust level anchors.
* Point forecast = Tweedie GBM, average of 3 seeds, × one-off uplift: one-off orders are
  capped in the training target but keep happening, so the forecast is scaled by
  demand ÷ capped target over the last 24 months before the origin (1.010–1.013).
* P10/P90 from quantile GBMs (SKU × Region level).
* Aggregate bands (family, region, total): bottom-up P50 × empirical 10th / 90th
  percentile of actual ÷ forecast for that aggregate in the backtest, by horizon bucket
  (summing SKU-level P10/P90 would be far too wide).
* Rolling-origin backtest, 4 origins, full 19-month horizon, WAPE and bias by family, region
  and horizon.
* Known future events enter as target-month features. Example: the Orion Driveline North
  America retooling shutdown (EV-062, 2027-07..08) is forecast at ~0.22× its normal volume,
  the same ratio as the only precedent (Nexon Europe strike, 2023-10..11), with no catch-up
  month afterwards (none followed the Nexon strike either).
* Lifecycle rules: zero after EOL / before launch, EOL run-down and new-launch ramp-up curves
  learned from the 6 historical successor pairs (`Replaces_SKU`).
* SKU-level forecasts are summed bottom-up, so family / region / total figures add up.
