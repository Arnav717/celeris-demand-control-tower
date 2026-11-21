# Celeris Automobili — Demand Forecasting & Planning Control Tower

End-to-end pipeline for the Celeris Automobili Industrial case: clean a messy ERP extract,
separate data errors from real demand events, estimate unconstrained demand, forecast
`Units_Sold` for every row of `test.csv` (2026-11 → 2028-05, 19 months) and serve a
planning dashboard from the pipeline outputs.

## Quick start

```bash
# 1. environment (Python 3.12, uv)
make setup                       # or: uv venv --python 3.12 .venv && uv pip install -e ".[dev]"

# 2. data: put the organiser's zip at data/raw/student_data.zip, then
make data                        # extracts the CSVs into data/raw/

# 3. whole pipeline (clean -> anomalies -> features -> forecast), ~15 min on 16 cores
make all                         # = python -m celeris all

# 4. tests
make test
```

Run single stages with `python -m celeris clean|anomalies|features|forecast`.
Set `CELERIS_SKIP_BACKTEST=1` to skip the rolling-origin backtest and only refit the
final model.

## Outputs

| File | What it is |
|---|---|
| `outputs/submission.csv` | Final forecast, one row per `test.csv` row (`Date, SKU_ID, Region, Units_Sold_Forecast`) |
| `outputs/anomaly_register.csv` | Every flagged point / series / family event with type, category, reason, action |
| `reports/dq_report.json`, `reports/dq_issues.csv` | Machine-readable data-quality report: issue, file, rows affected, action, before/after counts |
| `reports/anomaly_summary.json` | Register counts, demand vs shipments uplift, price-rise buying |
| `reports/drivers_summary.json` | Leading indicator and lag per family, lag stability, EV shift |
| `reports/backtest_metrics.json` | Rolling-origin WAPE / bias by model, family, region, horizon |
| `data/processed/*.parquet` | Clean tables the dashboard reads (train_clean, panel, demand, forecast, backtest, drivers…) |

## Repository layout

```
src/celeris/
  clean/       labels, train cleaning, reference files, SKU x Region panel, DQ log
  anomalies/   unconstrained demand, outlier detection + event attribution, structural checks
  features/    direct multi-horizon feature matrix, macro lead/lag drivers, intermittency
  models/      baselines, LightGBM, rolling-origin backtest, lifecycle rules, final forecast
tests/         unit tests (labels) + invariants on every output (business rules 1-7)
app/           dashboard (reads data/processed only)
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
* Point forecast = mean of Tweedie and L1 models; P10/P90 from quantile models.
* Rolling-origin backtest, 4 origins, full 19-month horizon, WAPE and bias by family, region
  and horizon.
* Lifecycle rules: zero after EOL / before launch, EOL run-down and new-launch ramp-up curves
  learned from the 6 historical successor pairs (`Replaces_SKU`).
* SKU-level forecasts are summed bottom-up, so family / region / total figures add up.
