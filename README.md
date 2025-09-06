# \$ONE x Save Finance Simulation

This project provides a configurable simulator for modeling the impact of a **Save Finance permissionless lending pool** on a \$ONE/USDC liquidity ecosystem.

It combines:

* A **Save-style lending market** (kinked borrow curve, utilization-based rates, emissions).
* A **constant-product AMM (DEX)** for \$ONE/USDC trading.
* Feedback loops where liquidity providers and borrowers react to APYs, shifting liquidity and affecting price/depth.

---

## Features

* **Two-scenario comparison**:

  * Baseline (no emissions)
  * With emissions (configurable daily rewards)
* **Grid simulation**:

  * Low / Medium / High emissions
  * Low / Medium / High borrow inflow rates
  * Produces CSVs + heatmaps showing price, liquidity depth, utilization changes
* **Visualizations**:

  * ONE/USDC price path
  * DEX reserves & depth (slippage resistance)
  * Save pool utilization & APY
* **Outputs**:

  * CSVs of all time-series and summaries
  * PNG plots for scenarios and heatmaps
  * JSON of the exact config used

---

## Installation

Requires Python 3.9+ with:

```bash
pip install numpy pandas matplotlib
```

---

## Usage

Run the simulator:

```bash
python simulation.py
```

All results are written to the `outputs/` folder:

* `two_scenario/` → Baseline vs With Emissions (charts + CSVs)
* `grid/` → Emissions × Borrow inflow grid (heatmaps + CSV)
* `config_used.json` → snapshot of all parameters used

---

## Configuration

At the top of `simulation.py` you’ll find a `CONFIG` dictionary.
Edit values to adjust scenarios:

* **Market / AMM**
  `P0`, `one_reserve0`, `dex_fee`, `daily_volume_as_tvls`

* **Save pool parameters**
  `optimal_util`, `min_borrow_apr`, `optimal_borrow_apr`, `max_borrow_apr`
  `rf_low_util`, `rf_kink`, `rf_low_val`, `rf_mid_val`, `rf_high_val`
  `save_supply_usd`, `deposit_limit_usd`, `daily_borrow_inflow_pct`

* **Behavioral response**
  `apy_sensitivity` (how fast LPs chase yield differences)
  `theta_long` / `theta_short` (borrow flows into ONE buys or sells)
  `allow_one_borrowing` (toggle short-side)

* **Emissions**
  `emissions_base_usd_per_day` (baseline daily value)
  `emissions_decay_breakpoints` (schedule)
  `emissions_low_util_boost` (extra reward when util < 40%)
  `grid_emissions_levels` / `grid_borrow_levels` (for heatmaps)

---

## Outputs

Example files generated:

```
outputs/
  config_used.json
  two_scenario/
    scenario_A_baseline.csv
    scenario_B_with_emissions.csv
    end_of_horizon_summary.csv
    price_paths.png
    dex_one_reserve.png
    depth_to_plus1.png
    save_supplier_apy.png
    save_utilization.png
  grid/
    grid_results.csv
    heatmap_price.png
    heatmap_depth.png
    heatmap_util.png
```

---

## Interpreting Results

* **Price path**: Mid-price evolution in the AMM under borrow-driven flows.
* **Depth to +1%**: How much USDC is required to move price up by 1% (liquidity metric).
* **Save Utilization**: Borrow demand vs supply cap.
* **APY curves**: Supplier APY with and without emissions.
* **Heatmaps**: Show how emissions and borrow inflow levels affect:

  * Final price (Δ% vs baseline)
  * Buy-side depth (Δ% vs baseline)
  * Utilization (absolute percentage point change)

---

## Next Steps

* Plug in **realistic \$ONE pool reserves** (Raydium/Meteora data).
* Adjust borrow inflow scenarios for volatility regimes.
* Add **borrower-side emissions** or **dual-token reward schedules**.
* Extend to multi-asset collateral (e.g. USDC + ONE cross-collateralization).


