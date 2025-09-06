
"""
simulation_dual_emissions.py — $ONE x Save Finance dual‑token incentives
Author: ChatGPT

Adds:
- Dual-token reward schedules:
  * Token A → supplier emissions (boost Save supply APY)
  * Token B → borrower emissions (offset borrow APR / boost demand)
- Buyer-attracting config: directs borrow flow to buy ONE on DEX and tunes sensitivities.
- Optional demand response: borrower emissions increase daily new borrow inflow.

Outputs:
- Two scenarios (No emissions vs Dual emissions) with plots + CSVs
- A compact grid for borrower-emission intensity vs supplier-emission intensity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict

# =========================
# ======== CONFIG =========
# =========================
CONFIG = dict(
    # Horizon
    T=60,

    # AMM / Market
    P0=88_000.0,
    one_reserve0=60.0,               # slightly deeper initial ONE liquidity
    dex_fee=0.003,
    daily_volume_as_tvls=0.30,       # a bit higher baseline activity to reward LPs
    lp_external_emissions_usd_per_day=0.0,

    # Save rate curve
    optimal_util=0.80,
    min_borrow_apr=0.00,
    optimal_borrow_apr=0.18,
    max_borrow_apr=1.50,

    # Reserve factor schedule (optimize for generous supplier yield below kink)
    rf_low_util=0.40,
    rf_kink=0.80,
    rf_low_val=0.03,     # 3% RF under-util → more to suppliers early
    rf_mid_val=0.10,
    rf_high_val=0.22,

    # Save supply & borrow dynamics
    save_supply_usd=2_500_000.0,
    save_borrow_util0=0.28,
    daily_borrow_inflow_pct=0.012,    # 1.2%/day base demand
    borrow_fee=0.001,
    deposit_limit_usd=4_000_000.0,

    # Ecosystem stocks
    total_one_liquidity=100.0,
    total_usdc_liquidity=4_000_000.0,

    # Behavioral response
    apy_sensitivity=3.2,
    theta_long=0.78,                  # stronger bias to buy ONE with borrowed USDC
    theta_short=0.22,
    allow_one_borrowing=False,

    # ===== Dual Emissions =====
    # Token A → Suppliers
    tokenA_price=2.0,
    tokenA_base_per_day=8_000.0,      # USD/day at mark price
    tokenA_decay_breakpoints=(30, 60),
    tokenA_low_util_boost=1.30,
    # Token B → Borrowers
    tokenB_price=1.0,
    tokenB_base_per_day=10_000.0,     # USD/day at mark price
    tokenB_decay_breakpoints=(30, 60),
    tokenB_high_util_boost=1.25,      # extra when util > 70% to keep demand strong

    # Borrower demand response (how emissions improve borrow inflow)
    demand_alpha=1.50,        # sensitivity multiplier
    demand_threshold_apy=0.05,# only boosts when borrower subsidy APY > 5%
    demand_cap=0.75,          # cap +75% boost to inflow

    # Grid knobs (optional small grid)
    grid_supplier_tokenA=[4000.0, 8000.0, 16000.0],
    grid_borrower_tokenB=[4000.0, 10000.0, 20000.0],

    # Output directory
    output_dir="outputs_dual"
)

# =========================
# ====== CORE LOGIC =======
# =========================
@dataclass
class RateModel:
    optimal_util: float
    min_borrow_apr: float
    optimal_borrow_apr: float
    max_borrow_apr: float
    rf_low_util: float
    rf_kink: float
    rf_low_val: float
    rf_mid_val: float
    rf_high_val: float

    def borrow_apr(self, u: float) -> float:
        u = max(0.0, min(1.0, u))
        if u <= self.optimal_util:
            if self.optimal_util == 0:
                return self.min_borrow_apr
            return self.min_borrow_apr + (self.optimal_borrow_apr - self.min_borrow_apr) * (u / self.optimal_util)
        rem = (u - self.optimal_util) / max(1e-9, (1.0 - self.optimal_util))
        return self.optimal_borrow_apr + rem * (self.max_borrow_apr - self.optimal_borrow_apr)

    def reserve_factor(self, u: float) -> float:
        if u < self.rf_low_util:
            return self.rf_low_val
        if u <= self.rf_kink:
            return self.rf_mid_val
        return self.rf_high_val

    def supply_apr(self, u: float) -> float:
        return self.borrow_apr(u) * u * (1.0 - self.reserve_factor(u))


def tokenA_fn_factory(base_usd_per_day: float, bp=(30,60), low_util_boost=1.30):
    d1, d2 = bp
    def _fn(day:int, util:float) -> float:
        if day <= d1: v = base_usd_per_day
        elif day <= d2: v = base_usd_per_day * 0.60
        else: v = base_usd_per_day * 0.35
        if util < 0.40: v *= low_util_boost
        return v
    return _fn

def tokenB_fn_factory(base_usd_per_day: float, bp=(30,60), high_util_boost=1.25):
    d1, d2 = bp
    def _fn(day:int, util:float) -> float:
        if day <= d1: v = base_usd_per_day
        elif day <= d2: v = base_usd_per_day * 0.60
        else: v = base_usd_per_day * 0.35
        if util > 0.70: v *= high_util_boost
        return v
    return _fn

# AMM helpers
def pool_price(x,y): return y/x if x>0 else float("inf")
def swap_usdc_for_one(x,y,usdc_in,fee):
    usdc_in_eff = usdc_in*(1.0-fee); k=x*y; new_y=y+usdc_in_eff; new_x=k/new_y; one_out=x-new_x
    return max(new_x,1e-9), max(new_y,1e-9), max(one_out,0.0)
def swap_one_for_usdc(x,y,one_in,fee):
    one_in_eff = one_in*(1.0-fee); k=x*y; new_x=x+one_in_eff; new_y=k/new_x; usdc_out=y-new_y
    return max(new_x,1e-9), max(new_y,1e-9), max(usdc_out,0.0)
def lp_fee_apy(daily_volume,tvl,fee):
    if tvl<=0: return 0.0
    return (daily_volume*fee/tvl)*365.0
def depth_to_move_up_1pct(x,y,fee):
    p0=pool_price(x,y); target=p0*1.01; usdc=0.0
    for _ in range(100):
        nx,ny,_=swap_usdc_for_one(x,y,usdc,fee); p=pool_price(nx,ny)
        if p>=target: break
        usdc+=max(y*0.001,1.0)
    return usdc
def soft_share(save_apy,lp_apy,sens):
    delta=save_apy-lp_apy
    return 1.0/(1.0+np.exp(-sens*delta))

# =========================
# ====== SCENARIO =========
# =========================
def run_scenario(cfg:Dict, tokenA_usd_per_day:float=0.0, tokenB_usd_per_day:float=0.0):
    # Unpack
    T=cfg["T"]; P0=cfg["P0"]; x=float(cfg["one_reserve0"]); y=float(cfg["one_reserve0"]*P0)
    dex_fee=cfg["dex_fee"]; daily_volume_as_tvls=cfg["daily_volume_as_tvls"]; lp_ext=cfg["lp_external_emissions_usd_per_day"]
    rm=RateModel(cfg["optimal_util"],cfg["min_borrow_apr"],cfg["optimal_borrow_apr"],cfg["max_borrow_apr"],
                 cfg["rf_low_util"],cfg["rf_kink"],cfg["rf_low_val"],cfg["rf_mid_val"],cfg["rf_high_val"])
    save_supply=min(cfg["save_supply_usd"],cfg["deposit_limit_usd"]); borrow_balance=save_supply*cfg["save_borrow_util0"]
    base_inflow=cfg["daily_borrow_inflow_pct"]
    total_one_liquidity=cfg["total_one_liquidity"]; apy_sensitivity=cfg["apy_sensitivity"]
    theta_long=cfg["theta_long"]; allow_one_borrowing=cfg["allow_one_borrowing"]; theta_short=cfg["theta_short"]
    one_idle=max(total_one_liquidity-x,0.0)

    tokenA_fn = tokenA_fn_factory(tokenA_usd_per_day, cfg["tokenA_decay_breakpoints"], cfg["tokenA_low_util_boost"])
    tokenB_fn = tokenB_fn_factory(tokenB_usd_per_day, cfg["tokenB_decay_breakpoints"], cfg["tokenB_high_util_boost"])

    rows=[]
    for day in range(1, T+1):
        price=pool_price(x,y); dex_tvl=x*price+y; daily_vol=daily_volume_as_tvls*dex_tvl
        lp_apy=lp_fee_apy(daily_vol,dex_tvl,dex_fee)+(lp_ext/dex_tvl*365.0 if dex_tvl>0 else 0.0)

        util=min(borrow_balance/max(save_supply,1e-9),1.0)
        supply_apr=rm.supply_apr(util)
        # Emissions
        tokenA_usd = tokenA_fn(day, util) if tokenA_usd_per_day>0 else 0.0
        tokenB_usd = tokenB_fn(day, util) if tokenB_usd_per_day>0 else 0.0
        save_apy_from_tokenA = (tokenA_usd/max(save_supply,1e-9))*365.0
        save_apy_total = supply_apr + save_apy_from_tokenA

        # Borrow demand response from Token B (convert to APY-per-supply proxy then boost inflow)
        borrower_subsidy_apy = (tokenB_usd/max(save_supply,1e-9))*365.0  # proxy
        boost = 0.0
        if borrower_subsidy_apy > cfg["demand_threshold_apy"]:
            boost = min(cfg["demand_alpha"] * (borrower_subsidy_apy - cfg["demand_threshold_apy"]), cfg["demand_cap"])
        daily_borrow_inflow_pct = base_inflow * (1.0 + boost)

        # ONE LP share shift — suppliers chasing APY
        share_to_save=soft_share(save_apy_total, lp_apy, apy_sensitivity)
        one_shift=(share_to_save-0.5)*0.10*total_one_liquidity
        max_withdrawable=max(x-6.0,0.0) # keep a slightly larger buffer
        if one_shift>0:
            move=min(one_shift,max_withdrawable); x-=move; one_idle+=move
        else:
            move=min(-one_shift,one_idle); x+=move; one_idle-=move
        y=x*price

        # Borrow accrual + inflow (after emissions boost)
        r_b=rm.borrow_apr(util)
        new_borrow_raw = save_supply*daily_borrow_inflow_pct
        remaining_liq=max(save_supply-borrow_balance,0.0)
        new_borrow=min(new_borrow_raw, remaining_liq)
        borrow_balance+=borrow_balance*(r_b/365.0)
        borrow_balance+=new_borrow

        # Directional flow to DEX: buy pressure from new USDC borrows
        usdc_for_buys=new_borrow*theta_long
        if usdc_for_buys>0:
            x,y,_=swap_usdc_for_one(x,y,usdc_for_buys,dex_fee)

        if allow_one_borrowing:
            one_for_sells=(new_borrow*theta_short)/max(price,1e-9)
            if one_for_sells>0: x,y,_=swap_one_for_usdc(x,y,one_for_sells,dex_fee)

        depth_up_1pct=depth_to_move_up_1pct(x,y,dex_fee)

        rows.append({
            "Day":day, "Price":pool_price(x,y), "DEX ONE Reserve":x, "DEX USDC Reserve":y,
            "DEX TVL (USD)":dex_tvl, "LP APY (fees+emissions)":lp_apy*100.0,
            "Save Utilization":util, "Save Supply (USD)":save_supply,
            "Save Borrow Balance (USD)":borrow_balance,
            "Supplier Emissions USD/day": tokenA_usd,
            "Borrower Emissions USD/day": tokenB_usd,
            "Supplier APY Emissions (pp)": save_apy_from_tokenA*100.0,
            "Borrower Subsidy APY (proxy, pp)": borrower_subsidy_apy*100.0,
            "Save APY (total)": save_apy_total*100.0,
            "Depth to +1% (USDC)": depth_up_1pct,
            "ONE idle": one_idle,
            "Borrow inflow pct (effective)": daily_borrow_inflow_pct*100.0
        })

        # Grow Save supply when attractive
        if save_apy_total > lp_apy:
            save_supply=min(save_supply*1.01, cfg["deposit_limit_usd"])

    return pd.DataFrame(rows)

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_df(df, path): df.to_csv(path, index=False)

def plot_series(xs, ys, title, xlabel, ylabel, out_path, legend=None):
    plt.figure()
    for i,y in enumerate(ys):
        plt.plot(xs, y, label=None if not legend else legend[i])
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True)
    if legend: plt.legend()
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def run_dual_scenarios(cfg:Dict, outdir:str):
    ensure_dir(outdir)
    # Scenario 1: No emissions
    s0 = run_scenario(cfg, 0.0, 0.0)
    # Scenario 2: Dual emissions (TokenA to suppliers, TokenB to borrowers)
    s1 = run_scenario(cfg, cfg["tokenA_base_per_day"], cfg["tokenB_base_per_day"])

    save_df(s0, os.path.join(outdir, "scenario0_no_emissions.csv"))
    save_df(s1, os.path.join(outdir, "scenario1_dual_emissions.csv"))

    # Summary
    summary = pd.DataFrame({
        "Metric":[
            "Final Price (USD)","DEX ONE Reserve","DEX USDC Reserve","DEX TVL (USD)",
            "Depth to +1% (USDC)","Save Utilization","Save Borrow Balance (USD)",
            "Supplier APY Emissions (pp)","Borrower Subsidy APY (pp)","Borrow inflow pct (effective)"
        ],
        "No Emissions":[
            s0["Price"].iloc[-1], s0["DEX ONE Reserve"].iloc[-1], s0["DEX USDC Reserve"].iloc[-1], s0["DEX TVL (USD)"].iloc[-1],
            s0["Depth to +1% (USDC)"].iloc[-1], s0["Save Utilization"].iloc[-1], s0["Save Borrow Balance (USD)"].iloc[-1],
            s0["Supplier APY Emissions (pp)"].iloc[-1] if "Supplier APY Emissions (pp)" in s0 else 0.0,
            s0["Borrower Subsidy APY (proxy, pp)"].iloc[-1] if "Borrower Subsidy APY (proxy, pp)" in s0 else 0.0,
            s0["Borrow inflow pct (effective)"].iloc[-1]
        ],
        "Dual Emissions":[
            s1["Price"].iloc[-1], s1["DEX ONE Reserve"].iloc[-1], s1["DEX USDC Reserve"].iloc[-1], s1["DEX TVL (USD)"].iloc[-1],
            s1["Depth to +1% (USDC)"].iloc[-1], s1["Save Utilization"].iloc[-1], s1["Save Borrow Balance (USD)"].iloc[-1],
            s1["Supplier APY Emissions (pp)"].iloc[-1], s1["Borrower Subsidy APY (proxy, pp)"].iloc[-1],
            s1["Borrow inflow pct (effective)"].iloc[-1]
        ]
    })
    save_df(summary, os.path.join(outdir, "summary.csv"))

    xs = s0["Day"]
    plot_series(xs, [s0["Price"], s1["Price"]], "ONE/USDC Price (Dual Emissions vs None)", "Day", "Price (USD)",
                os.path.join(outdir, "price.png"), legend=["No Emissions","Dual Emissions"])
    plot_series(xs, [s0["Depth to +1% (USDC)"], s1["Depth to +1% (USDC)"]], "Buy-Side Depth to Move Price +1%", "Day", "USDC depth",
                os.path.join(outdir, "depth.png"), legend=["No Emissions","Dual Emissions"])
    plot_series(xs, [s0["Save Utilization"], s1["Save Utilization"]], "Save Utilization", "Day", "Utilization",
                os.path.join(outdir, "util.png"), legend=["No Emissions","Dual Emissions"])
    plot_series(xs, [s1["Supplier APY Emissions (pp)"], s1["Borrower Subsidy APY (proxy, pp)"]],
                "Emissions APY (Suppliers vs Borrowers)", "Day", "percentage points",
                os.path.join(outdir, "emissions_apys.png"), legend=["Supplier APY (pp)","Borrower Subsidy APY (pp)"])
    plot_series(xs, [s0["Borrow inflow pct (effective)"], s1["Borrow inflow pct (effective)"]],
                "Effective Borrow Inflow % (daily)", "Day", "% of supply",
                os.path.join(outdir, "borrow_inflow_pct.png"), legend=["No Emissions","Dual Emissions"])

    return s0, s1, summary


def run_mini_grid(cfg:Dict, outdir:str):
    ensure_dir(outdir)
    rows=[]
    for a in cfg["grid_supplier_tokenA"]:
        for b in cfg["grid_borrower_tokenB"]:
            df = run_scenario(cfg, a, b)
            rows.append({
                "SupplierTokenA_USDday": a,
                "BorrowerTokenB_USDday": b,
                "FinalPrice": df["Price"].iloc[-1],
                "DepthTo+1%": df["Depth to +1% (USDC)"].iloc[-1],
                "Utilization": df["Save Utilization"].iloc[-1],
                "BorrowInflowEff%": df["Borrow inflow pct (effective)"].iloc[-1]
            })
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(outdir, "grid_dual_emissions.csv"), index=False)
    return grid


def main(cfg:Dict):
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    import json; open(os.path.join(out,"config_used.json"),"w").write(json.dumps(cfg, indent=2))

    s0,s1,sumdf = run_dual_scenarios(cfg, os.path.join(out,"dual_scenarios"))
    grid = run_mini_grid(cfg, os.path.join(out,"mini_grid"))
    print("Done. Outputs in", os.path.abspath(out))


if __name__ == "__main__":
    main(CONFIG)
