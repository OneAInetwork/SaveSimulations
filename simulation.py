
# (same content as before, compressed for brevity in this cell)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict

CONFIG = dict(
    T=60, P0=88000.0, one_reserve0=50.0, dex_fee=0.003, daily_volume_as_tvls=0.25,
    lp_external_emissions_usd_per_day=0.0,
    optimal_util=0.80, min_borrow_apr=0.00, optimal_borrow_apr=0.18, max_borrow_apr=1.50,
    rf_low_util=0.50, rf_kink=0.80, rf_low_val=0.05, rf_mid_val=0.10, rf_high_val=0.20,
    save_supply_usd=2_000_000.0, save_borrow_util0=0.30, daily_borrow_inflow_pct=0.01,
    borrow_fee=0.001, deposit_limit_usd=3_000_000.0, total_one_liquidity=80.0,
    total_usdc_liquidity=3_000_000.0, apy_sensitivity=3.0, theta_long=0.7, theta_short=0.3,
    allow_one_borrowing=False, emissions_base_usd_per_day=10_000.0,
    emissions_decay_breakpoints=(30, 45), emissions_low_util_boost=1.20,
    grid_emissions_levels={"Low": 5000.0, "Med": 10000.0, "High": 20000.0},
    grid_borrow_levels={"Low": 0.0025, "Med": 0.01, "High": 0.03},
    output_dir="outputs"
)

@dataclass
class RateModel:
    optimal_util: float; min_borrow_apr: float; optimal_borrow_apr: float; max_borrow_apr: float
    rf_low_util: float; rf_kink: float; rf_low_val: float; rf_mid_val: float; rf_high_val: float
    def borrow_apr(self, u):
        u = max(0.0, min(1.0, u))
        if u <= self.optimal_util:
            if self.optimal_util == 0: return self.min_borrow_apr
            return self.min_borrow_apr + (self.optimal_borrow_apr - self.min_borrow_apr) * (u / self.optimal_util)
        rem = (u - self.optimal_util) / max(1e-9, (1.0 - self.optimal_util))
        return self.optimal_borrow_apr + rem * (self.max_borrow_apr - self.optimal_borrow_apr)
    def reserve_factor(self, u):
        if u < self.rf_low_util: return self.rf_low_val
        if u <= self.rf_kink: return self.rf_mid_val
        return self.rf_high_val
    def supply_apr(self, u):
        return self.borrow_apr(u) * u * (1.0 - self.reserve_factor(u))

def emissions_fn_factory(base_per_day, bp=(30,45), low_util_boost=1.20):
    d1, d2 = bp
    def _fn(day, util):
        if day <= d1: v = base_per_day
        elif day <= d2: v = base_per_day * 0.60
        else: v = base_per_day * 0.30
        if util < 0.40: v *= low_util_boost
        return v
    return _fn

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

def soft_share(save_apy, lp_apy, sens):
    delta = save_apy - lp_apy
    return 1.0/(1.0+np.exp(-sens*delta))

def run_scenario(cfg, emissions_on, emissions_base_usd_per_day=None):
    T=cfg["T"]; P0=cfg["P0"]; x=float(cfg["one_reserve0"]); y=float(cfg["one_reserve0"]*P0)
    dex_fee=cfg["dex_fee"]; daily_volume_as_tvls=cfg["daily_volume_as_tvls"]
    lp_ext=cfg["lp_external_emissions_usd_per_day"]
    rm=RateModel(cfg["optimal_util"],cfg["min_borrow_apr"],cfg["optimal_borrow_apr"],cfg["max_borrow_apr"],
                 cfg["rf_low_util"],cfg["rf_kink"],cfg["rf_low_val"],cfg["rf_mid_val"],cfg["rf_high_val"])
    save_supply=min(cfg["save_supply_usd"],cfg["deposit_limit_usd"]); borrow_balance=save_supply*cfg["save_borrow_util0"]
    daily_borrow_inflow_pct=cfg["daily_borrow_inflow_pct"]
    total_one_liquidity=cfg["total_one_liquidity"]; apy_sensitivity=cfg["apy_sensitivity"]
    theta_long=cfg["theta_long"]; allow_one_borrowing=cfg["allow_one_borrowing"]; theta_short=cfg["theta_short"]
    one_idle=max(total_one_liquidity-x,0.0)
    base=cfg["emissions_base_usd_per_day"] if emissions_base_usd_per_day is None else emissions_base_usd_per_day
    efn=emissions_fn_factory(base,cfg["emissions_decay_breakpoints"],cfg["emissions_low_util_boost"])
    rows=[]
    for day in range(1,T+1):
        price=y/x; dex_tvl=x*price+y; daily_vol=daily_volume_as_tvls*dex_tvl
        lp_apy=lp_fee_apy(daily_vol,dex_tvl,dex_fee)+(lp_ext/dex_tvl*365.0 if dex_tvl>0 else 0.0)
        util=min(borrow_balance/max(save_supply,1e-9),1.0)
        supply_apr=rm.supply_apr(util); emissions_value=efn(day,util) if emissions_on else 0.0
        save_apy=supply_apr+(emissions_value/max(save_supply,1e-9))*365.0
        share_to_save=soft_share(save_apy,lp_apy,apy_sensitivity)
        one_shift=(share_to_save-0.5)*0.10*total_one_liquidity; max_withdrawable=max(x-5.0,0.0)
        if one_shift>0:
            move=min(one_shift,max_withdrawable); x-=move; one_idle+=move
        else:
            move=min(-one_shift,one_idle); x+=move; one_idle-=move
        y=x*price
        r_b=rm.borrow_apr(util); new_borrow_raw=save_supply*daily_borrow_inflow_pct
        remaining_liq=max(save_supply-borrow_balance,0.0); new_borrow=min(new_borrow_raw,remaining_liq)
        borrow_balance+=borrow_balance*(r_b/365.0); borrow_balance+=new_borrow
        usdc_for_buys=new_borrow*theta_long
        if usdc_for_buys>0: x,y,_=swap_usdc_for_one(x,y,usdc_for_buys,dex_fee)
        if allow_one_borrowing:
            one_for_sells=(new_borrow*theta_short)/max(price,1e-9)
            if one_for_sells>0: x,y,_=swap_one_for_usdc(x,y,one_for_sells,dex_fee)
        depth_up_1pct=depth_to_move_up_1pct(x,y,dex_fee)
        rows.append({"Day":day,"Price":y/x,"DEX ONE Reserve":x,"DEX USDC Reserve":y,"DEX TVL (USD)":dex_tvl,
                     "LP APY (fees+emissions)":lp_apy*100.0,"Save Utilization":util,
                     "Save Supply (USD)":save_supply,"Save Borrow Balance (USD)":borrow_balance,
                     "Save APY (supply+emissions)":save_apy*100.0,"Save Emissions USD/day":emissions_value,
                     "Depth to +1% (USDC)":depth_up_1pct,"ONE idle (outside LP)":one_idle})
        if save_apy>lp_apy: save_supply=min(save_supply*1.01,cfg["deposit_limit_usd"])
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

def simple_heatmap(matrix, title, out_path):
    plt.figure(); plt.imshow(matrix.values, aspect="auto"); plt.title(title)
    plt.xlabel("Borrow Inflow"); plt.ylabel("Emissions")
    plt.xticks(range(len(matrix.columns)), matrix.columns); plt.yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j,i,f"{matrix.values[i,j]:.1f}%",ha="center",va="center")
    plt.colorbar(); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def run_two_scenario_and_plots(cfg, outdir):
    ensure_dir(outdir)
    baseline=run_scenario(cfg, emissions_on=False)
    with_em=run_scenario(cfg, emissions_on=True, emissions_base_usd_per_day=cfg["emissions_base_usd_per_day"])
    save_df(baseline, os.path.join(outdir, "scenario_A_baseline.csv"))
    save_df(with_em, os.path.join(outdir, "scenario_B_with_emissions.csv"))
    summary=pd.DataFrame({
        "Metric":[ "Final Price (USD)","Final DEX ONE Reserve","Final DEX USDC Reserve","Final DEX TVL (USD)",
                   "Final Depth to +1% (USDC)","Final Save Utilization","Final Save Borrow Balance (USD)",
                   "Final Save APY (supply+emissions) %"],
        "Baseline":[baseline["Price"].iloc[-1],baseline["DEX ONE Reserve"].iloc[-1],baseline["DEX USDC Reserve"].iloc[-1],
                    baseline["DEX TVL (USD)"].iloc[-1],baseline["Depth to +1% (USDC)"].iloc[-1],
                    baseline["Save Utilization"].iloc[-1],baseline["Save Borrow Balance (USD)"].iloc[-1],
                    baseline["Save APY (supply+emissions)"].iloc[-1]],
        "With Emissions":[with_em["Price"].iloc[-1],with_em["DEX ONE Reserve"].iloc[-1],with_em["DEX USDC Reserve"].iloc[-1],
                    with_em["DEX TVL (USD)"].iloc[-1],with_em["Depth to +1% (USDC)"].iloc[-1],
                    with_em["Save Utilization"].iloc[-1],with_em["Save Borrow Balance (USD)"].iloc[-1],
                    with_em["Save APY (supply+emissions)"].iloc[-1]]
    })
    save_df(summary, os.path.join(outdir, "end_of_horizon_summary.csv"))
    xs=baseline["Day"]
    plot_series(xs,[baseline["Price"],with_em["Price"]],"ONE/USDC Price (AMM mid)","Day","Price (USD)",
                os.path.join(outdir,"price_paths.png"),legend=["Baseline","With Emissions"])
    plot_series(xs,[baseline["DEX ONE Reserve"],with_em["DEX ONE Reserve"]],"DEX ONE Reserve Over Time","Day","ONE",
                os.path.join(outdir,"dex_one_reserve.png"),legend=["Baseline","With Emissions"])
    plot_series(xs,[baseline["Depth to +1% (USDC)"],with_em["Depth to +1% (USDC)"]],"Buy-Side Depth to Move Price +1%","Day","USDC depth",
                os.path.join(outdir,"depth_to_plus1.png"),legend=["Baseline","With Emissions"])
    plot_series(xs,[baseline["Save APY (supply+emissions)"],with_em["Save APY (supply+emissions)"]],"Save Supplier APY (incl. Emissions)","Day","APY (%)",
                os.path.join(outdir,"save_supplier_apy.png"),legend=["Baseline","With Emissions"])
    plot_series(xs,[baseline["Save Utilization"],with_em["Save Utilization"]],"Save Utilization Over Time","Day","Utilization",
                os.path.join(outdir,"save_utilization.png"),legend=["Baseline","With Emissions"])
    return baseline, with_em, summary

def run_grid_and_heatmaps(cfg, outdir):
    ensure_dir(outdir); emis=cfg["grid_emissions_levels"]; bor=cfg["grid_borrow_levels"]
    base_by_inflow={}
    for blabel, inflow in bor.items():
        cfg_tmp=dict(cfg); cfg_tmp["daily_borrow_inflow_pct"]=inflow
        base_by_inflow[blabel]=run_scenario(cfg_tmp, emissions_on=False)
    rows=[]
    for elabel, ebase in emis.items():
        for blabel, inflow in bor.items():
            cfg_tmp=dict(cfg); cfg_tmp["daily_borrow_inflow_pct"]=inflow
            df=run_scenario(cfg_tmp, emissions_on=True, emissions_base_usd_per_day=ebase)
            base=base_by_inflow[blabel]
            rows.append({
                "Emissions":elabel,"BorrowInflow":blabel,"FinalPrice":df["Price"].iloc[-1],
                "FinalDepthTo+1%":df["Depth to +1% (USDC)"].iloc[-1],"FinalDEXTVL":df["DEX TVL (USD)"].iloc[-1],
                "FinalUtil":df["Save Utilization"].iloc[-1],
                "ΔPrice% vs Baseline":(df["Price"].iloc[-1]/base["Price"].iloc[-1]-1.0)*100.0,
                "ΔDepth% vs Baseline":(df["Depth to +1% (USDC)"].iloc[-1]/base["Depth to +1% (USDC)"].iloc[-1]-1.0)*100.0,
                "ΔDEXTVL% vs Baseline":(df["DEX TVL (USD)"].iloc[-1]/base["DEX TVL (USD)"].iloc[-1]-1.0)*100.0,
                "ΔUtil (abs pts)":(df["Save Utilization"].iloc[-1]-base["Save Utilization"].iloc[-1])*100.0})
    grid_df=pd.DataFrame(rows); ensure_dir(outdir); grid_df.to_csv(os.path.join(outdir,"grid_results.csv"), index=False)
    price_pivot=grid_df.pivot(index="Emissions", columns="BorrowInflow", values="ΔPrice% vs Baseline")
    depth_pivot=grid_df.pivot(index="Emissions", columns="BorrowInflow", values="ΔDepth% vs Baseline")
    util_pivot=grid_df.pivot(index="Emissions", columns="BorrowInflow", values="ΔUtil (abs pts)")
    simple_heatmap(price_pivot,"Δ Price vs Baseline (%)",os.path.join(outdir,"heatmap_price.png"))
    simple_heatmap(depth_pivot,"Δ Buy-Side Depth vs Baseline (%)",os.path.join(outdir,"heatmap_depth.png"))
    simple_heatmap(util_pivot,"Δ Save Utilization (abs percentage points)",os.path.join(outdir,"heatmap_util.png"))
    return grid_df

def main(cfg):
    out=cfg["output_dir"]; os.makedirs(out,exist_ok=True)
    import json; open(os.path.join(out,"config_used.json"),"w").write(json.dumps(cfg,indent=2))
    run_two_scenario_and_plots(cfg, os.path.join(out,"two_scenario"))
    run_grid_and_heatmaps(cfg, os.path.join(out,"grid"))
    print("Done. Outputs in", os.path.abspath(out))

if __name__=="__main__":
    main(CONFIG)
