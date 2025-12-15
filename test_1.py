
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion LCOE sweep & plots — compact high-field tokamak

Features
--------
1) Physics:
   - Pfus ∝ B^4 * V (calibrated to SPARC-like anchor).
   - Simple torus geometry: V = 2 π^2 R a^2; a = R/aspect.

2) Magnet CAPEX (select one):
   - per_meter (recommended): HTS priced as $/m of cable/tape
   - per_kAm    (advanced):   HTS priced as $/(kA·m)
     NOTE: 'per_kAm' uses the total sum(length × cable_current), i.e., kA·m,
           which equals N_TF * (2πR)^2 * B0 / (μ0 * 1e3). It is independent of
           I_turn or cable ratings because it is the aggregate kA·m across *all*
           parallel cables. If you don't have realistic $/kA·m inputs, prefer
           per_meter (supply-chain tends to quote per meter of cable).

3) Blanket/Shield & Structure CAPEX:
   - Shield cost ~ surface area × shield thickness (effective volume) × $/m^3.
   - Structure/stress term ~ B^2 * R (captures force/mass trend).

4) BOP & O&M:
   - BOP $/kWe on net electric capacity.
   - Fixed/variable O&M included.

5) LCOE:
   - EIA CRF method.
   - Capacity factor (CF) included.

6) Plotting & Filtering:
   - Hard filter: drop LCOE > $3,000 from plots/filtered CSV.
   - Scatter plots: gold star marks every point ≤ $150/MWh.
   - Heatmap: clipped colormap; $150/MWh contour drawn if possible.

Adjust the "SCENARIO" block to match your cost book and design targets.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for scripts/servers
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy.ma as ma

# ----------------------------- Physics calibration -----------------------------
# SPARC-like anchor for Pfus ∝ B^4 * V
B_anchor = 12.2        # T
R_anchor = 1.85        # m
a_anchor = 0.57        # m
V_anchor = 2 * math.pi**2 * R_anchor * a_anchor**2
Pfus_anchor_W = 140e6  # W (thermal)
k_pf = Pfus_anchor_W / (B_anchor**4 * V_anchor)

mu0 = 4 * math.pi * 1e-7  # H/m

# ----------------------------- Helper functions -----------------------------
def crf(r: float, n_years: int) -> float:
    """Capital recovery factor (EIA/DOE)."""
    return (r * (1 + r) ** n_years) / ((1 + r) ** n_years - 1)

def fusion_plant_metrics(B0, R, aspect=3.2, eta_th=0.42, parasitic_MW=25.0, f_field=4.0):
    """
    Compute geometry, power, and stored magnetic energy (for diagnostics).
    Returns: a, V_plasma [m^3], Pfus [MW_th], Pe [MW_e net], Wmag [J]
    """
    a = R / aspect
    V_plasma = 2 * math.pi**2 * R * a**2
    Pfus_W = k_pf * (B0**4) * V_plasma
    Pe_MW = eta_th * Pfus_W / 1e6 - parasitic_MW
    # Diagnostic only; not used for capex if 'per_meter' mode selected
    V_field = f_field * V_plasma
    Wmag_J = (B0**2 / (2 * mu0)) * V_field
    return a, V_plasma, Pfus_W / 1e6, Pe_MW, Wmag_J

def tf_conductor_metrics(B0, R, N_TF=18, I_turn_kA=20.0):
    """
    Estimate TF conductor length and total kA·m for the toroidal field system.

    - Required ampere-turns: AT = B0 * 2πR / μ0  [A-turns]
    - Choose I_turn_kA (turn current in kA): N_eff = AT / (I_turn_A)
    - Per-turn length ~ 2πR; total length L_total = N_TF * N_eff * 2πR  [m]
    - Total kA·m across all cables:
        total_kAm_sum = N_TF * (2πR)^2 * B0 / (μ0 * 1e3)
      (equal to sum(length × cable_current_kA) across all cables).
    """
    AT_req_Aturns = B0 * 2 * math.pi * R / mu0  # A-turns
    I_turn_A = I_turn_kA * 1e3
    N_eff = AT_req_Aturns / I_turn_A
    per_turn_len_m = 2 * math.pi * R
    L_total_m = N_TF * N_eff * per_turn_len_m
    total_kAm_sum = N_TF * (per_turn_len_m**2) * B0 / (mu0 * 1e3)
    return {
        'AT_req_Aturns': AT_req_Aturns,
        'N_eff': N_eff,
        'L_total_m': L_total_m,
        'total_kAm': total_kAm_sum
    }

def capex_estimate(
    B0, R, a, Pe_MW, V_plasma_m3, Wmag_J,
    # ---- HTS costing mode & prices ----
    hts_cost_mode='per_meter',     # 'per_meter' (recommended) or 'per_kAm' or 'per_joule'
    price_per_meter=40.0,          # $/m (default placeholder for HTS cable/tape)
    price_per_kAm=200.0,           # $/(kA·m) — USE ONLY IF YOU HAVE REAL kA·m PRICING
    price_per_joule=0.1,           # $/J (legacy option; generally not used now)
    N_TF=18, I_turn_kA=20.0,
    # ---- Nuclear & structure ----
    t_shield_m=1.0,                # m (effective radial build for blanket/shield)
    c_shield_per_m3=4e5,           # $/m^3 (fabrication + integration)
    c_struct_stress_MUSD_per_T2m=2.0,  # M$/(T^2·m) — structural scaling
    # ---- BOP & adders ----
    c_BOP_per_kWe=1200,            # $/kWe for power block/BOP
    adders_MUSD=220,               # M$ (site, licensing, diagnostics, RH, T-systems)
    fudge_nuclear_multiplier=2.0   # integration multiplier
):
    # --- Magnet cost ---
    tfm = tf_conductor_metrics(B0, R, N_TF=N_TF, I_turn_kA=I_turn_kA)
    if hts_cost_mode == 'per_meter':
        magnet_MUSD = (price_per_meter * tfm['L_total_m']) / 1e6
    elif hts_cost_mode == 'per_kAm':
        # Cost directly from total kA·m (aggregate); ONLY if you truly price per (kA·m).
        magnet_MUSD = (price_per_kAm * tfm['total_kAm']) / 1e6
    elif hts_cost_mode == 'per_joule':
        magnet_MUSD = (price_per_joule * Wmag_J) / 1e6
    else:
        raise ValueError("hts_cost_mode must be 'per_meter', 'per_kAm', or 'per_joule'.")

    # --- Blanket/Shield cost ---
    S_torus = 4 * math.pi**2 * R * a  # torus surface
    V_shield_eff = S_torus * t_shield_m
    nuclear_base_MUSD = (c_shield_per_m3 * V_shield_eff) / 1e6

    # --- Structural/stress scaling ---
    struct_stress_MUSD = c_struct_stress_MUSD_per_T2m * (B0**2) * R

    nuclear_MUSD = fudge_nuclear_multiplier * (nuclear_base_MUSD + struct_stress_MUSD)

    # --- BOP ---
    bop_MUSD = (c_BOP_per_kWe * Pe_MW * 1e3) / 1e6

    total_MUSD = magnet_MUSD + nuclear_MUSD + bop_MUSD + adders_MUSD
    return {
        'total_MUSD': total_MUSD,
        'magnet_MUSD': magnet_MUSD,
        'nuclear_MUSD': nuclear_MUSD,
        'bop_MUSD': bop_MUSD,
        'struct_stress_MUSD': struct_stress_MUSD,
    }

def lcoe(capex_MUSD, fixed_OM_MUSD_per_year, var_OM_fuel_USD_per_MWh,
         Pe_MW, CF, lifetime_years=30, discount_rate=0.08):
    """LCOE ($/MWh)."""
    annual_MWh = Pe_MW * 1e3 * 8760 * CF / 1e3
    annualized_capex_USD = capex_MUSD * 1e6 * crf(discount_rate, lifetime_years)
    annual_fixed_USD = fixed_OM_MUSD_per_year * 1e6
    return (annualized_capex_USD + annual_fixed_USD) / annual_MWh + var_OM_fuel_USD_per_MWh

# ----------------------------- SCENARIO (edit me) -----------------------------
SCENARIO = dict(
    # --- HTS cost mode & pricing (recommended: 'per_meter') ---
    hts_cost_mode='per_meter',     # 'per_meter' | 'per_kAm' | 'per_joule'
    price_per_meter=40.0,          # $/m for cable/tape (adjust to supplier quote)
    price_per_kAm=200.0,           # $/(kA·m), use only with real kA·m pricing
    price_per_joule=0.1,           # legacy

    N_TF=18,
    I_turn_kA=20.0,

    # --- Nuclear & structure ---
    t_shield_m=1.0,                # m
    c_shield_per_m3=4e5,           # $/m^3
    c_struct=2.0,                  # M$/(T^2·m)

    # --- BOP & adders ---
    c_bop=1200,                    # $/kWe
    adders=220,                    # M$
    fudge=2.0,                     # integration multiplier

    # --- O&M and finance ---
    fixed_OM=60,                   # M$/yr
    var_OM=5,                      # $/MWh
    eta=0.42,                      # thermal-to-electric (sCO2 baseline)
    parasitic=25.0,                # MW
    CF=0.88,                       # capacity factor
    r=0.08,                        # discount rate
    nyrs=30                        # economic life
)

# ----------------------------- Sweep ranges -----------------------------
B_vals = np.arange(10, 16.1, 1.0)                 # T
R_vals = np.array([1.6,1.8,2.0,2.2,2.5,3.0,3.5,4.0])  # m

# ----------------------------- Run grid -----------------------------
records = []
for B0 in B_vals:
    for R in R_vals:
        a, Vp, Pfus, Pe, Wmag = fusion_plant_metrics(
            B0, R, aspect=3.2, eta_th=SCENARIO['eta'], parasitic_MW=SCENARIO['parasitic']
        )
        if Pe <= 0:
            continue
        cap = capex_estimate(
            B0, R, a, Pe, Vp, Wmag,
            hts_cost_mode=SCENARIO['hts_cost_mode'],
            price_per_meter=SCENARIO['price_per_meter'],
            price_per_kAm=SCENARIO['price_per_kAm'],
            price_per_joule=SCENARIO['price_per_joule'],
            N_TF=SCENARIO['N_TF'], I_turn_kA=SCENARIO['I_turn_kA'],
            t_shield_m=SCENARIO['t_shield_m'],
            c_shield_per_m3=SCENARIO['c_shield_per_m3'],
            c_struct_stress_MUSD_per_T2m=SCENARIO['c_struct'],
            c_BOP_per_kWe=SCENARIO['c_bop'],
            adders_MUSD=SCENARIO['adders'],
            fudge_nuclear_multiplier=SCENARIO['fudge']
        )
        L = lcoe(cap['total_MUSD'], SCENARIO['fixed_OM'], SCENARIO['var_OM'],
                 Pe, SCENARIO['CF'], SCENARIO['nyrs'], SCENARIO['r'])
        records.append({
            'B': B0, 'R': R, 'a': a, 'Pfus': Pfus, 'Pe': Pe, 'LCOE': L,
            'capex': cap['total_MUSD'],
            'cap_mag': cap['magnet_MUSD'], 'cap_nuc': cap['nuclear_MUSD'],
            'cap_bop': cap['bop_MUSD'], 'cap_struct': cap['struct_stress_MUSD']
        })

DF = pd.DataFrame(records)
DF.to_csv('fusion_lcoe_full_grid.csv', index=False)

# ----------------------------- Filter and save -----------------------------
# Hard line for visualization: exclude LCOE > $3,000/MWh
DFf = DF[DF['LCOE'] <= 3000].copy()
DFf.to_csv('fusion_lcoe_filtered.csv', index=False)

# ----------------------------- Plots -----------------------------
# 1) Scatter: R vs Pe, star-mark all target hits
fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
norm = Normalize(vmin=max(0.0, DFf['LCOE'].min()),
                 vmax=min(800.0, DFf['LCOE'].max()))  # clip palette to ≤ $800
sc = ax.scatter(DFf['R'], DFf['Pe'], c=DFf['LCOE'], s=90, cmap='viridis_r', norm=norm)
cb = plt.colorbar(sc, ax=ax)
cb.set_label('LCOE ($/MWh) — clipped to ≤ $800')
ax.set_title('R vs Net Electric Power (filtered to LCOE ≤ $3,000/MWh)')
ax.set_xlabel('Major radius R (m)')
ax.set_ylabel('Net electric power Pe (MW)')

hits = DFf[DFf['LCOE'] <= 150]
ax.scatter(hits['R'], hits['Pe'], marker='*', s=180, edgecolors='black',
           facecolors='gold', linewidths=1.2, label='≤ $150/MWh (target)')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig('scatter_R_vs_Pe_filtered.png')
plt.close(fig)

# 2) Scatter: R vs Pfus (log color map), star-mark target hits
fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
min_pos = max(1.0, float(DFf['LCOE'].min()))
sc = ax.scatter(DFf['R'], DFf['Pfus'], c=DFf['LCOE'], s=90, cmap='plasma_r',
                norm=LogNorm(vmin=min_pos, vmax=DFf['LCOE'].max()))
cb = plt.colorbar(sc, ax=ax)
cb.set_label('LCOE ($/MWh) — log scale')
ax.set_title('R vs Fusion Power (filtered to LCOE ≤ $3,000/MWh)')
ax.set_xlabel('Major radius R (m)')
ax.set_ylabel('Fusion power Pfus (MW_th)')
ax.scatter(hits['R'], hits['Pfus'], marker='*', s=180, edgecolors='black',
           facecolors='cyan', linewidths=1.2, label='≤ $150/MWh (target)')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig('scatter_R_vs_Pfus_filtered.png')
plt.close(fig)

# 3) Heatmap: LCOE over (B, R) with $150 contour if possible
piv = DFf.pivot_table(index='B', columns='R', values='LCOE', aggfunc='min')
fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
arr = piv.values
if np.isfinite(arr).sum() == 0:
    ax.text(0.5, 0.5, 'No points within filtered range', transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='red')
else:
    vmin = np.nanmin(arr)
    vmax = min(800.0, np.nanmax(arr))  # clip to keep palette meaningful
    norm = Normalize(vmin=vmin, vmax=vmax)
    masked = ma.array(arr, mask=~np.isfinite(arr))
    im = ax.imshow(masked, origin='lower', aspect='auto',
                   extent=[piv.columns.min()-0.05, piv.columns.max()+0.05,
                           piv.index.min()-0.05,  piv.index.max()+0.05],
                   cmap='viridis_r', norm=norm)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('LCOE ($/MWh) — clipped to ≤ $800')
    try:
        Rs = np.array(list(piv.columns))
        Bs = np.array(list(piv.index))
        Rg, Bg = np.meshgrid(Rs, Bs)
        cs = ax.contour(Rg, Bg, arr, levels=[150], colors='red', linewidths=1.8)
        ax.clabel(cs, fmt={150: '$150/MWh'})
    except Exception:
        ax.text(0.5, 0.93, 'No valid points for $150 contour', transform=ax.transAxes,
                ha='center', va='center', color='red', fontsize=10)

ax.set_xlabel('Major radius R (m)')
ax.set_ylabel('Toroidal field B0 (T)')
ax.set_title('LCOE map over (B0, R) — filtered to LCOE ≤ $3,000/MWh')
fig.tight_layout()
# fig.savefig('heatmap_filtered.png')
# plt.close(fig)
plt.plot()
plt.show()

# ----------------------------- Console summary -----------------------------
targets = DFf[DFf['LCOE'] <= 150].sort_values('LCOE')
print("\nTargets (≤ $150/MWh) in filtered set:")
if targets.empty:
    print("  None in current settings. Consider lowering price_per_meter or increasing CF/η.")
else:
    print(targets[['B','R','Pfus','Pe','LCOE','capex']].to_string(index=False))
