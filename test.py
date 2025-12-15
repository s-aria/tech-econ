
# -*- coding: utf-8 -*-
"""
Alt-C: LCOE surface over (Major radius R, Fusion power Pf), plus mark Alt-C point.
ASSUMPTIONS ARE CLEARLY FLAGGED BELOW — replace with programme values to make it non-illustrative.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# User-tunable financial inputs
# -----------------------------
discount_rate = 0.07        # real WACC (illustrative) — replace with programme value
lifetime_years = 30         # Alt-C report uses 30 years total plant lifetime (consistent with PROCESS SPP) [placeholder]
availability = 0.75         # availability/capacity factor (placeholder; Alt-C expects improved maintenance access)
annual_OM_frac = 0.02       # O&M as fraction of CAPEX per year (placeholder); replace with your O&M

# -----------------------------
# Alt‑C thermodynamic constants from the report
# -----------------------------
eta_el = 0.393              # steam Rankine gross efficiency (Table 6)
heat_to_cycle_per_Pf = 2517.0/1750.0  # Alt‑C cycle heat budget per MW_fusion (Table 3 / Table 6)

# -----------------------------
# Recirculation (parasitic) model
# -----------------------------
# Baseline parasitic loads at Alt‑C point (from Table 6):
parasitic_base_no_EBW = 1028.1  # MWe
parasitic_base_with_EBW = 683.1 # MWe

use_ebw = True  # toggle EBW scenario
parasitic_base = parasitic_base_with_EBW if use_ebw else parasitic_base_no_EBW

# Simple scaling model:
#   P_recirc(R, Pf) = parasitic_base * [a_R*(R/3.6) + a_Pf*(Pf/1750) + a0]
# Coefficients below are placeholders chosen so that at (R=3.6, Pf=1750) we recover 'parasitic_base'.
a_R  = 0.50   # sensitivity of parasitics to size (Alt‑C shows TF ohmic ≈ linear in R at fixed Bt)
a_Pf = 0.30   # sensitivity to fusion power (more heat/loads throughout plant)
a0   = 0.20   # remaining fixed fraction (buildings, control, etc.)
# Normalization check:
assert abs(a_R + a_Pf + a0 - 1.0) < 1e-9

def recirc_power(R, Pf):
    return parasitic_base * (a_R*(R/3.6) + a_Pf*(Pf/1750.0) + a0)

# -----------------------------
# CAPEX scaling with major radius (illustrative)
# -----------------------------
# Alt‑C report: copper mass ~ R^2; many other systems scale sub‑linearly.
# Let CAPEX(R) = CAPEX0 * (R/3.6)**capex_exp. C0 is a placeholder — plug in your SPP baseline * percentage (Fig. 23).
capex_exp = 2.0
CAPEX0 = 10e9 * 0.6384   # £: example if SPP baseline were £10bn and Alt‑C (Rankine+EBW) were 63.84% of it (Fig. 23)  <-- REPLACE

def capex_R(R):
    return CAPEX0 * (R/3.6)**capex_exp

# Annual O&M (illustrative proportional to CAPEX; replace with your value)
def annual_OM(R):
    return annual_OM_frac * capex_R(R)

# Capital recovery factor
def crf(r, n):
    return r * (1+r)**n / ((1+r)**n - 1)

CRF = crf(discount_rate, lifetime_years)

# -----------------------------
# Net electric and LCOE
# -----------------------------
def net_electric_MWe(R, Pf):
    # Cycle heat from fusion + additional integrated heat sources (Alt‑C ratio) → gross electric
    gross_MWe = eta_el * heat_to_cycle_per_Pf * Pf
    # Subtract recirculation (parasitic) loads
    return gross_MWe - recirc_power(R, Pf)

def annual_net_MWh(R, Pf):
    return max(net_electric_MWe(R, Pf), 0.0) * 8760.0 * availability

def lcoe(R, Pf):
    Enet = annual_net_MWh(R, Pf)
    if Enet <= 0.0:
        return np.nan  # undefined if net export ≤ 0
    return (capex_R(R)*CRF + annual_OM(R)) / Enet  # £/MWh

# -----------------------------
# Build the grid and compute
# -----------------------------
R_vals  = np.linspace(2.8, 4.6, 60)       # m
Pf_vals = np.linspace(1200, 2000, 60)     # MW
RR, PP  = np.meshgrid(R_vals, Pf_vals)
LCOE    = lcoe(RR, PP)

# Alt‑C point
R_altc  = 3.6
Pf_altc = 1750.0
lcoe_altc = lcoe(R_altc, Pf_altc)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(9,6))
c = plt.pcolormesh(RR, PP, LCOE, shading='nearest', cmap='viridis')
plt.colorbar(c, label='LCOE ( £/MWh )  — illustrative until CAPEX/O&M are set')
plt.scatter([R_altc], [Pf_altc], marker='x', s=120, c='k', label='Alt‑C (3.6 m, 1750 MW)')
plt.xlabel('Major radius R (m)')
plt.ylabel('Fusion power $P_f$ (MW)')
plt.title('LCOE surface over (R, $P_f$) — Alt‑C marked')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('lcoe_grid.png', dpi=200)
print(f"Alt‑C point: LCOE={lcoe_altc:.1f} £/MWh (illustrative), Net MWe={net_electric_MWe(R_altc, Pf_altc):.1f} MWe")
