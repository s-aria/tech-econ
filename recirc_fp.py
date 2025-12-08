
import numpy as np
import matplotlib.pyplot as plt

# Nonlinear recirculation model: LCOE vs CAPEX with bending curves
# Aria: this shows how LCOE curves bend upward when recirculation grows nonlinearly with design field B

# ---------------------------
# Plant & economic assumptions (edit as needed)
# ---------------------------
P_gross_MW = 1000            # gross electrical capacity [MW]
CF = 0.90                    # capacity factor
WACC = 0.08                  # discount rate (real)
life_yrs = 30                # economic life [years]
FOM_per_kW_per_yr = 100      # fixed O&M per gross kW-year [$]
VOM_plus_fuel_per_MWh = 5    # variable O&M + fuel [$ / MWh]

# Capital recovery factor
CRF = WACC * (1 + WACC) ** life_yrs / ((1 + WACC) ** life_yrs - 1)

# CAPEX per *gross* kW range (design dial)
capex_per_kW = np.linspace(2000, 8000, 250)  # [$ / kW_gross]

# Map CAPEX to notional field B (proxy): CAPEX ∝ B^2 R^3, fix R to isolate nonlinearity
R_fixed = 8.0  # [m]
# Choose constants so B spans a realistic high-field range (e.g., 4–12 T)
# CAPEX_per_kW ~ k_cap * B^2  (with R^3 folded into k_cap)
k_cap = 45.0   # scaling factor to give plausible B for given CAPEX range
B_T = np.sqrt(capex_per_kW / k_cap)

# ---------------------------
# Recirculation models
# ---------------------------
# A) Baseline constant fractions (will give *linear* LCOE vs CAPEX)
const_fracs = [0.10, 0.20, 0.30]

# B) Field-dependent quadratic growth: f = f0 + k*(B/B_ref)^2 (nonlinear)
f0 = 0.08
B_ref = 6.0
k_quad = 0.06
f_quad = np.clip(f0 + k_quad * (B_T / B_ref)**2, 0.0, 0.60)

# C) Saturating (logistic-like) growth: starts low, rises, then saturates near f_max
f0_sat = 0.06
f_max = 0.45
B_sat = 9.0
f_sat = np.clip(f0_sat + f_max * (B_T / B_sat)**2 / (1.0 + (B_T / B_sat)**2), 0.0, 0.60)

# ---------------------------
# LCOE computation
# ---------------------------
hours_per_year = 8760
gross_kW = P_gross_MW * 1000
annual_fixed_OM = FOM_per_kW_per_yr * gross_kW

# Annualized CAPEX across the CAPEX range
annualized_capex = capex_per_kW * gross_kW * CRF

# Helper to compute LCOE for a recirc profile over CAPEX
def lcoe_for_recirc(recirc_profile):
    net_MW = P_gross_MW * (1.0 - recirc_profile)
    annual_net_MWh = net_MW * CF * hours_per_year
    return (annualized_capex + annual_fixed_OM) / annual_net_MWh + VOM_plus_fuel_per_MWh

# Compute LCOE curves
curves = {
    "Const 10%": lcoe_for_recirc(np.full_like(capex_per_kW, const_fracs[0])),
    "Const 20%": lcoe_for_recirc(np.full_like(capex_per_kW, const_fracs[1])),
    "Const 30%": lcoe_for_recirc(np.full_like(capex_per_kW, const_fracs[2])),
    "Quad f(B)": lcoe_for_recirc(f_quad),
    "Sat f(B)":  lcoe_for_recirc(f_sat),
}

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(9, 6.5))
for label, y in curves.items():
    style = '-' if 'Const' in label else '--'
    plt.plot(capex_per_kW, y, style, lw=2, label=label)

plt.title("LCOE vs CAPEX with Nonlinear Recirculation Models\nGross=1,000 MW, CF=90%, WACC=8%, Life=30y, FOM=$100/kW-yr, VOM=$5/MWh")
plt.xlabel("Overnight CAPEX (USD per gross kW)")
plt.ylabel("LCOE (USD/MWh)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# plt.savefig("capex_lcoe_nonlin_recirc.png", dpi=160)
plt.show()
# Print a few sample values for comparison
sample_caps = [2500, 5000, 7500]
print("Sample LCOE values (USD/MWh):")
for cap in sample_caps:
    idx = np.argmin(np.abs(capex_per_kW - cap))
    values = {label: float(y[idx]) for label, y in curves.items()}
    print(f"CAPEX ${cap}/kW -> " +
          ", ".join([f"{k}: {v:5.1f}" for k, v in values.items()]))
