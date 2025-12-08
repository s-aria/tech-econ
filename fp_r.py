
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Size (R) vs Fusion thermal power (MW_th): LCOE contours (MW units)
# ----------------------------------------------------

# Grid of plant size and fusion thermal power
R_m = np.linspace(1.0, 10.0, 121)          # major radius [m]
Pfus_MWth = np.linspace(100.0, 1000.0, 101)  # fusion thermal power [MW_th]
R, Pfus = np.meshgrid(R_m, Pfus_MWth)

# Performance proxy: Pfus ∝ k_perf * R^3 * B^4  --> infer required B
# (Synthetic scaling; replace with your design code outputs as needed.)
k_perf = 2e-5    # [MW_th / (m^3 * T^4)]
B_required_T = ((Pfus / (k_perf * R**3))**0.25)  # on-axis field

# Cost proxy: magnet/structure CAPEX sensitivity ~ B^2 R^3 (magnetic energy)
# Map it to overnight CAPEX per gross kW via a scale factor (tunable calibration).
capex_scale = 220.0
capex_per_kW_gross = capex_scale * (B_required_T**2 * R**3) / 1000.0  # $/kW_gross

# Thermal-to-electric conversion efficiency
eta_th = 0.35
P_gross_e_MW = eta_th * Pfus  # gross electric power [MW_e]

# Recirculating power fraction: base + field-dependent term
f0 = 0.20
B_ref = 5.0
k_recirc = 0.10
f_recirc = np.clip(f0 + k_recirc * (B_required_T / B_ref)**2, 0.0, 0.60)

# Net electric output [MW_e]
P_net_e_MW = P_gross_e_MW * (1.0 - f_recirc)

# Economics
CF = 0.85
WACC = 0.08
life_yrs = 30
FOM_per_kW_per_yr = 150
VOM_fuel_per_MWh = 3

# Capital Recovery Factor
CRF = WACC * (1 + WACC) ** life_yrs / ((1 + WACC) ** life_yrs - 1)

# Annualized CAPEX ($/yr), Fixed O&M ($/yr) using gross capacity in kW
gross_kW = P_gross_e_MW * 1000.0
annualized_capex = capex_per_kW_gross * gross_kW * CRF
annual_fixed_OM = FOM_per_kW_per_yr * gross_kW

# Annual net generation (MWh/yr)
annual_net_MWh = P_net_e_MW * CF * 8760.0

# LCOE [$ / MWh]
LCOE = (annualized_capex + annual_fixed_OM) / annual_net_MWh + VOM_fuel_per_MWh

# Mask unrealistic regions (e.g., net power too small)
mask = (P_net_e_MW <= 50) | ~np.isfinite(LCOE)
LCOE_masked = np.ma.masked_where(mask, LCOE)

# Dynamic levels based on actual data
finite_vals = LCOE[~mask]
vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
levels = np.linspace(vmin, vmax, 20)

# Plot
plt.figure(figsize=(9.5, 7.5))
cs = plt.contourf(R, Pfus, LCOE_masked, levels=levels, cmap='viridis', extend='both')
cl = plt.contour(R, Pfus, LCOE_masked, levels=levels, colors='k', linewidths=0.6, alpha=0.6)
plt.clabel(cl, fmt=lambda x: f"{x:.0f}")

cbar = plt.colorbar(cs)
cbar.set_label('LCOE (USD/MWh)')

plt.xlabel('Major radius R (m)')
plt.ylabel('Fusion thermal power (MW_th)')
plt.title('Size vs Fusion Thermal Power (MW_th): LCOE contours\neta_th=35%, CF=85%, recirc ≈ 20% + k·(B/B_ref)^2')

# Example annotations (all in MW_th and MW_e)
# Feel free to change these to your concept points.
example_points = {
    'Pt1 (R~6 m, Pfus~2500 MW_th)': (6.0, 2500.0),
    'Pt2 (R~8 m, Pfus~3500 MW_th)': (8.0, 3500.0),
    'Pt3 (R~10 m, Pfus~5000 MW_th)': (10.0, 5000.0),
}
for label, (r,p) in example_points.items():
    i = np.argmin(np.abs(R_m - r))
    j = np.argmin(np.abs(Pfus_MWth - p))
    lcoe_val = float(LCOE[j, i])
    net_power = float(P_net_e_MW[j, i])
    plt.plot(r, p, 'ro', ms=5)
    plt.text(r+0.15, p+50.0,
             f"{label}\nLCOE~{lcoe_val:.0f} $/MWh\nNet~{net_power:.0f} MW_e",
             fontsize=8, color='white',
             bbox=dict(facecolor='black', alpha=0.35, pad=1, edgecolor='none'))

plt.tight_layout()
plt.savefig('size_vs_power_lcoe_MW.png', dpi=160)

# Diagnostics
print(f"LCOE range: {vmin:.1f}–{vmax:.1f} $/MWh")
print(f"Net electric range: {np.nanmin(P_net_e_MW):.1f}–{np.nanmax(P_net_e_MW):.1f} MW_e")
