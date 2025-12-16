
import re, json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------
# Read PROCESS OUT file
# ------------------------------
F_OUT = 'SPP-002-4_1GW_scan_down.OUT.DAT'
try:
    with open(F_OUT, 'r', encoding='utf-8', errors='ignore') as f:
        out_text = f.read()
except Exception:
    out_text = ''

def find_float(pattern, text, flags=re.IGNORECASE|re.DOTALL):
    m = re.search(pattern, text, flags)
    return float(m.group(1)) if m else None

# Baselines pulled from OUT where available; otherwise fallbacks from PDF summary
P_fus_MW  = find_float(r"p_fusion_total_mw\)\s*([0-9e+\-\.]+)", out_text)
P_gross_MW_reported = find_float(r"p_plant_electric_gross_mw\)\s*([0-9e+\-\.]+)", out_text)
eta_turbine = find_float(r"eta_turbine\)\s*([0-9e+\-\.]+)", out_text)
P_net_MW   = find_float(r"p_plant_electric_net_mw\)\s*([0-9e+\-\.]+)", out_text)
P_recirc_MW= find_float(r"p_plant_electric_recirc_mw\)\s*([0-9e+\-\.]+)", out_text)
B_peak_T   = find_float(r"b_tf_inboard_peak_with_ripple\)\s*([0-9e+\-\.]+)", out_text)
R_major_m  = find_float(r"Major radius .* \(rmajor\)\s*([0-9e+\-\.]+)", out_text)

burn_s  = find_float(r"t_plant_pulse_burn\)\s*([0-9e+\-\.]+)", out_text)
cycle_s = find_float(r"t_plant_pulse_total\)\s*([0-9e+\-\.]+)", out_text)
duty_frac = (burn_s / cycle_s) if (burn_s and cycle_s) else 0.8

# Recirculation components from OUT (may be absent depending on formatter)
cryo_mwe    = find_float(r"crymw\)\s*([0-9e+\-\.]+)", out_text)
pumps_mwe   = find_float(r"p_coolant_pump_elec_total_mw\)\s*([0-9e+\-\.]+)", out_text)
tritium_mwe = find_float(r"p_tritium_plant_electric_mw\)\s*([0-9e+\-\.]+)", out_text)
vacuum_mwe  = find_float(r"vachtmw\.\.\)\s*([0-9e+\-\.]+)", out_text)
base_mwe    = find_float(r"p_plant_electric_base_total_mw\)\s*([0-9e+\-\.]+)", out_text)
tf_mwe      = find_float(r"p_tf_electric_supplies_mw\)\s*([0-9e+\-\.]+)", out_text)
pf_mwe      = find_float(r"p_pf_electric_supplies_mw\)\s*([0-9e+\-\.]+)", out_text)
hcd_mwe     = find_float(r"p_hcd_electric_total_mw\)\s*([0-9e+\-\.]+)", out_text)

recirc_components = {
    'Cryogenics': cryo_mwe or 0.0,
    'Tritium':    tritium_mwe or 0.0,
    'Vacuum':     vacuum_mwe or 0.0,
    'Coolant pumps': pumps_mwe or 0.0,
    'Base load':  base_mwe or 0.0,
    'TF coil supplies': tf_mwe or 0.0,
    'PF coil supplies': pf_mwe or 0.0,
    'H&CD electric': hcd_mwe or 0.0,
}
sum_comp = sum(recirc_components.values())

# ---- Robust fallback to PDF summary values if the OUT text misses numbers ----
if sum_comp == 0:
    # PDF summary (your run): gross ~593.98 MWe; net ~111.11 MWe; recirc ~482.87 MWe; eta_turbine ~0.42
    # Components: cryo~64.965, tritium~57.68, vacuum~0.5, pumps~85.405, base~82.42, TF~3.799, PF~0.936, H&CD~187.16
    recirc_components = {
        'Cryogenics': 64.965,
        'Tritium': 57.68,
        'Vacuum': 0.5,
        'Coolant pumps': 85.405,
        'Base load': 82.42,
        'TF coil supplies': 3.799,
        'PF coil supplies': 0.936,
        'H&CD electric': 187.16,
    }
    sum_comp = sum(recirc_components.values())
    P_recirc_MW = P_recirc_MW or 482.87
    P_gross_MW  = P_gross_MW_reported or 593.98
    eta_turbine = eta_turbine or 0.42
    P_fus_MW    = P_fus_MW or 993.1
else:
    P_gross_MW = P_gross_MW_reported or ((eta_turbine or 0.42)*(P_fus_MW or 1000.0))

norm_scale = (P_recirc_MW / sum_comp) if (P_recirc_MW and sum_comp>0) else 1.0
recirc_norm = {k: v*norm_scale for k,v in recirc_components.items()}

# ------------------------------
# Economics & knobs (edit as needed)
# ------------------------------
@dataclass
class Econ:
    WACC: float = 0.08
    life_years: int = 30
    FOM_per_kW_per_yr: float = 150.0
    VOM_fuel_per_MWh: float = 3.0
    availability: float = 0.75   # plant time-availability from OUT; CF_effective = availability * duty

E = Econ()
CF_effective = E.availability * duty_frac
CRF = E.WACC * (1+E.WACC)**E.life_years / ((1+E.WACC)**E.life_years - 1)

# CAPEX dial and physics proxy mapping
capex_per_kW = np.linspace(2000, 8000, 250)  # $/kW_gross
capex_scale  = 220.0                         # $/kW_gross per unit (B^2 R^3)/1000

# ------------------------------
# Nonlinear recirculation (physics-informed scalers)
# ------------------------------
R_ref = R_major_m or 4.2
B_from_capex = lambda cap: np.sqrt(np.maximum(cap,1e-6) / (capex_scale * (R_ref**3)/1000.0))
B_series = B_from_capex(capex_per_kW)

T_cold_K = 20.0
B0 = max(B_peak_T or 13.0, 1e-3)

def cryo_scale(B):  # HTS cryo scaling
    return np.clip((B/B0)**2 * (20.0/max(T_cold_K,1e-6)), 0.2, 3.0)

dp_he = find_float(r"dp_he\)\s*([0-9e+\-\.]+)", out_text) or 4.3e5
def pump_scale(dp):
    return np.clip((dp/max(1e3,dp_he))**0.5, 0.5, 2.0)

P_inj_MW = find_float(r"p_hcd_injected_total_mw\)\s*([0-9e+\-\.]+)", out_text) or 115.0
eta_inj  = find_float(r"eta_hcd_primary_injector_wall_plug\)\s*([0-9e+\-\.]+)", out_text) or 0.6
def hcd_elec(P_inj, eta):
    return (P_inj / max(eta, 0.1))

cryo_base  = recirc_norm.get('Cryogenics', 0.0)
pump_base  = recirc_norm.get('Coolant pumps', 0.0)
others_sum = sum(recirc_norm[k] for k in recirc_norm.keys() if k not in ['Cryogenics','Coolant pumps','H&CD electric'])

recirc_vs_capex = cryo_base*cryo_scale(B_series) + pump_base*pump_scale(dp_he) + hcd_elec(P_inj_MW, eta_inj) + others_sum

# LCOE curves
hours = 8760
gross_kW = P_gross_MW * 1000.0
annual_fixed_OM  = E.FOM_per_kW_per_yr * gross_kW
annualized_capex = capex_per_kW * gross_kW * CRF
annual_net_MWh_nonlin = (P_gross_MW - recirc_vs_capex) * CF_effective * hours
LCOE_nonlin = (annualized_capex + annual_fixed_OM) / np.maximum(annual_net_MWh_nonlin,1e-6) + E.VOM_fuel_per_MWh

const_fracs = [0.15, 0.25, 0.40]
LCOE_const = {}
for f in const_fracs:
    P_net_const = P_gross_MW * (1.0 - f)
    LCOE_const[f] = (annualized_capex + annual_fixed_OM) / (P_net_const * CF_effective * hours) + E.VOM_fuel_per_MWh

plt.figure(figsize=(9.6,6.2))
for f, y in LCOE_const.items():
    plt.plot(capex_per_kW, y, '-', lw=2, label=f"Const recirc {int(f*100)}%")
plt.plot(capex_per_kW, LCOE_nonlin, '--', lw=2.5, label='Nonlinear recirc (from file scalers)')
plt.title("LCOE vs CAPEX with Nonlinear Recirculation\nInputs calibrated to PROCESS outputs")
plt.xlabel("Overnight CAPEX (USD per gross kW)")
plt.ylabel("LCOE (USD/MWh)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("nonlinear_recirc_vs_capex_lcoe_from_file.png", dpi=160)

# ------------------------------
# R–Pfus LCOE surface + gradient
# ------------------------------
R0   = R_major_m or 4.2
Bcal = B_peak_T or 13.0
Pfus0= P_fus_MW or 1000.0
k_perf = Pfus0 / (R0**3 * Bcal**4)  # calibrate

R_vals    = np.linspace(max(3.5, R0-1.0), R0+1.5, 121)
Pfus_vals = np.linspace(700.0, 1300.0, 121)
R, Pfus_grid = np.meshgrid(R_vals, Pfus_vals)

B_req = (Pfus_grid / (k_perf * R**3))**0.25
capex_per_kW_grid = capex_scale * (B_req**2 * R**3) / 1000.0
eta_th = eta_turbine or 0.42
P_gross_grid_MW = eta_th * Pfus_grid

cryo_grid   = cryo_base * cryo_scale(B_req)
pumps_grid  = pump_base * pump_scale(dp_he)
hcd_grid    = hcd_elec(P_inj_MW, eta_inj)
others_grid = others_sum
P_recirc_grid = cryo_grid + pumps_grid + hcd_grid + others_grid

P_net_grid_MW        = np.maximum(P_gross_grid_MW - P_recirc_grid, 1e-6)
annual_fixed_grid    = E.FOM_per_kW_per_yr * (P_gross_grid_MW*1000.0)
annualized_grid      = capex_per_kW_grid * (P_gross_grid_MW*1000.0) * CRF
annual_net_MWh_grid  = P_net_grid_MW * CF_effective * hours
LCOE_grid            = (annualized_grid + annual_fixed_grid) / np.maximum(annual_net_MWh_grid,1e-6) + E.VOM_fuel_per_MWh

# Gradient and safer contour levels
dR = R_vals[1]-R_vals[0]
dP = Pfus_vals[1]-Pfus_vals[0]
gy, gx = np.gradient(LCOE_grid, dP, dR)  # gy=dL/dPfus, gx=dL/dR

vmin = np.nanpercentile(LCOE_grid, 5)
vmax = np.nanpercentile(LCOE_grid,95)
levels = np.linspace(vmin, vmax, 18)

plt.figure(figsize=(9.8,7.4))
cs = plt.contourf(R, Pfus_grid, LCOE_grid, levels=levels, cmap='viridis')
cl = plt.contour(R, Pfus_grid, LCOE_grid, levels=levels, colors='k', linewidths=0.5)
plt.clabel(cl, fmt=lambda x: f"{x:.0f}")
plt.colorbar(cs, label='LCOE (USD/MWh)')
step = 8
plt.quiver(R[::step,::step], Pfus_grid[::step,::step], -gx[::step,::step], -gy[::step,::step],
           color='white', angles='xy', scale_units='xy', scale=1.5, width=0.003, alpha=0.8)
plt.xlabel('Major radius R (m)')
plt.ylabel('Fusion thermal power (MW_th)')
plt.title('Fusion power vs major radius with LCOE gradient\n(calibrated to PROCESS outputs)')
plt.tight_layout()
plt.savefig('R_vs_Pfus_LCOE_gradient.png', dpi=160)

# Diagnostics
summary = {'baseline': {
    'Pfus_MW':  P_fus_MW, 'P_gross_MW': P_gross_MW, 'P_net_MW': P_net_MW,
    'P_recirc_MW': P_recirc_MW, 'eta_turbine': eta_turbine,
    'B_peak_T': B_peak_T, 'R_major_m': R_major_m, 'duty_frac': duty_frac,
}, 'recirc_components_MWe_normalized': recirc_norm}
print(json.dumps(summary, indent=2))

for cap in [2500,5000,7500]:
    i = np.argmin(np.abs(capex_per_kW - cap))
    print(f"CAPEX ${cap}/kW -> Nonlinear LCOE: {LCOE_nonlin[i]:.1f}, "
          f"Const15%: {LCOE_const[0.15][i]:.1f}, "
          f"Const25%: {LCOE_const[0.25][i]:.1f}, "
          f"Const40%: {LCOE_const[0.40][i]:.1f}")
