
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
fn = 'plant_data.csv'
df = pd.read_csv(fn, skipinitialspace=True)
df.columns = [c.strip().replace('\u200b','') for c in df.columns]
df = df.rename(columns={
    'major_radius_m': 'R_m',
    'net_power_MWe': 'Pnet_MWe',
    'capex_GBP': 'Capex_GBP',
    'fixed_OandM_GBP_per_year': 'FixedOM_GBP_per_year',
    'discount_rate_pct': 'i_pct',
    'lifetime_years': 'n_years',
    'efficiency_after_sec_cycle_%': 'eta_sec_pct',
})

# Convert numerics
for col in ['R_m','Pnet_MWe','Capex_GBP','FixedOM_GBP_per_year','i_pct','n_years','eta_sec_pct']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Capacity factor per plant (user provided): first plant 31%, second 70%
CF_points = np.array([0.31, 0.70])
assert len(CF_points) == len(df), "CF list must align with CSV rows"

# Economic parameters
i = df['i_pct'].iloc[0]/100.0
n = int(df['n_years'].iloc[0])
CRF = i*(1+i)**n/((1+i)**n - 1)

variable_OandM_GBP_per_MWh = 0.0
fuel_GBP_per_MWh = 0.0
fx_GBP_to_USD = 1.0      # set to your GBP->USD rate (e.g., 1.27)
LCOE_target_USD_per_MWh = 150.0
LCOE_target_GBP_per_MWh = LCOE_target_USD_per_MWh / fx_GBP_to_USD

# Per-point LCOE using row-specific CF
annual_energy_MWh_points = df['Pnet_MWe'] * 8760.0 * CF_points
LCOE_points_GBP_per_MWh = (CRF*df['Capex_GBP'] + df['FixedOM_GBP_per_year'])/annual_energy_MWh_points \
                           + variable_OandM_GBP_per_MWh + fuel_GBP_per_MWh

# Fit Pnet(R) = k * R^gamma (log-log)
logR = np.log(df['R_m'].values)
logP = np.log(df['Pnet_MWe'].values)
A = np.vstack([np.ones_like(logR), logR]).T
k_log, gamma = np.linalg.lstsq(A, logP, rcond=None)[0]
k = np.exp(k_log)

# Fit Capex(R) = a * R^alpha (log-log)
logC = np.log(df['Capex_GBP'].values)
A2 = np.vstack([np.ones_like(logR), logR]).T
alog, alpha = np.linalg.lstsq(A2, logC, rcond=None)[0]
a = np.exp(alog)

# Fit efficiency after secondary cycle as a simple linear function of R (two points)
R_vals = df['R_m'].values
eta_vals = df['eta_sec_pct'].values/100.0
eta_coeffs = np.polyfit(R_vals, eta_vals, 1)  # eta(R) ~ m*R + b

def eta_of_R(R):
    return np.clip(eta_coeffs[0]*R + eta_coeffs[1], 0.01, 0.95)

def Pnet_of_R(R):
    return k * (R**gamma)

def Capex_of_R(R):
    return a * (R**alpha)

# Fixed O&M: assume constant fraction of CapEx per year inferred from provided points
fo_frac = (df['FixedOM_GBP_per_year']/df['Capex_GBP']).mean()

def FixedOM_of_R(R):
    return fo_frac * Capex_of_R(R)

# LCOE(R) for a given CF
def LCOE_of_R_GBP(R, CF):
    Pnet = Pnet_of_R(R)
    annual_MWh = Pnet * 8760.0 * CF
    return (CRF*Capex_of_R(R) + FixedOM_of_R(R))/annual_MWh + variable_OandM_GBP_per_MWh + fuel_GBP_per_MWh

# Fusion thermal power LOWER BOUND: P_th >= P_net / eta (net already subtracts parasitics)
# True P_th = (P_net + parasitics)/eta; without parasitics data we show lower bound.
def Pth_lower_of_R(R):
    return Pnet_of_R(R) / eta_of_R(R)

# Solve for R where LCOE == target for two CF scenarios (FOAK vs NOAK bounds)
CF_scenarios = {
    'FOAK_CF_0.31': 0.31,
    'NOAK_CF_0.70': 0.70,
}
solutions = {}
for label, CF_s in CF_scenarios.items():
    R_min, R_max = 0.5, 20.0
    f_min = LCOE_of_R_GBP(R_min, CF_s) - LCOE_target_GBP_per_MWh
    f_max = LCOE_of_R_GBP(R_max, CF_s) - LCOE_target_GBP_per_MWh
    sol = None
    if f_min == 0:
        sol = R_min
    elif f_max == 0:
        sol = R_max
    elif f_min * f_max <= 0:
        lo, hi = R_min, R_max
        for _ in range(120):
            mid = 0.5*(lo+hi)
            f_mid = LCOE_of_R_GBP(mid, CF_s) - LCOE_target_GBP_per_MWh
            if np.sign(f_mid) == np.sign(f_min):
                lo = mid; f_min = f_mid
            else:
                hi = mid
        sol = 0.5*(lo+hi)
    solutions[label] = {
        'R_m': sol,
        'P_net_MWe': (Pnet_of_R(sol) if sol is not None else None),
        'P_th_lower_MWth': (Pth_lower_of_R(sol) if sol is not None else None),
        'LCOE_GBP_per_MWh': (LCOE_of_R_GBP(sol, CF_s) if sol is not None else None),
    }

# --- Plotting ---
R_plot = np.linspace(1.5, 6.0, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: LCOE vs R for CF scenarios
for label, CF_s in CF_scenarios.items():
    LCOE_curve = [LCOE_of_R_GBP(R, CF_s) for R in R_plot]
    axes[0].plot(R_plot, LCOE_curve, label=f'LCOE (CF={CF_s:.2f})')
    sol = solutions[label]['R_m']
    if sol is not None:
        axes[0].axhline(LCOE_target_GBP_per_MWh, color='gray', ls='--', lw=0.8)
        axes[0].plot(sol, LCOE_of_R_GBP(sol, CF_s), 'o', ms=6,
                     label=f'Target @ R={sol:.2f} m')

# Show the two original points (computed with their specific CFs)
axes[0].scatter(df['R_m'], LCOE_points_GBP_per_MWh, c='k', marker='x', label='CSV points')
axes[0].set_title('LCOE vs Major Radius')
axes[0].set_xlabel('Major radius R (m)')
axes[0].set_ylabel('LCOE (GBP/MWh)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Fusion thermal power lower bound vs R
Pth_curve = [Pth_lower_of_R(R) for R in R_plot]
axes[1].plot(R_plot, Pth_curve, color='tab:red', label='P_th lower bound (net/eta)')
for label in CF_scenarios.keys():
    sol = solutions[label]['R_m']
    if sol is not None:
        axes[1].plot(sol, Pth_lower_of_R(sol), 'o', ms=6, label=f'@ LCOE target ({label})')

axes[1].set_title('Fusion thermal power (lower bound) vs Major Radius')
axes[1].set_xlabel('Major radius R (m)')
axes[1].set_ylabel('Fusion thermal power MW_th (lower bound)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('lcoe_power_vs_radius.png', dpi=180)
plt.plot()
plt.show()

# Summary for logging
summary = {
    'CRF': CRF,
    'Pnet_fit_k': k,
    'Pnet_fit_gamma': gamma,
    'Capex_fit_a_GBP': a,
    'Capex_fit_alpha': alpha,
    'FixedOM_fraction_of_capex_per_year': fo_frac,
    'Per_point_LCOE_GBP_per_MWh': LCOE_points_GBP_per_MWh.tolist(),
    'Solutions': solutions,
    'LCOE_target_GBP_per_MWh': LCOE_target_GBP_per_MWh,
}
print(summary)

#####################################################################


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load CSV
# df = pd.read_csv('plant_data.csv', skipinitialspace=True)
# df = df.rename(columns={
#     'major_radius_m': 'R_m',
#     'net_power_MWe': 'Pnet_MWe',
#     'fusion_power_MW': 'Pfusion_MWth',
#     'power_total_MW': 'Ptotal_MW',
#     'capex_GBP': 'Capex_GBP',
#     'fixed_OandM_GBP_per_year': 'FixedOM_GBP_per_year',
#     'discount_rate_pct': 'i_pct',
#     'lifetime_years': 'n_years',
#     'efficiency_after_sec_cycle_%': 'eta_sec_pct',
#     'LCOE': 'LCOE_internal',
# })
# for c in df.columns:
#     df[c] = pd.to_numeric(df[c], errors='coerce')

# # Economics
# i = df['i_pct'].iloc[0]/100.0
# n = int(df['n_years'].iloc[0])
# CRF = i*(1+i)**n/((1+i)**n - 1)
# CF = 0.70  # single panel; change to 0.31 for FOAK

# # Fits from your two points
# R = df['R_m'].values
# eta = df['eta_sec_pct'].values/100.0
# eta_m, eta_b = np.polyfit(R, eta, 1)
# def eta_of_R(R): return np.clip(eta_m*R + eta_b, 0.01, 0.95)

# parasitics_pts = eta*df['Pfusion_MWth'].values - df['Pnet_MWe'].values
# parasitics_pts = np.maximum(parasitics_pts, 1e-6)
# logR = np.log(R); logParas = np.log(parasitics_pts)
# A = np.vstack([np.ones_like(logR), logR]).T
# p0_log, p1 = np.linalg.lstsq(A, logParas, rcond=None)[0]
# p0 = np.exp(p0_log)
# def P_paras_of_R(R): return p0 * (R**p1)

# logC = np.log(df['Capex_GBP'].values)
# A2 = np.vstack([np.ones_like(logR), logR]).T
# alog, alpha = np.linalg.lstsq(A2, logC, rcond=None)[0]
# a = np.exp(alog)
# def Capex_of_R(R): return a * (R**alpha)

# fo_frac = (df['FixedOM_GBP_per_year']/df['Capex_GBP']).mean()
# def FixedOM_of_R(R): return fo_frac * Capex_of_R(R)

# # Grid
# R_grid = np.linspace(max(0.5, df['R_m'].min()*0.5), df['R_m'].max()*1.5, 220)
# Pf_grid = np.linspace(max(300.0, df['Pfusion_MWth'].min()*0.5), df['Pfusion_MWth'].max()*1.5, 220)
# R_mesh, Pf_mesh = np.meshgrid(R_grid, Pf_grid)

# eta_mesh = eta_of_R(R_mesh)
# paras_mesh = P_paras_of_R(R_mesh)
# Pnet_mesh = eta_mesh*Pf_mesh - paras_mesh

# # Mask invalid (net <= 1 MWe) to avoid huge LCOE
# valid = Pnet_mesh > 1.0
# Z = np.full_like(Pnet_mesh, np.nan, dtype=float)
# Z[valid] = (CRF*Capex_of_R(R_mesh[valid]) + FixedOM_of_R(R_mesh[valid])) / (Pnet_mesh[valid]*8760.0*CF)

# # Clip to reasonable range for color scale
# Z_clip = np.clip(Z, 0, 2000)  # GBP/MWh
# LCOE_target = 150.0  # if you want $150/MWh, set fx and convert accordingly

# # Plot single heatmap
# plt.figure(figsize=(7,5))
# pcm = plt.pcolormesh(R_mesh, Pf_mesh, Z_clip, shading='auto', cmap='viridis')
# try:
#     cs = plt.contour(R_mesh, Pf_mesh, Z, levels=[LCOE_target], colors='white', linewidths=1.3)
#     plt.clabel(cs, fmt={LCOE_target: f'{LCOE_target:.0f} target'}, inline=True, fontsize=9)
# except Exception as e:
#     print('Contour warning:', e)
# plt.scatter(df['R_m'], df['Pfusion_MWth'], c='red', s=40, marker='x', label='CSV points')
# plt.title('LCOE heatmap (CF=0.70)')
# plt.xlabel('Major radius R (m)')
# plt.ylabel('Fusion power (MW_th)')
# plt.legend(loc='best')
# plt.colorbar(pcm, label='LCOE (GBP/MWh)')
# plt.tight_layout()
# # plt.savefig('lcoe_heatmap_single.png', dpi=180)
# plt.plot()
# plt.show()