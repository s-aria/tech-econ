
"""
Integrated HTS Cost Scaling + Detailed LCOE + FMEA (Plotly Interactive)
-----------------------------------------------------------------------
Raw Python code that includes:
  A) HTS cost scaling (baseline, Monte Carlo uncertainty, alpha & C0 sensitivities)
  B) Detailed LCOE (present-value method) + sensitivity vs availability (MW-normalized)
  C) CapEx vs LCOE scatter for 14 plants (replace placeholders with real values)
  D) FMEA heat map (Severity vs Likelihood) with component overlays

Notes:
- All figures use Plotly and display with fig.show() (no saving).
- Edit parameter sections to match your scenario.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# A) HTS COST SCALING
# =============================================================================
# =============================================================================
# Supplier-calibrated HTS cost baseline using TOTAL COST for 200 km tape
# =============================================================================
def supplier_cost_per_kAm_from_200km(total_cost_200km: float, ic_a_77k_0t: float) -> float:
    """
    Convert supplier offer (total cost for 200 km) to $/kA·m.
    Formula:
      Price_per_meter = total_cost_200km / 200,000
      Cost_per_kA_m   = Price_per_meter / (Ic_A / 1000)
    """
    price_per_meter = total_cost_200km / 200_000.0  # 200 km = 200,000 m
    return price_per_meter / (ic_a_77k_0t / 1000.0)

# ---  supplier data (7 suppliers) ---
suppliers_200km_data = [
    {"Supplier": "Eastern Superconducting Technology", "Total_cost_200km": 6090000, "Ic_A_77K_0T": 550},
    {"Supplier": "Farady Factory", "Total_cost_200km": 5532000, "Ic_A_77K_0T": 570},
    {"Supplier": "Fujikura Europe", "Total_cost_200km": 7000000, "Ic_A_77K_0T": 450},
    # {"Supplier": "Furukawa Electric Europe", "Total_cost_200km": 6999990, "Ic_A_77K_0T": 150},
    {"Supplier": "MetOx International", "Total_cost_200km": 8851516, "Ic_A_77K_0T": 450},
    {"Supplier": "Shanghai Superconductor Technology", "Total_cost_200km": 3500000, "Ic_A_77K_0T": 675},
    {"Supplier": "Supermag Technology", "Total_cost_200km": 4350000, "Ic_A_77K_0T": 600},
]

# Compute $/kA·m for each supplier
rows = []
for s in suppliers_200km_data:
    cost_kAm = supplier_cost_per_kAm_from_200km(s["Total_cost_200km"], s["Ic_A_77K_0T"])
    rows.append({"Supplier": s["Supplier"], "Cost_per_kA_m": cost_kAm})
df_sup = pd.DataFrame(rows)
print("Supplier-derived $/kA·m:")
print(df_sup)

# Baseline C0 = median of supplier values
C0_baseline = df_sup["Cost_per_kA_m"].median()
print(f"\nBaseline C0 (median): ${C0_baseline:.2f}/kA·m")

# =============================================================================
# HTS cost scaling model
# =============================================================================
def cost_scaling(C0: float, V0: float, alpha: float, volumes: np.ndarray) -> np.ndarray:
    return C0 * (volumes / V0) ** (-alpha)

# Parameters
V0 = 1000.0
alpha = 0.25
volumes = np.logspace(3, 6, 200)
alpha_range = (0.20, 0.30)

# Baseline projection
costs = cost_scaling(C0_baseline, V0, alpha, volumes)
fig_baseline = go.Figure()
fig_baseline.add_trace(go.Scatter(x=volumes, y=costs, mode='lines', name=f'Baseline C0={C0_baseline:.0f}'))
fig_baseline.update_layout(title='HTS Cost Scaling (Baseline)', xaxis_type='log', yaxis_type='log',
                            xaxis_title='Production Volume (kA·m)', yaxis_title='Cost ($/kA·m)')
fig_baseline.show()



# Monte Carlo uncertainty
np.random.seed(42)
N = 10000
alpha_samples = np.random.uniform(alpha_range[0], alpha_range[1], N)
C0_samples = np.random.choice(df_sup["Cost_per_kA_m"].values, size=N, replace=True)
noise = np.random.lognormal(mean=0.0, sigma=0.10, size=N)
C0_samples *= noise

cost_matrix = np.zeros((N, volumes.size))
for i in range(N):
    cost_matrix[i, :] = cost_scaling(C0_samples[i], V0, alpha_samples[i], volumes)

median_cost = np.percentile(cost_matrix, 50, axis=0)
p10_cost = np.percentile(cost_matrix, 10, axis=0)
p90_cost = np.percentile(cost_matrix, 90, axis=0)

fig_uncertainty = go.Figure()
fig_uncertainty.add_trace(go.Scatter(x=volumes, y=median_cost, mode='lines', name='Median'))
fig_uncertainty.add_trace(go.Scatter(x=volumes, y=p10_cost, mode='lines', name='10th pct', line=dict(dash='dot')))
fig_uncertainty.add_trace(go.Scatter(x=volumes, y=p90_cost, mode='lines', name='90th pct', line=dict(dash='dot')))
fig_uncertainty.update_layout(title='HTS Cost Scaling with Uncertainty', xaxis_type='log', yaxis_type='log',
                              xaxis_title='Production Volume (kA·m)', yaxis_title='Cost ($/kA·m)')
fig_uncertainty.show()

# Sensitivity: alpha
alphas = [0.20, 0.225, 0.25, 0.275, 0.30]
fig_alpha = go.Figure()
for a in alphas:
    fig_alpha.add_trace(go.Scatter(x=volumes, y=cost_scaling(C0_baseline, V0, a, volumes), mode='lines', name=f'alpha={a}'))
fig_alpha.update_layout(title='Sensitivity: alpha', xaxis_type='log', yaxis_type='log',
                        xaxis_title='Production Volume (kA·m)', yaxis_title='Cost ($/kA·m)')
fig_alpha.show()

# Sensitivity: C0 (supplier values)
fig_C0 = go.Figure()
for c in df_sup["Cost_per_kA_m"].values:
    fig_C0.add_trace(go.Scatter(x=volumes, y=cost_scaling(c, V0, alpha, volumes), mode='lines', name=f'C0=${c:.0f}'))
fig_C0.update_layout(title='Sensitivity: C0 (Supplier-derived)', xaxis_type='log', yaxis_type='log',
                     xaxis_title='Production Volume (kA·m)', yaxis_title='Cost ($/kA·m)')
fig_C0.show()


# =============================================================================
# B) DETAILED LCOE (PV METHOD) + SENSITIVITY VS AVAILABILITY (MW-NORMALIZED)
# =============================================================================
def lcoe_detailed(capex: float,
                  replacements: list,
                  annual_opex: float,
                  plant_size_MW: float,
                  availability: float,
                  discount_rate: float,
                  lifetime_years: int) -> float:
    """
    LCOE ($/MWh) via present-value method (normalized per MW of plant capacity).

    Parameters:
      capex           : Initial capital cost at year 0 ($)
      replacements    : List of tuples [(year:int, cost:$), ...]
      annual_opex     : Annual operating cost ($/year)
      plant_size_MW   : Nameplate capacity (MW)
      availability    : Capacity factor (0–1)
      discount_rate   : Annual discount rate (e.g., 0.08)
      lifetime_years  : Project lifetime (years)
    """
    # Annual energy output per year (MWh)
    annual_MWh = plant_size_MW * availability * 8760.0

    # Present value (PV) of costs
    pv_costs = capex  # year 0
    for year in range(1, lifetime_years + 1):
        pv_costs += annual_opex / ((1.0 + discount_rate) ** year)
        for rep_year, rep_cost in replacements:
            if year == rep_year:
                pv_costs += rep_cost / ((1.0 + discount_rate) ** year)

    # PV of energy (constant each year)
    pv_energy = 0.0
    for year in range(1, lifetime_years + 1):
        pv_energy += annual_MWh / ((1.0 + discount_rate) ** year)

    return pv_costs / pv_energy

# --- LCOE parameters  ---
capex_lcoe        = 19329800230               # $ initial
replacements_lcoe = [(10, 1_000_000), (20, 1_000_000)]  # component replacements
annual_opex_lcoe  = 230782574                 # $/year
plant_size_MW     = 609.2454215               # MW
discount_rate     = 0.07                      # 8%
lifetime_years    = 40                        # years
availability_values = np.linspace(0.30, 0.95, 50)

# --- LCOE sensitivity vs availability ---
lcoe_values = [
    lcoe_detailed(capex_lcoe, replacements_lcoe, annual_opex_lcoe,
                  plant_size_MW, cf, discount_rate, lifetime_years)
    for cf in availability_values
]

fig_lcoe = go.Figure()
fig_lcoe.add_trace(go.Scatter(
    x=availability_values, y=lcoe_values, mode='lines+markers',
    name='LCOE vs Availability',
    hovertemplate='Availability (CF): %{x:.2f}<br>LCOE: $%{y:.2f}/MWh<extra></extra>'
))
fig_lcoe.update_layout(
    title='LCOE Sensitivity: Availability vs LCOE (PV Method, MW-Normalized)',
    xaxis_title='Availability / Capacity Factor (0–1)',
    yaxis_title='LCOE ($/MWh)',
    template='plotly_white'
)
fig_lcoe.show()

# --- CapEx vs LCOE for 9 plants SPP + Alt C ---
plants_data = [
    {"Plant": "SPP-P",  "CapEx": 22000000000, "LCOE": 6228.31584},
    {"Plant": "SPP-P-LiPb",  "CapEx": 21219272134, "LCOE": 3320.072346},
    {"Plant": "SPP-C",  "CapEx": 19329800230, "LCOE": 1619.123913},
    {"Plant": "SPP-C-LiPb",  "CapEx": 18770407608, "LCOE": 949.7461228},
    {"Plant": "Alt-C-C1", "CapEx": 14880363793, "LCOE": 766.0636001},
    {"Plant": "Alt-C-C2", "CapEx": 14582984172, "LCOE": 501.7081473},
    {"Plant": "Alt-C-C3", "CapEx": 14545420595, "LCOE": 736.1265386},
    {"Plant": "Alt-C-C4", "CapEx": 12609525873, "LCOE": 431.074169},
    {"Plant": "Alt-C-C5", "CapEx": 12645488025, "LCOE": 461.3284788},
]

# Compute LCOE for plants if not provided but parameters exist
for p in plants_data:
    if "LCOE" not in p:
        required_keys = {"annual_opex", "plant_size_MW", "availability", "discount_rate", "lifetime_years"}
        if required_keys.issubset(p.keys()):
            repl = p.get("replacements", [])
            p["LCOE"] = lcoe_detailed(
                capex=p["CapEx"],
                replacements=repl,
                annual_opex=p["annual_opex"],
                plant_size_MW=p["plant_size_MW"],
                availability=p["availability"],
                discount_rate=p["discount_rate"],
                lifetime_years=p["lifetime_years"]
            )
        else:
            p["LCOE"] = None

df_plants = pd.DataFrame(plants_data)
df_plot = df_plants.dropna(subset=["CapEx", "LCOE"])

# fig_capex_lcoe = go.Figure()
# fig_capex_lcoe.add_trace(go.Scatter(
#     x=df_plot["CapEx"], y=df_plot["LCOE"], mode='markers+text',
#     text=df_plot["Plant"], textposition='top center',
#     marker=dict(size=12, color='royalblue', opacity=0.85, line=dict(color='black', width=0.5)),
#     name='Plants',
#     hovertemplate='Plant: %{text}<br>CapEx: $%{x:,.0f}<br>LCOE: %{y:.2f} $/MWh<extra></extra>'
# ))
# fig_capex_lcoe.update_layout(
#     title='LCOE vs CapEx (SPP vs Alt-C)',
#     xaxis_title='CapEx ($)',
#     yaxis_title='LCOE ($/MWh)',
#     template='plotly_white'
# )
# fig_capex_lcoe.show()


fig = px.scatter(
    df_plants,
    x="CapEx",
    y="LCOE",
    text="Plant",
    trendline="ols",
    title="LCOE vs CapEx with Regression (Plotly Express)",
    labels={"CapEx": "CapEx ($)", "LCOE": "LCOE ($/MWh)"}
)

fig.update_traces(textposition="top center")

# Extract regression details
results = px.get_trendline_results(fig)
model = results.iloc[0]["px_fit_results"]
slope = model.params[1]
intercept = model.params[0]
r_squared = model.rsquared

fig.show()