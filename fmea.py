import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =============================================================================
# C) FMEA HEAT MAP (SEVERITY VS LIKELIHOOD) WITH COMPONENT OVERLAYS
# =============================================================================
"""
FMEA rating scales:
- Severity (S): 1-10 (10 = most severe)
- Likelihood/Occurrence (O): 1-10 (10 = most likely)
- Detection (D): 1-10 (10 = hardest to detect) — optional; default D=1 if omitted
- RPN = S * O * D
Replace 'fmea_items' with your system components and ratings.
"""

fmea_items = [
    {"Component": "HTS Magnet Coil",      "S": 9, "O": 4, "D": 6},
    {"Component": "Cryogenic Plant",      "S": 8, "O": 5, "D": 5},
    {"Component": "Vacuum Vessel Seals",  "S": 7, "O": 6, "D": 6},
    {"Component": "Power Electronics",    "S": 6, "O": 5, "D": 4},
    {"Component": "RF Heating System",    "S": 7, "O": 3, "D": 5},
    {"Component": "Control System (PLC)", "S": 5, "O": 3, "D": 3},
    {"Component": "Cooling Loops",        "S": 6, "O": 7, "D": 5},
    {"Component": "Quench Detection",     "S": 9, "O": 3, "D": 8},
]

df_fmea = pd.DataFrame(fmea_items)
if "D" not in df_fmea.columns:
    df_fmea["D"] = 1
df_fmea["RPN"] = df_fmea["S"] * df_fmea["O"] * df_fmea["D"]

# Create severity-likelihood grid and aggregate max RPN per cell
sev_range = np.arange(1, 11)
occ_range = np.arange(1, 11)
pivot = df_fmea.pivot_table(index="S", columns="O", values="RPN", aggfunc="max")
pivot = pivot.reindex(index=sev_range, columns=occ_range, fill_value=0)

fig_fmea = go.Figure(data=go.Heatmap(
    z=pivot.values, x=occ_range, y=sev_range,
    colorscale="YlOrRd", colorbar=dict(title="RPN"),
    hovertemplate="Likelihood (O): %{x}<br>Severity (S): %{y}<br>Max RPN: %{z}<extra></extra>"
))
fig_fmea.update_layout(
    title="FMEA Heat Map (Severity vs Likelihood) — Color = Max RPN per Cell",
    xaxis_title="Likelihood / Occurrence (O)",
    yaxis_title="Severity (S)",
    template="plotly_white"
)

# Overlay components
fig_fmea.add_trace(go.Scatter(
    x=df_fmea["O"], y=df_fmea["S"],
    mode="markers+text",
    text=df_fmea["Component"],
    textposition="top center",
    marker=dict(
        size=np.clip(df_fmea["RPN"] / 2.0, 8, 30),
        color=df_fmea["RPN"],
        colorscale="YlOrRd",
        showscale=False,
        line=dict(color="black", width=0.5)
    ),
    name="Components",
    hovertemplate=(
        "Component: %{text}<br>"
        "Severity (S): %{y}<br>"
        "Likelihood (O): %{x}<br>"
        "RPN: %{marker.color:.0f}<extra></extra>"
    )
))
# fig_fmea.show()

# Optional console output: ranked FMEA table
print("\nFMEA Ranked by RPN (Desc):")
print(df_fmea.sort_values("RPN", ascending=False)[["Component", "S", "O", "D", "RPN"]].to_string(index=False))
