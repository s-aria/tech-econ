
import plotly.express as px
import pandas as pd

# Data
capex = [22000000000, 19329800230, 18770407608, 14880363793,
         14582984172, 14545420595, 12609525873, 12645488025]
lcoe  = [6228.31584, 1619.123913, 949.7461228, 766.0636001,
         501.7081473, 736.1265386, 431.074169, 461.3284788]

hover_labels = [
    "SPP", "Project 2", "Project 3", "Project 4",
    "Project 5", "Project 6", "Project 7", "Project 8"
]

# Create DataFrame for Plotly
df = pd.DataFrame({"LCOE": lcoe, "CapEx": capex, "Label": hover_labels})

# Scatter plot with linear regression trendline
fig = px.scatter(
    df,
    x="LCOE",
    y="CapEx",
    hover_name="Label",
    labels={"LCOE": "LCOE", "CapEx": "CapEx"},
    title="CapEx vs LCOE with Linear Regression",
    trendline="ols"  # Ordinary Least Squares regression
)

# Adjust axes
fig.update_layout(
    xaxis=dict(range=[0, max(lcoe)*1.1], showgrid=True),
    yaxis=dict(range=[0, max(capex)*1.1], showgrid=True)
)

fig.show()
