
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAPEX comparison plots:
 - Per-reactor pies (TF/PF highlighted, TF/PF labels outside with leader lines)
 - Stacked bars across reactors (absolute £ and percent-stacked)
Compatible with .csv or .xlsx ("CAPEX" sheet).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams
import matplotlib.patches as mpatches


# =============================================================================
# CONFIG
# =============================================================================
# INPUT
input_path = "capex.csv"   # or "reactor_capex_template.csv"
xlsx_sheet = "CAPEX"                         # used only for .xlsx

# HIGHLIGHTS
HIGHLIGHT_SYSTEMS = ["TF coils", "PF coils"]
HIGHLIGHT_COLORS = {
    "tf coils": "#d62728",   # red
    "pf coils": "#1f77b4",   # blue
}

# PIE SETTINGS
donut_hole_ratio = 0.55
min_pct_slice = 1.0          # collapse slices <1% into "Other" (but never TF/PF)
label_min_pct_pie = 3.0      # only show internal numbers if slice >= 3%
highlight_explode = 0.12
other_explode = 0.00
number_fontsize_pie = 10
outside_pct_threshold = 0.0  # move TF/PF labels outside if >= this % (0 => always outside)
pie_figsize = (9, 9)

# BAR SETTINGS
figure_size_bars = (12, 8)
bar_label_min_fraction = 0.03   # label only segments >= 3% of tallest bar
stacked_alpha = 0.95
edge_width_highlight = 1.2
edge_width_normal = 0.7

# OUTPUT
output_dir_pies = "pies"
output_dir_bars = "bars"

# Optional: neutralize default color cycle (we set colors explicitly anyway)
rcParams['axes.prop_cycle'] = plt.cycler(color=['#666666', '#999999', '#BBBBBB'])


# =============================================================================
# HELPERS
# =============================================================================
def canon(s: str) -> str:
    return str(s).strip().lower()

def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/*?:\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name

def add_total_below_figure(fig, total_value: float, y=0.03, bottom_pad=0.15, title="Total CAPEX"):
    # Reserve bottom space and write the total centered under the chart
    plt.subplots_adjust(bottom=bottom_pad)
    fig.text(0.5, y, f"{title}: £{total_value:,.0f}",
             ha="center", va="center", fontsize=10, fontweight="bold")

def collapse_small_slices(series: pd.Series, min_pct: float, never_collapse: set) -> pd.Series:
    total = float(series.sum())
    if total <= 0:
        return series

    keep = []
    small_sum = 0.0
    for cat, val in series.items():
        if canon(cat) in never_collapse:
            keep.append(cat)
            continue
        pct = (val / total) * 100.0
        if pct < min_pct:
            small_sum += float(val)
        else:
            keep.append(cat)

    collapsed = series.loc[keep].copy()
    if small_sum > 0:
        collapsed.loc["Other"] = collapsed.get("Other", 0.0) + small_sum
    return collapsed

def autopct_factory(values, total, min_pct):
    def inner(pct):
        if total <= 0 or pct < min_pct:
            return ""
        abs_val = pct * total / 100.0
        return f"{pct:.1f}%\n£{abs_val:,.0f}"
    return inner


def make_ppb_palette(n: int,
                     cmaps=("Purples", "PuRd", "BuPu"),
                     lo=0.35, hi=0.90) -> list:
    """
    Return 'n' RGBA colors sampled across purple–pink–blue colormaps.
    Uses version-safe APIs. Always returns a non-empty list (falls back to tab20).
    """
    if n <= 0:
        return []

    colors = []

    # Version-safe getter
    def get_cmap(name: str):
        try:
            return mpl.colormaps.get_cmap(name)   # modern Matplotlib (preferred)
        except Exception:
            return plt.get_cmap(name)             # fallback for older versions

    steps_per_cmap = int(np.ceil(n / max(1, len(cmaps))))

    try:
        for cmap_name in cmaps:
            cmap = get_cmap(cmap_name)
            # sample evenly between lo..hi and append RGBA tuples
            for t in np.linspace(lo, hi, steps_per_cmap):
                colors.append(cmap(float(t)))
                if len(colors) >= n:
                    break
            if len(colors) >= n:
                break

        # Cycle to reach n if needed, else trim to n
        if 0 < len(colors) < n:
            reps = int(np.ceil(n / len(colors)))
            colors = (colors * reps)[:n]
        else:
            colors = colors[:n]

    except Exception:
        # Last-resort categorical fallback (ensures a non-empty list)
        base = list(plt.cm.tab20.colors)
        if not base:
            base = [(0.5, 0.5, 0.5, 1.0)]  # single grey if absolutely            base = [(0.5, 0.5, 0.5, 1.0)]  # single grey if absolutely necessary
        reps = int(np.ceil(n / len(base)))
        colors = (base * reps)[:n]



def build_pie_colors_and_explode(categories, highlight_canon_set):
    # Neutral greys for non-highlighted categories
    neutral_cycle = list(plt.cm.BuPu(np.linspace(0.35, 0.85, max(10, len(categories)))))
    # neutral_cycle = make_ppb_palette(max(10, len(categories)),cmaps=("Purples", "PuRd", "BuPu"),lo=0.35, hi=0.90)
    
    # if not isinstance(neutral_cycle, (list, tuple)) or len(neutral_cycle) == 0:
    #         neutral_cycle = list(plt.cm.tab20.colors)
    #         if len(neutral_cycle) == 0:
    #             neutral_cycle = [(0.6, 0.6, 0.6, 1.0)]

    colors, explodes = [], []
    i_neutral = 0
    for cat in categories:
        ckey = canon(cat)
        if ckey in HIGHLIGHT_COLORS:
            colors.append(HIGHLIGHT_COLORS[ckey])
            explodes.append(highlight_explode)
        else:
            colors.append(neutral_cycle[i_neutral % len(neutral_cycle)])
            explodes.append(other_explode)
            i_neutral += 1
    return colors, explodes

def label_highlights_outside(
    ax,
    categories,          # category names in pie order
    values,              # absolute values in same order
    wedges,              # from ax.pie
    autotexts,           # from ax.pie
    highlight_names,     # set of canonical names to highlight
    outside_pct_threshold=0.0,  # move outside if slice >= this %
    edge_radius=1.02,    # where the leader touches the wedge edge
    text_radius=1.22,    # where the external label sits
    fontsize=10
):
    total = float(np.nansum(values))
    if total <= 0:
        return

    cat2val = {c: float(v) for c, v in zip(categories, values)}

    # small alternating bumps to reduce external label collisions
    extra_bumps = [0.00, 0.06, 0.12]
    bump_idx = 0

    for cat, w, at in zip(categories, wedges, autotexts):
        ckey = canon(cat)
        if ckey not in highlight_names:
            continue

        val = cat2val.get(cat, 0.0)
        pct = (val / total) * 100.0 if total > 0 else 0.0

        if pct < outside_pct_threshold:
            # keep internal label if any
            continue

        # Hide internal number for the highlight (avoid duplicates)
        if at is not None:
            at.set_text("")
            at.set_visible(False)

        # Compute mid-angle of wedge
        ang = 0.5 * (w.theta1 + w.theta2)
        rad = np.deg2rad(ang)

        # Line start (near wedge edge) and text position (further out)
        r0 = edge_radius
        r_txt = text_radius + (extra_bumps[bump_idx % len(extra_bumps)])
        bump_idx += 1

        x0, y0 = np.cos(rad) * r0,  np.sin(rad) * r0
        xt, yt = np.cos(rad) * r_txt, np.sin(rad) * r_txt
        ha = "left" if xt >= 0 else "right"

        label = f"{pct:.1f}%\n£{val:,.0f}"

        # Leader line
        # ax.annotate(
        #     "", xy=(x0, y0), xytext=(xt, yt),
        #     arrowprops=dict(arrowstyle="-", color="black", lw=1.0,
        #                     shrinkA=0, shrinkB=0,
        #                     connectionstyle="arc3,rad=0.15")
        # )

        # External label with stroke for contrast
        txt = ax.text(xt, yt, label, ha=ha, va="center",
                      fontsize=fontsize, fontweight="bold", color="black")
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])


# =============================================================================
# LOAD DATA
# =============================================================================
ext = os.path.splitext(input_path)[1].lower()
if ext == ".xlsx":
    df = pd.read_excel(input_path, sheet_name=xlsx_sheet, engine="openpyxl")
elif ext == ".csv":
    df = pd.read_csv(input_path)
else:
    raise ValueError("Unsupported file type. Use .xlsx or .csv")

required_cols = {"Reactor", "System", "Capex (£)"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean
df["Reactor"] = df["Reactor"].astype(str).str.strip()
df["System"]  = df["System"].astype(str).str.strip()
df["Capex (£)"] = pd.to_numeric(df["Capex (£)"], errors="coerce")
df = df.dropna(subset=["Reactor", "System", "Capex (£)"])
df = df[df["Capex (£)"] >= 0]
df = df[~df["System"].str.contains("TOTAL", case=False, na=False)]

highlight_canon = {canon(s) for s in HIGHLIGHT_SYSTEMS}

# =============================================================================
# PIES (per reactor; TF/PF highlighted; TF/PF labels outside)
# =============================================================================
os.makedirs(output_dir_pies, exist_ok=True)
reactors = df["Reactor"].dropna().unique().tolist()

for reactor in reactors:
    sub = df[df["Reactor"] == reactor]
    agg = sub.groupby("System", as_index=False)["Capex (£)"].sum().set_index("System")["Capex (£)"]
    total = float(agg.sum())
    if total <= 0:
        print(f"Skipping pie for {reactor}: total CAPEX is zero.")
        continue

    # Ensure TF/PF exist even if zero (keeps color/legend consistent)
    for must in highlight_canon:
        if not any(canon(k) == must for k in agg.index):
            display_name = next((k for k in HIGHLIGHT_SYSTEMS if canon(k) == must), None) or must
            agg.loc[display_name] = 0.0

    agg = agg.sort_values(ascending=False)
    data = collapse_small_slices(agg, min_pct_slice, highlight_canon).sort_values(ascending=False)

    categories = data.index.tolist()
    values = data.values
    total = float(values.sum())

    colors, explodes = build_pie_colors_and_explode(categories, highlight_canon)

    fig, ax = plt.subplots(figsize=pie_figsize)
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,  # names in legend to avoid rim collisions
        startangle=90,
        counterclock=False,
        colors=colors,
        explode=explodes,
        autopct=autopct_factory(values, total, label_min_pct_pie),
        pctdistance=0.70 if donut_hole_ratio > 0 else 0.6,
        wedgeprops=dict(linewidth=0.6, edgecolor="white")
    )
    

    legend_labels = categories

    # Use proxy patches with the same face/edge colors and line widths as wedges
    handles = []
    for cat, w in zip(categories, wedges):
        patch = mpatches.Patch(
            facecolor=w.get_facecolor(),
            edgecolor=w.get_edgecolor(),
            linewidth=w.get_linewidth()
        )
        handles.append(patch)

    # Create legend (no reordering); TF/PF remain where their slices are
    leg = ax.legend(
        handles,
        legend_labels,
        title="System",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0
    )


    # Style internal numbers
    for t in autotexts:
        if t is None:
            continue
        t.set_fontsize(number_fontsize_pie)
        t.set_color("black")
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # Donut hole
    if donut_hole_ratio > 0:
        ax.add_artist(plt.Circle((0, 0), donut_hole_ratio, fc="white"))

    # Move TF/PF labels outside with leader lines (always outside by default)
    label_highlights_outside(
        ax,
        categories=categories,
        values=values,
        wedges=wedges,
        autotexts=autotexts,
        highlight_names=highlight_canon,
        outside_pct_threshold=outside_pct_threshold,
        edge_radius=1.02,
        text_radius=1.22,
        fontsize=10
    )

    # Legend: TF/PF first, bold
    # legend_labels, legend_colors = [], []
    # # First, add highlights in requested order
    # for h in HIGHLIGHT_SYSTEMS:
    #     for cat, col in zip(categories, colors):
    #         if canon(cat) == canon(h):
    #             legend_labels.append(cat)
    #             legend_colors.append(col)
    # # Then add the rest
    # for cat, col in zip(categories, colors):
    #     if canon(cat) not in {canon(x) for x in HIGHLIGHT_SYSTEMS}:
    #         legend_labels.append(cat)
    #         legend_colors.append(col)

    # handles = [
    #     plt.Rectangle((0, 0), 1, 1, color=c,
    #                   ec=("black" if canon(l) in highlight_canon else "white"),
    #                   lw=1.0)
    #     for l, c in zip(legend_labels, legend_colors)
    # ]
    # leg = ax.legend(handles, legend_labels, title="System",
    #                 bbox_to_anchor=(1.02, 1), loc="upper left")
    # for txt in leg.get_texts():
    #     if canon(txt.get_text()) in highlight_canon:
    #         txt.set_fontweight("bold")

    ax.set_title(f"{reactor} – CAPEX by System (TF/PF highlighted)", fontsize=14)
    ax.axis("equal")

    # Total beneath chart
    add_total_below_figure(fig, total)

    # Save
    os.makedirs(output_dir_pies, exist_ok=True)
    fname = os.path.join(output_dir_pies, f"pie_{sanitize_filename(reactor)}_system_highlight_tfpfs.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved:", fname)

# =============================================================================
# STACKED BARS (absolute £ and percent-stacked)
# =============================================================================
os.makedirs(output_dir_bars, exist_ok=True)

# Aggregate Reactor × System
agg_rs = df.groupby(["Reactor", "System"], as_index=False)["Capex (£)"].sum()

# Order systems: highlights first, then remaining by global total
systems_global = agg_rs.groupby("System")["Capex (£)"].sum().sort_values(ascending=False)
systems_order = [s for s in HIGHLIGHT_SYSTEMS if s in systems_global.index]
systems_order += [s for s in systems_global.index if s not in systems_order]

# Reactor order by total descending
reactor_totals = agg_rs.groupby("Reactor")["Capex (£)"].sum().sort_values(ascending=False)
reactor_order = reactor_totals.index.tolist()

# Pivot: reactors (rows) × systems (cols)
pivot = agg_rs.pivot(index="Reactor", columns="System", values="Capex (£)").fillna(0.0)
pivot = pivot.reindex(index=reactor_order, columns=systems_order)

# Colors for bars
neutral_cycle = list(plt.cm.Greys(np.linspace(0.35, 0.85, max(10, len(pivot.columns)))))

# palette = make_ppb_palette(max(10, len(pivot.columns)),cmaps=("Purples", "PuRd", "BuPu"),lo=0.35, hi=0.90)

# if not isinstance(palette, (list, tuple)) or len(palette) == 0:
#     palette = list(plt.cm.tab20.colors)
#     if len(palette) == 0:
#         palette = [(0.6, 0.6, 0.6, 1.0)]

bar_colors = []
i_neutral = 0
for system in pivot.columns:
    ckey = canon(system)
    if ckey in HIGHLIGHT_COLORS:
        bar_colors.append(HIGHLIGHT_COLORS[ckey])
    else:
        bar_colors.append(neutral_cycle[i_neutral % len(neutral_cycle)])
        i_neutral += 1

# --- Absolute stacked bar ---
fig, ax = plt.subplots(figsize=figure_size_bars)
bottom = np.zeros(len(pivot.index))
containers = []
for i, system in enumerate(pivot.columns):
    values = pivot[system].values
    cont = ax.bar(
        pivot.index, values, bottom=bottom,
        label=system, color=bar_colors[i],
        edgecolor=("black" if canon(system) in highlight_canon else "white"),
        linewidth=(edge_width_highlight if canon(system) in highlight_canon else edge_width_normal),
        alpha=stacked_alpha
    )
    containers.append(cont)
    bottom = bottom + values

ax.set_title("CAPEX by System across Reactors (Stacked)", fontsize=14)
ax.set_ylabel("CAPEX (£)")
ax.set_xlabel("Reactor")
ax.legend(title="System", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
ax.margins(y=0.05)

# Label segments >= threshold (container-safe) and apply path effects after
max_total = pivot.sum(axis=1).max()
threshold = max_total * bar_label_min_fraction

for cont in containers:
    labels = []
    for p in cont.patches:
        h = p.get_height()
        labels.append(f"£{h:,.0f}" if h >= threshold else "")
    text_objs = ax.bar_label(
        cont,
        labels=labels,
        label_type="center",
        fontsize=9,
        color="white",
        padding=1
    )
    for t in text_objs:
        if t.get_text():
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

totals = pivot.sum(axis=1)

# Make room below the axis
fig.subplots_adjust(bottom=0.20)  # push bottom margin to avoid clipping

# Position totals below the x-axis using axis transform
bottom_offset = 0.08  # amount below x-axis; increase if overlapping tick labels

for x_idx, (reactor, tot) in enumerate(totals.items()):
    txt = ax.text(
        x_idx, -bottom_offset,           # x in data coords (bar index), y in axes fraction
        f"£{tot:,.0f}",
        transform=ax.get_xaxis_transform(),  # (data-x, axes-y) space
        ha="center", va="top",
        fontsize=9, color="black"
    )
    # Stroke for readability (version-safe: set after creation)
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

plt.tight_layout()
fig.subplots_adjust(bottom=0.12)

bar_abs_file = os.path.join(output_dir_bars, "capex_by_system_stacked.png")
plt.savefig(bar_abs_file, dpi=300)
plt.close(fig)
print("Saved:", bar_abs_file)

# --- Percent-stacked bar ---
pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

fig2, ax2 = plt.subplots(figsize=figure_size_bars)
bottom = np.zeros(len(pct.index))
containers2 = []
for i, system in enumerate(pct.columns):
    values = pct[system].values
    cont = ax2.bar(
        pct.index, values, bottom=bottom,
        label=system, color=bar_colors[i],
        edgecolor=("black" if canon(system) in highlight_canon else "white"),
        linewidth=(edge_width_highlight if canon(system) in highlight_canon else edge_width_normal),
        alpha=stacked_alpha
    )
    containers2.append(cont)
    bottom = bottom + values


totals_abs = pivot.sum(axis=1)  # absolute totals per reactor (even though bars are in %)

fig2.subplots_adjust(bottom=0.20)
bottom_offset_pct = 0.08

for x_idx, (reactor, tot) in enumerate(totals_abs.items()):
    txt = ax2.text(
        x_idx, -bottom_offset_pct,
        f"£{tot:,.0f}",
        transform=ax2.get_xaxis_transform(),
        ha="center", va="top",
        fontsize=9, color="black"
    )
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
 
ax2.set_title("CAPEX by System across Reactors (Percent-Stacked)", fontsize=14)
ax2.set_ylabel("Share of Reactor Total (%)")
ax2.set_xlabel("Reactor")
ax2.set_ylim(0, 100)
ax2.legend(title="System", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
ax2.margins(y=0.05)

# Label percent segments >= threshold (container-safe) and apply path effects after
pct_threshold = 100.0 * bar_label_min_fraction

for cont in containers2:
    labels = []
    for p in cont.patches:
        h = p.get_height()
        labels.append(f"{h:.1f}%" if h >= pct_threshold else "")
    text_objs = ax2.bar_label(
        cont,
        labels=labels,
        label_type="center",
        fontsize=9,
        color="white",
        padding=1
    )
    for t in text_objs:
        if t.get_text():
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

plt.tight_layout()
fig2.subplots_adjust(bottom=0.12)
# fig2.text(0.5, 0.03, "Totals vary per reactor; bars normalized to 100%",
#           ha="center", va="center", fontsize=10)

bar_pct_file = os.path.join(output_dir_bars, "capex_by_system_percent_stacked.png")
plt.savefig(bar_pct_file, dpi=300)
plt.close(fig2)
print("Saved:", bar_pct_file)