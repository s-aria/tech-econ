
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-plant totals for: (magnet systems + magnet-dependent systems),
report the delta between two plants, and plot one pie per plant over ONLY
the selected systems (magnets + dependencies) with total shown under the chart.

Works with .csv or .xlsx (sheet 'CAPEX').
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

# =============================================================================
# CONFIG
# =============================================================================
INPUT_PATH  = "capex.csv"  # or "reactor_capex_template.csv"
XLSX_SHEET  = "CAPEX"                        # used only for .xlsx

# Two plants to compare
REACTOR_A   = "SPP2.4"
REACTOR_B   = "SPP2.5"

# Define your magnets & dependencies
MAGNET_SYSTEMS = [
    "TF coils",
    "PF coils",
]
DEPENDENT_SYSTEMS = [
    "Cryoplant",
    "Cryostat",
    # Add/remove as needed: e.g., "Maintenance", "Buildings", "Vacuum vessel"
]

# Colors (fixed, simple, readable)
SYSTEM_COLORS = {
    "tf coils":  "#d62728",  # red
    "pf coils":  "#1f77b4",  # blue
    "cryoplant": "#7b2cbf",  # purple
    "cryostat":  "#ff7fbf",  # pink
}
DEFAULT_COLOR = "#bfbfbf"    # fallback if a listed system isn't in the dict

# Pie styling (one pie per plant over magnets+deps only)
PIE_FIGSIZE      = (9, 9)
DONUT_HOLE_RATIO = 0.55   # 0 => full pie, 0.55 => donut
LABEL_MIN_PCT    = 3.0    # show slice labels only for >= 3%
NUMBER_FONTSIZE  = 10
EDGE_WIDTH       = 0.6

OUTPUT_DIR = "pies"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def autopct_factory(values, total, min_pct):
    """Percent + absolute £, with threshold."""
    def inner(pct):
        if total <= 0 or pct < min_pct:
            return ""
        abs_val = pct * total / 100.0
        return f"{pct:.1f}%\n£{abs_val:,.0f}"
    return inner

def add_total_below_figure(fig, total_value: float, y=0.03, bottom_pad=0.18, title="Magnets-related total CAPEX"):
    plt.subplots_adjust(bottom=bottom_pad)
    fig.text(0.5, y, f"{title}: £{total_value:,.0f}",
             ha="center", va="center", fontsize=10, fontweight="bold")

def load_capex(path, sheet="CAPEX"):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .xlsx or .csv")
    # Basic cleanup
    required = {"Reactor", "System", "Capex (£)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Reactor"] = df["Reactor"].astype(str).str.strip()
    df["System"]  = df["System"].astype(str).str.strip()
    df["Capex (£)"] = pd.to_numeric(df["Capex (£)"], errors="coerce")
    df = df.dropna(subset=["Reactor", "System", "Capex (£)"])
    df = df[df["Capex (£)"] >= 0]
    df = df[~df["System"].str.contains("TOTAL", case=False, na=False)]
    return df

def sum_selected_systems(df, reactor_name, systems_list):
    """Sum CAPEX for the given reactor across the specified systems_list."""
    sub = df[df["Reactor"] == reactor_name]
    if sub.empty:
        return 0.0
    # Ensure all systems exist (missing treated as 0)
    capex = 0.0
    for sys in systems_list:
        capex += float(sub[sub["System"] == sys]["Capex (£)"].sum())
    return capex

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df = load_capex(INPUT_PATH, XLSX_SHEET)

    # The set of systems to include in the pies and totals
    selected_systems = MAGNET_SYSTEMS + DEPENDENT_SYSTEMS
    selected_canon = {canon(s) for s in selected_systems}

    # Compute totals (magnets + dependencies) for the two reactors
    total_A = sum_selected_systems(df, REACTOR_A, selected_systems)
    total_B = sum_selected_systems(df, REACTOR_B, selected_systems)
    delta   = total_B - total_A

    # Print summary
    print("=== Magnets-related CAPEX (magnets + dependencies) ===")
    print(f"{REACTOR_A}: £{total_A:,.0f}")
    print(f"{REACTOR_B}: £{total_B:,.0f}")
    sign = "+" if delta >= 0 else "-"
    print(f"Δ ({REACTOR_B} - {REACTOR_A}): {sign}£{abs(delta):,.0f}")

    # --- Build one pie per reactor over ONLY selected systems ---
    reactors = [REACTOR_A, REACTOR_B]
    for reactor in reactors:
        sub = df[df["Reactor"] == reactor]

        # Aggregate selected systems
        rows = []
        for sys in selected_systems:
            val = float(sub[sub["System"] == sys]["Capex (£)"].sum())
            rows.append((sys, val))

        categories = [name for name, _ in rows]
        values     = [val for _,   val in rows]

        total_sel  = float(np.sum(values))
        if total_sel <= 0:
            print(f"Skipping {reactor}: magnets-related total is zero.")
            continue

        # Colors mapped to categories (fallback color if not specified)
        colors = []
        for cat in categories:
            ckey = canon(cat)
            colors.append(SYSTEM_COLORS.get(ckey, DEFAULT_COLOR))

        # Plot pie (no labels around rim; legend will carry names)
        fig, ax = plt.subplots(figsize=PIE_FIGSIZE)
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            startangle=90,
            counterclock=False,
            colors=colors,
            explode=[0.0] * len(values),
            autopct=autopct_factory(values, total_sel, LABEL_MIN_PCT),
            pctdistance=0.70 if DONUT_HOLE_RATIO > 0 else 0.6,
            wedgeprops=dict(linewidth=EDGE_WIDTH, edgecolor="white")
        )

        # Donut hole
        if DONUT_HOLE_RATIO > 0:
            ax.add_artist(plt.Circle((0, 0), DONUT_HOLE_RATIO, fc="white"))

        # Style numbers
        for t in autotexts:
            if t is None:
                continue
            t.set_fontsize(NUMBER_FONTSIZE)
            t.set_color("black")
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        # Legend in wedge order (no manual reordering)
        handles = []
        legend_labels = categories
        for w in wedges:
            patch = mpatches.Patch(
                facecolor=w.get_facecolor(),
                edgecolor=w.get_edgecolor(),
                linewidth=w.get_linewidth()
            )
            handles.append(patch)

        leg = ax.legend(
            handles,
            legend_labels,
            title="System",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0
        )

        # Optional: bold magnets (TF/PF) in legend
        for txt in leg.get_texts():
            if canon(txt.get_text()) in {canon(x) for x in MAGNET_SYSTEMS}:
                txt.set_fontweight("bold")

        ax.set_title(f"{reactor} - Magnets + Dependencies CAPEX", fontsize=14)
        ax.axis("equal")

        # Show magnets-related total under the chart
        add_total_below_figure(fig, total_sel, y=0.03, bottom_pad=0.20,
                               title="Magnets + dependencies total CAPEX")

        # Save
        fname = os.path.join(OUTPUT_DIR, f"pie_{sanitize_filename(reactor)}_magnets_plus_dependencies.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close(fig)
        print("Saved:", fname)

    # --- Optional: quick delta bar (two bars + delta annotation) ---
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        reactors = [REACTOR_A, REACTOR_B]
        totals   = [total_A, total_B]
        colors   = ["#B200ED", "#8F00FF"]  # neutral for A, blue for B

        bars = ax.bar(reactors, totals, color=colors, edgecolor="black", linewidth=1.0)

        ax.set_title("Magnets + Dependencies CAPEX — totals and Δ")
        ax.set_ylabel("CAPEX (£)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # Labels on top
        for b in bars:
            h  = b.get_height()
            xc = b.get_x() + b.get_width()/2.0
            txt = ax.text(xc, h, f"£{h:,.0f}", ha="center", va="bottom", fontsize=9, color="black")
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        
        fig.subplots_adjust(bottom=0.20)

        delta_label = f"Δ ({REACTOR_B} - {REACTOR_A}) = {('+' if delta >= 0 else '-')}£{abs(delta):,.0f}"

        # x in data coords (use center between bars), y in axes coords (slightly below the axis line)
        ax.text(
            0.5, -0.10,                 # x=0.5 (axes center), y below x-axis
            delta_label,
            transform=ax.transAxes,     # (axes fraction coordinates)
            ha="center", va="top",
            fontsize=10, fontweight="bold", color="black"
        )


        plt.tight_layout()
        out_bar = os.path.join(OUTPUT_DIR, "magnets_plus_dependencies_totals_and_delta.png")
        plt.savefig(out_bar, dpi=300)
        plt.close(fig)
        print("Saved:", out_bar)
    except Exception as e:
        # Chart is optional; keep the numeric results
        print("Delta bar chart skipped due to:", e)
