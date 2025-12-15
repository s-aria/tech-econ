
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# =========================
# CONFIG
# =========================
input_path = "capex.csv"  # or .csv
sheet_name = "CAPEX"                        # used only for .xlsx
output_dir = "pies"

HIGHLIGHT_SYSTEMS = ["TF coils", "PF coils"]
HIGHLIGHT_COLORS = {
    "tf coils": "#d62728",  # red
    "pf coils": "#1f77b4",  # blue
}
HIGHLIGHT_EXPLODE = 0.12
OTHER_EXPLODE = 0.00

donut_hole_ratio = 0.55
min_pct_slice = 1.0        # collapse slices <1% into "Other" (but never TF/PF)
label_min_pct = 3.0        # show numbers only for slices >= 3%
number_fontsize = 10       # inside-wedge number font

# =========================
# LOAD & CLEAN
# =========================
ext = os.path.splitext(input_path)[1].lower()
if ext == ".xlsx":
    df = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")
elif ext == ".csv":
    df = pd.read_csv(input_path)
else:
    raise ValueError("Unsupported file type. Use .xlsx or .csv")

df["Reactor"] = df["Reactor"].astype(str).str.strip()
df["System"]  = df["System"].astype(str).str.strip()
df["Capex (£)"] = pd.to_numeric(df["Capex (£)"], errors="coerce")
df = df.dropna(subset=["Reactor", "System", "Capex (£)"])
df = df[df["Capex (£)"] >= 0]
df = df[~df["System"].str.contains("TOTAL", case=False, na=False)]

def canon(s: str) -> str:
    return str(s).strip().lower()

highlight_canon = {canon(s) for s in HIGHLIGHT_SYSTEMS}

def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/*?:\"<>|]", "_", name)
    name = re.sub(r"\\s+", "_", name)
    return name

def collapse_small_slices(series: pd.Series, min_pct: float, never_collapse: set):
    total = series.sum()
    if total <= 0: return series
    keep = []
    small_sum = 0.0
    for cat, val in series.items():
        if canon(cat) in never_collapse:
            keep.append(cat)
        else:
            pct = (val / total) * 100.0
            if pct < min_pct: small_sum += val
            else: keep.append(cat)
    collapsed = series.loc[keep].copy()
    if small_sum > 0:
        collapsed.loc["Other"] = collapsed.get("Other", 0.0) + small_sum
    return collapsed

def build_colors(categories):
    neutral_cycle = list(plt.cm.Greys(np.linspace(0.35, 0.85, 10)))
    colors, i_neutral = [], 0
    for cat in categories:
        ccat = canon(cat)
        if ccat in HIGHLIGHT_COLORS:
            colors.append(HIGHLIGHT_COLORS[ccat])
        else:
            colors.append(neutral_cycle[i_neutral % len(neutral_cycle)])
            i_neutral += 1
    return colors

def build_explode(categories):
    return [HIGHLIGHT_EXPLODE if canon(c) in highlight_canon else OTHER_EXPLODE for c in categories]

def autopct_factory(values, total):
    def inner(pct):
        if total <= 0 or pct < label_min_pct:
            return ""
        abs_val = pct * total / 100.0
        return f"{pct:.1f}%\n£{abs_val:,.0f}"
    return inner

# =========================
# GENERATE PIES (names in legend; only numbers inside)
# =========================
os.makedirs(output_dir, exist_ok=True)
reactors = df["Reactor"].dropna().unique().tolist()

saved = []
for reactor in reactors:
    sub = df[df["Reactor"] == reactor]
    agg = sub.groupby("System")["Capex (£)"].sum().sort_values(ascending=False)
    total = agg.sum()
    if total <= 0:
        print(f"Skipping {reactor}: total CAPEX is zero.")
        continue

    # Ensure TF/PF exist (even if zero), so legend/color stays consistent
    for must in highlight_canon:
        if not any(canon(k) == must for k in agg.index):
            display_name = next((k for k in HIGHLIGHT_SYSTEMS if canon(k) == must), None) or must
            agg.loc[display_name] = 0.0

    agg = agg.sort_values(ascending=False)
    data = collapse_small_slices(agg, min_pct=min_pct_slice, never_collapse=highlight_canon).sort_values(ascending=False)

    categories = data.index.tolist()
    values = data.values
    colors = build_colors(categories)
    explode = build_explode(categories)

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # ← no names around the pie
        startangle=90, counterclock=False,
        colors=colors, explode=explode,
        autopct=autopct_factory(values, total),
        pctdistance=0.70 if donut_hole_ratio > 0 else 0.6,
        wedgeprops=dict(linewidth=0.5, edgecolor="white")
    )

    # Make numbers readable (stroke around text)
    for t in autotexts:
        if t is None: continue
        t.set_fontsize(number_fontsize)
        t.set_color("black")
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # Donut hole
    if donut_hole_ratio > 0:
        ax.add_artist(plt.Circle((0, 0), donut_hole_ratio, fc="white"))

    # Legend: TF/PF first, bold text
    legend_labels, legend_colors = [], []
    for h in HIGHLIGHT_SYSTEMS:
        for cat, col in zip(categories, colors):
            if canon(cat) == canon(h):
                legend_labels.append(cat); legend_colors.append(col)
    for cat, col in zip(categories, colors):
        if canon(cat) not in {canon(x) for x in HIGHLIGHT_SYSTEMS}:
            legend_labels.append(cat); legend_colors.append(col)

    leg_handles = [
        plt.Rectangle((0,0),1,1,color=c, ec="black" if canon(l) in highlight_canon else "white", lw=1.0)
        for l, c in zip(legend_labels, legend_colors)
    ]
    leg = ax.legend(leg_handles, legend_labels, title="System", bbox_to_anchor=(1.02, 1), loc="upper left")
    # Bold TF/PF in legend
    for txt in leg.get_texts():
        if canon(txt.get_text()) in highlight_canon:
            txt.set_fontweight("bold")

    ax.set_title(f"{reactor} – CAPEX by System (TF/PF highlighted)", fontsize=14)
    ax.axis("equal")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"pie_{sanitize_filename(reactor)}_system_highlight_tfpfs.png")
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    saved.append(fname)

print("Saved pies (no overlap, names in legend):")
for f in saved:
    print(" -", f)
