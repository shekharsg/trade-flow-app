# app.py
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc

st.set_page_config(page_title="Virtual N Global Trade Flow Explorer", layout="wide")

# =====================================================================================
# DATA LOADING (robust, with sidebar diagnostics and Git LFS pointer detection)
# =====================================================================================
DATA_CANDIDATES = [
    "data/trade_major_crops.csv.xz",      # preferred reduced file (small)
    "data/trade_major_crops.parquet",     # optional parquet
    "data/exp_import_max_value_n_ton.csv" # legacy large file (often causes issues)
]

REQUIRED_COLS = [
    "Year", "Category", "Item", "original_Item",
    "Source_Countries", "Target_Countries", "value_kg_n"
]

def list_data_dir():
    st.sidebar.subheader("📂 data/ folder")
    if not os.path.isdir("data"):
        st.sidebar.error("`data/` folder missing at repo root.")
        return
    for fname in sorted(os.listdir("data")):
        fpath = os.path.join("data", fname)
        try:
            st.sidebar.write(f"• `{fname}` — {os.path.getsize(fpath)/1024/1024:.2f} MB")
        except Exception:
            st.sidebar.write(f"• `{fname}`")

def is_lfs_pointer(path: str) -> bool:
    # LFS pointer files are tiny text files starting with this line:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs")
    except Exception:
        return False

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".gz", ".bz2", ".xz"}:
        return pd.read_csv(path, compression="infer")
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data_auto() -> pd.DataFrame:
    list_data_dir()
    for p in DATA_CANDIDATES:
        if not os.path.exists(p):
            st.sidebar.info(f"Skip (missing): `{p}`")
            continue
        if is_lfs_pointer(p):
            st.sidebar.error(f"`{p}` is a **Git LFS pointer**. Commit the real file (untrack via LFS).")
            continue
        try:
            size = os.path.getsize(p)
            if size == 0:
                st.sidebar.error(f"`{p}` is **0 bytes**.")
                continue
            df = _read_any(p)
            if df is None or df.shape[1] == 0:
                st.sidebar.error(f"`{p}` loaded but has **no columns**.")
                continue
            st.sidebar.success(f"Loaded dataset: `{p}` ({size/1024/1024:.2f} MB)")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed reading `{p}` → {e}")
            continue
    st.error(
        "No usable dataset found.\n\n"
        "Ensure `data/trade_major_crops.csv.xz` is committed **without Git LFS** "
        "and is larger than a few MB (not a tiny pointer file)."
    )
    st.stop()

trade_df = load_data_auto()

# Validate columns
missing = [c for c in REQUIRED_COLS if c not in trade_df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# Types and helper lowercase columns
trade_df["Year"] = pd.to_numeric(trade_df["Year"], errors="coerce").astype("Int64").astype(int)
trade_df["value_kg_n"] = pd.to_numeric(trade_df["value_kg_n"], errors="coerce").fillna(0.0).astype(float)
trade_df["cat_lc"]  = trade_df["Category"].astype(str).str.strip().str.lower()
trade_df["item_lc"] = trade_df["Item"].astype(str).str.strip().str.lower()
trade_df["orig_item_lc"] = trade_df["original_Item"].astype(str).str.strip().str.lower()
trade_df["src_lc"]  = trade_df["Source_Countries"].astype(str).str.strip().str.lower()
trade_df["tgt_lc"]  = trade_df["Target_Countries"].astype(str).str.strip().str.lower()

# =====================================================================================
# BASEMAP (GeoPandas built-in, no external shapefiles)
# =====================================================================================
geod = Geod(ellps="WGS84")

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))[["name", "geometry"]]
world = world.rename(columns={"name": "ADMIN"})
world_proj = world.to_crs(epsg=3857)
world["coords"] = world_proj.centroid.to_crs(world.crs)
country_coords = world.set_index("ADMIN")["coords"].to_dict()
world_names_lc = {k.lower(): k for k in country_coords.keys()}

def _canonical_country(name: str):
    if not isinstance(name, str):
        return None
    return world_names_lc.get(name.strip().lower())

def great_circle_points(src, tgt, npoints=50):
    pts = geod.npts(src.x, src.y, tgt.x, tgt.y, npoints)
    lons = [src.x] + [p[0] for p in pts] + [tgt.x]
    lats = [src.y] + [p[1] for p in pts] + [tgt.y]
    return lons, lats

# =====================================================================================
# PLOTTING
# =====================================================================================
def plot_trade_flow(year, category, crop, orig_crop, country, flow_type="export", partner="All countries"):
    cat_lc = str(category).lower()
    crop_lc = str(crop).lower()
    country_lc = str(country).lower()
    partner_lc = None if partner == "All countries" else str(partner).lower()

    if flow_type == "export":
        df = trade_df[
            (trade_df["Year"] == year) &
            (trade_df["cat_lc"] == cat_lc) &
            (trade_df["item_lc"] == crop_lc) &
            (trade_df["src_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n", "tgt_lc"]].copy()
        if partner_lc is not None:
            df = df[df["tgt_lc"] == partner_lc]
        df = df.rename(columns={"Source_Countries": "source", "Target_Countries": "target"})
        title = f"{country}: {crop} ({category}) / {orig_crop} — Exports ({year})"
    else:
        df = trade_df[
            (trade_df["Year"] == year) &
            (trade_df["cat_lc"] == cat_lc) &
            (trade_df["item_lc"] == crop_lc) &
            (trade_df["tgt_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n", "src_lc"]].copy()
        if partner_lc is not None:
            df = df[df["src_lc"] == partner_lc]
        df = df.rename(columns={"Source_Countries": "source", "Target_Countries": "target"})
        title = f"{country}: {crop} ({category}) / {orig_crop} — Imports ({year})"

    if df.empty:
        st.warning("⚠️ No trade data found for this selection.")
        return

    df = df.groupby(["source", "target"], as_index=False)["value_kg_n"].sum()
    df = df.rename(columns={"value_kg_n": "weight"}).query("weight > 0")

    # Map name canonicalization
    df["source_map"] = df["source"].map(_canonical_country)
    df["target_map"] = df["target"].map(_canonical_country)
    df = df.dropna(subset=["source_map", "target_map"])
    if df.empty:
        st.warning("⚠️ Countries not found in basemap for these flows.")
        return

    # Styling
    df["log_weight"] = np.log10(df["weight"] + 1)
    norm = plt.Normalize(vmin=df["log_weight"].min(), vmax=df["log_weight"].max())
    cmap = plt.colormaps.get_cmap("Spectral_r")

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, alpha=0.3)
    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    mx = df["log_weight"].max()
    for _, row in df.iterrows():
        src = country_coords.get(row["source_map"])
        tgt = country_coords.get(row["target_map"])
        if src is None or tgt is None:
            continue
        arc_x, arc_y = great_circle_points(src, tgt, npoints=50)
        color = cmap(norm(row["log_weight"]))
        lw = 0.5 + 2.5 * (row["log_weight"] / mx if mx > 0 else 1.0)
        ax.plot(arc_x, arc_y, transform=ccrs.Geodetic(), color=color, linewidth=lw, alpha=0.8)
        idx = int(len(arc_x) * 0.7)  # arrow toward target
        ax.annotate("", xy=(arc_x[idx], arc_y[idx]), xycoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    xytext=(arc_x[idx-1], arc_y[idx-1]), textcoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical", shrink=0.5, ax=ax)
    cbar.set_label("Log-scaled Trade Flow (kg N)")
    plt.title(title, fontsize=13)
    st.pyplot(fig)

def plot_sankey(df, country, year, crop, orig_crop, category, threshold, flow_type="export", palette_choice="Set3"):
    cat_lc = str(category).lower()
    crop_lc = str(crop).lower()
    country_lc = str(country).lower()

    if flow_type == "export":
        dff = df[
            (df["Year"] == year) & (df["cat_lc"] == cat_lc) &
            (df["item_lc"] == crop_lc) & (df["src_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]].copy()
    else:
        dff = df[
            (df["Year"] == year) & (df["cat_lc"] == cat_lc) &
            (df["item_lc"] == crop_lc) & (df["tgt_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]].copy()

    if dff.empty:
        st.warning(f"⚠️ No {'exports' if flow_type=='export' else 'imports'} found for {country} in {year}.")
        return

    dff = dff.groupby(["Source_Countries", "Target_Countries"], as_index=False).sum()
    dff = dff.rename(columns={"Source_Countries": "source", "Target_Countries": "target", "value_kg_n": "weight_n"})

    total = dff["weight_n"].sum()
    keep_mask = dff["weight_n"] > threshold * total
    if keep_mask.sum() < 10 and len(dff) > 10:
        dff = dff.nlargest(10, "weight_n")
    else:
        dff = dff[keep_mask]

    exporters = dff["source"].unique().tolist()
    importers = dff["target"].unique().tolist()
    all_nodes = exporters + importers
    node_map = {n: i for i, n in enumerate(all_nodes)}

    palettes = {"Set3": pc.qualitative.Set3, "Dark2": pc.qualitative.Dark2, "Pastel1": pc.qualitative.Pastel1}
    pal = palettes.get(palette_choice, pc.qualitative.Set3)

    if flow_type == "export":
        tcolors = {imp: pal[i % len(pal)] for i, imp in enumerate(importers)}
        node_colors = ["#636efa" if n in exporters else tcolors[n] for n in all_nodes]
        link_colors = [tcolors[t] for t in dff["target"]]
    else:
        scolors = {src: pal[i % len(pal)] for i, src in enumerate(exporters)}
        node_colors = [scolors[n] if n in exporters else "#0ed6c1" for n in all_nodes]
        link_colors = [scolors[s] for s in dff["source"]]

    sources = dff["source"].map(node_map)
    targets = dff["target"].map(node_map)
    values = dff["weight_n"]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap", orientation="h",
        node=dict(
            pad=24, thickness=20, line=dict(color="#333", width=1.7),
            label=[f"{n}" for n in all_nodes], color=node_colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate=('<b>Exporter:</b> %{source.label}<br>'
                           '<b>Importer:</b> %{target.label}<br>'
                           '<b>Value:</b> %{value:,.0f} kg N<br>'
                           f'<b>Year:</b> {year}<br>'
                           f'<b>Crop:</b> {crop} / {orig_crop}<br>'
                           f'<b>Category:</b> {category}<extra></extra>')
        )
    )])

    fig.update_layout(
        title_text=f"📊 {country}: {crop} ({category}) / {orig_crop} — {'Exports' if flow_type=='export' else 'Imports'} {year}",
        font=dict(family="Open Sans, Arial", size=14, color="#222"),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        margin=dict(l=20, r=80, t=60, b=40), height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# UI
# =====================================================================================
st.title("🌍 Virtual N Global Trade Flow Explorer (Major Crops)")

# Sidebar controls
st.sidebar.header("Filter & Customize Your View")

year_selected = st.sidebar.slider(
    "Select Year",
    min_value=int(trade_df["Year"].min()),
    max_value=int(trade_df["Year"].max()),
    value=int(trade_df["Year"].max()),
    step=1, format="%d"
)

category_selected = st.sidebar.selectbox(
    "Select Category",
    sorted(trade_df["Category"].dropna().unique())
)

pairs = (trade_df[["Item", "original_Item"]]
         .drop_duplicates()
         .sort_values(["Item", "original_Item"]))
labels = [f"{r.Item} → {r.original_Item}" for _, r in pairs.iterrows()]
label = st.sidebar.selectbox("Select Crop (standardized → original)", labels)
item_selected, original_selected = label.split(" → ", 1)

country_search = st.sidebar.text_input("Search Source Country (case-insensitive)", "India")
src_unique = trade_df["Source_Countries"].dropna().unique()
countries_lower = [c.lower() for c in src_unique]
source_selected = src_unique[countries_lower.index(country_search.strip().lower())] if country_search.strip().lower() in countries_lower else "India"

target_selected = st.sidebar.selectbox(
    "Select Importing Country",
    ["All countries"] + sorted(trade_df["Target_Countries"].dropna().unique())
)

threshold = st.sidebar.slider(
    "Minimum trade flow fraction (filter small flows)",
    min_value=0.0, max_value=0.1, value=0.01, step=0.005
)

pal_choice = st.sidebar.radio(
    "Choose Sankey Color Palette",
    ("Set3", "Dark2", "Pastel1"), index=0
)

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About this dashboard"):
    st.markdown(
        """
        - Reads a reduced dataset (**`data/trade_major_crops.csv.xz`** preferred).
        - If you used Git LFS earlier, make sure this file is **committed without LFS**.
        - Map shows great-circle arcs; Sankey shows major flows (threshold trims tiny links).
        """
    )

# Main views
st.header("🗺️ Export Flows Map")
plot_trade_flow(
    year_selected, category_selected, item_selected, original_selected,
    source_selected, flow_type="export", partner=target_selected
)

st.header("📤 Export Flows Sankey")
plot_sankey(
    trade_df, source_selected, year_selected, item_selected, original_selected,
    category_selected, threshold, flow_type="export", palette_choice=pal_choice
)

st.header("🗺️ Import Flows Map")
plot_trade_flow(
    year_selected, category_selected, item_selected, original_selected,
    target_selected if target_selected != "All countries" else source_selected,
    flow_type="import"
)

st.header("📥 Import Flows Sankey")
plot_sankey(
    trade_df, source_selected, year_selected, item_selected, original_selected,
    category_selected, threshold, flow_type="import", palette_choice=pal_choice
)
