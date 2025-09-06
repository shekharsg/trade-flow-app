# app_trade_major.py
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pyproj import Geod
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc

# =========================
# Config: update file paths
# =========================
DATA_FILE = "data/trade_major_crops.csv.xz"           # <-- set to your reduced file
WORLD_SHP = "shapefiles/ne_110m_admin_0_countries.shp"

# -------------------------
# Great-circle setup
# -------------------------
geod = Geod(ellps="WGS84")

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".gz", ".bz2", ".xz"]:
        return pd.read_csv(path, compression="infer")
    if ext in [".csv"]:
        return pd.read_csv(path)
    # Fallback: try CSV infer
    return pd.read_csv(path, compression="infer")

@st.cache_data(show_spinner=False)
def load_data(path=DATA_FILE) -> pd.DataFrame:
    df = _read_any(path)

    # Ensure required columns exist
    needed = [
        "Year", "Category", "Item", "original_Item",
        "Source_Countries", "Target_Countries", "value_kg_n"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Dtypes & cleaning
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64").astype(int)
    df["value_kg_n"] = pd.to_numeric(df["value_kg_n"], errors="coerce").fillna(0.0).astype(float)

    # Helper lowercase columns for fast case-insensitive filtering
    df["cat_lc"]  = df["Category"].astype(str).str.strip().str.lower()
    df["item_lc"] = df["Item"].astype(str).str.strip().str.lower()
    df["orig_item_lc"] = df["original_Item"].astype(str).str.strip().str.lower()
    df["src_lc"]  = df["Source_Countries"].astype(str).str.strip().str.lower()
    df["tgt_lc"]  = df["Target_Countries"].astype(str).str.strip().str.lower()
    return df

trade_df = load_data()

# -------------------------
# World shapefile & centroids
# -------------------------
world = gpd.read_file(WORLD_SHP)
if "ADMIN" not in world.columns and "NAME" in world.columns:
    world = world.rename(columns={"NAME": "ADMIN"})
world = world[["ADMIN", "geometry"]].copy()

# Compute centroids correctly (in projected CRS) then back to geographic
world_proj = world.to_crs(epsg=3857)
world["coords"] = world_proj.centroid.to_crs(world.crs)
country_coords = world.set_index("ADMIN")["coords"].to_dict()
world_names_lc = {k.lower(): k for k in country_coords.keys()}  # lowercase -> canonical

def _canonical_country(name: str):
    if not isinstance(name, str):
        return None
    return world_names_lc.get(name.strip().lower())

def great_circle_points(src, tgt, npoints=50):
    points = geod.npts(src.x, src.y, tgt.x, tgt.y, npoints)
    lons = [src.x] + [p[0] for p in points] + [tgt.x]
    lats = [src.y] + [p[1] for p in points] + [tgt.y]
    return lons, lats

def short_fmt(val):
    if val >= 1e6:
        return f"{val/1e6:.1f}M"
    if val >= 1e3:
        return f"{val/1e3:.1f}k"
    return f"{val:.0f}"

# -------------------------
# Map Plot Function
# -------------------------
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
        df = df.rename(columns={"Source_Countries":"source","Target_Countries":"target"})
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
        df = df.rename(columns={"Source_Countries":"source","Target_Countries":"target"})
        title = f"{country}: {crop} ({category}) / {orig_crop} — Imports ({year})"

    if df.empty:
        st.warning("⚠️ No trade data found for this selection.")
        return

    df = df.groupby(["source", "target"], as_index=False)["value_kg_n"].sum()
    df = df.rename(columns={"value_kg_n": "weight"})
    df = df[df["weight"] > 0]

    # Canonical map names; drop unmatched
    df["source_map"] = df["source"].map(_canonical_country)
    df["target_map"] = df["target"].map(_canonical_country)
    df = df.dropna(subset=["source_map", "target_map"])
    if df.empty:
        st.warning("⚠️ Countries not found in basemap for these flows.")
        return

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

    for _, row in df.iterrows():
        src = country_coords.get(row["source_map"])
        tgt = country_coords.get(row["target_map"])
        if src is None or tgt is None:
            continue
        arc_x, arc_y = great_circle_points(src, tgt, npoints=50)
        color = cmap(norm(row["log_weight"]))
        lw = 0.5 + 2.5 * (row["log_weight"] / df["log_weight"].max())
        ax.plot(arc_x, arc_y, transform=ccrs.Geodetic(), color=color, linewidth=lw, alpha=0.8)
        idx = int(len(arc_x) * 0.7)  # arrow
        ax.annotate("", xy=(arc_x[idx], arc_y[idx]), xycoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    xytext=(arc_x[idx-1], arc_y[idx-1]), textcoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical", shrink=0.5, ax=ax)
    cbar.set_label("Log-scaled Trade Flow (kg N)")
    plt.title(title, fontsize=13)
    st.pyplot(fig)

# -------------------------
# Sankey Plot Function
# -------------------------
def plot_sankey(df, country, year, crop, orig_crop, category, threshold, flow_type="export", palette_choice="Set3"):
    cat_lc = str(category).lower()
    crop_lc = str(crop).lower()
    country_lc = str(country).lower()

    if flow_type == "export":
        df_filtered = df[
            (df["Year"] == year) &
            (df["cat_lc"] == cat_lc) &
            (df["item_lc"] == crop_lc) &
            (df["src_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]].copy()
    else:
        df_filtered = df[
            (df["Year"] == year) &
            (df["cat_lc"] == cat_lc) &
            (df["item_lc"] == crop_lc) &
            (df["tgt_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]].copy()

    if df_filtered.empty:
        st.warning(f"⚠️ No {'exports' if flow_type=='export' else 'imports'} found for {country} in {year}.")
        return

    df_filtered = df_filtered.groupby(["Source_Countries", "Target_Countries"], as_index=False).sum()
    df_filtered = df_filtered.rename(columns={
        "Source_Countries": "source",
        "Target_Countries": "target",
        "value_kg_n": "weight_n"
    })

    total_trade = df_filtered["weight_n"].sum()
    keep_mask = df_filtered["weight_n"] > threshold * total_trade
    if keep_mask.sum() < 10 and len(df_filtered) > 10:
        df_filtered = df_filtered.nlargest(10, "weight_n")
    else:
        df_filtered = df_filtered[keep_mask]

    exporters = df_filtered["source"].unique().tolist()
    importers = df_filtered["target"].unique().tolist()
    all_nodes = exporters + importers
    node_map = {node: i for i, node in enumerate(all_nodes)}

    palettes = {
        "Set3": pc.qualitative.Set3,
        "Dark2": pc.qualitative.Dark2,
        "Pastel1": pc.qualitative.Pastel1,
    }
    selected_palette = palettes.get(palette_choice, pc.qualitative.Set3)

    if flow_type == "export":
        target_colors = {imp: selected_palette[i % len(selected_palette)] for i, imp in enumerate(importers)}
        node_colors = ["#636efa" if node in exporters else target_colors[node] for node in all_nodes]
        link_colors = [target_colors[t] for t in df_filtered["target"]]
    else:
        source_colors = {src: selected_palette[i % len(selected_palette)] for i, src in enumerate(exporters)}
        node_colors = [source_colors[node] if node in exporters else "#0ed6c1" for node in all_nodes]
        link_colors = [source_colors[s] for s in df_filtered["source"]]

    sources = df_filtered["source"].map(node_map)
    targets = df_filtered["target"].map(node_map)
    values = df_filtered["weight_n"]

    sankey_fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        orientation="h",
        node=dict(
            pad=24, thickness=20, line=dict(color="#333", width=1.7),
            label=[f"{node}" for node in all_nodes],
            color=node_colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate=(
                '<b>Exporter:</b> %{source.label}<br>' +
                '<b>Importer:</b> %{target.label}<br>' +
                '<b>Value:</b> %{value:,.0f} kg N<br>' +
                f'<b>Year:</b> {year}<br>' +
                f'<b>Crop:</b> {crop} / {orig_crop}<br>' +
                f'<b>Category:</b> {category}<extra></extra>'
            )
        )
    )])

    sankey_fig.update_layout(
        title_text=f"📊 {country}: {crop} ({category}) / {orig_crop} — {'Exports' if flow_type=='export' else 'Imports'} {year}",
        font=dict(family="Open Sans, Arial", size=14, color="#222"),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        margin=dict(l=20, r=80, t=60, b=40), height=600
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

# -------------------------
# Sidebar filters
# -------------------------
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

# Show standardized and original together for clarity
crop_pairs = (trade_df[["Item", "original_Item"]]
              .drop_duplicates()
              .sort_values(["Item", "original_Item"]))
crop_label_list = [f"{r.Item} → {r.original_Item}" for _, r in crop_pairs.iterrows()]
crop_label = st.sidebar.selectbox("Select Crop (standardized → original)", crop_label_list)
item_selected, original_selected = crop_label.split(" → ", 1)

# Source country search
country_search = st.sidebar.text_input("Search Source Country (case-insensitive)", "India")
src_unique = trade_df["Source_Countries"].dropna().unique()
countries_lower = [c.lower() for c in src_unique]
if country_search.strip().lower() in countries_lower:
    source_selected = src_unique[countries_lower.index(country_search.strip().lower())]
else:
    source_selected = "India"

# Importing partner country (or all)
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
        This interactive dashboard visualizes nitrogen trade flows for **major crops** from your reduced dataset.
        - Filter by year, category, standardized crop and its original label.
        - Sankey shows top flows (threshold can trim small links).
        - Maps draw great-circle arcs; hover for details.
        """
    )

# -------------------------
# Main page visualization
# -------------------------
st.title("🌍 Virtual N Global Trade Flow Explorer (Major Crops)")

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
