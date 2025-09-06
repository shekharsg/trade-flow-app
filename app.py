# app.py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc

# =========================
# Configurable file paths
# =========================
DATA_FILE = "data/trade_major_crops.csv.xz"   # change if needed
WORLD_SHP  = "shapefiles/ne_110m_admin_0_countries.shp"

# =========================
# Styling (professional)
# =========================
st.set_page_config(
    page_title="Virtual N Trade Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* widen content a bit */
.block-container {padding-top: 1.2rem; max-width: 1400px;}
/* subtle card-like headers */
h1, h2, h3 { font-weight: 700; letter-spacing: .2px; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #eee;
  border-radius: 12px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.04); }
.sidebar .sidebar-content { background: #fafafa; }
hr { border: none; height: 1px; background: #eee; margin: .8rem 0 1.2rem; }
.dataframe tbody tr:hover { background: #f7f9fc; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Helpers
# =========================
geod = Geod(ellps="WGS84")

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    elif ext in [".gz", ".bz2", ".xz"]:
        return pd.read_csv(path, compression="infer")
    elif ext in [".csv"]:
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, compression="infer")

@st.cache_data(show_spinner=False)
def load_data(path=DATA_FILE) -> pd.DataFrame:
    df = _read_any(path)
    needed = ["Year","Category","Item","Source_Countries","Target_Countries","value_kg_n"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # dtypes
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64").astype(int)
    df["value_kg_n"] = pd.to_numeric(df["value_kg_n"], errors="coerce").fillna(0.0).astype(float)

    # lower-case helpers
    df["cat_lc"] = df["Category"].astype(str).str.strip().str.lower()
    df["item_lc"] = df["Item"].astype(str).str.strip().str.lower()
    df["src_lc"]  = df["Source_Countries"].astype(str).str.strip().str.lower()
    df["tgt_lc"]  = df["Target_Countries"].astype(str).str.strip().str.lower()

    # optional original_Item
    if "original_Item" in df.columns:
        df["original_Item"] = df["original_Item"].astype(str)
        df["orig_lc"] = df["original_Item"].str.strip().str.lower()
    return df

trade_df = load_data()

# Basemap + centroids
world = gpd.read_file(WORLD_SHP)
if "ADMIN" not in world.columns:
    world = world.rename(columns={"NAME": "ADMIN"})
world = world[["ADMIN", "geometry"]].copy()
world_proj = world.to_crs(epsg=3857)
world["coords"] = world_proj.centroid.to_crs(world.crs)
country_coords = world.set_index("ADMIN")["coords"].to_dict()
world_names_lc = {k.lower(): k for k in country_coords.keys()}

def _canonical_country(name: str):
    if not isinstance(name, str): return None
    return world_names_lc.get(name.strip().lower())

def great_circle_points(src, tgt, npoints=50):
    points = geod.npts(src.x, src.y, tgt.x, tgt.y, npoints)
    lons = [src.x] + [p[0] for p in points] + [tgt.x]
    lats = [src.y] + [p[1] for p in points] + [tgt.y]
    return lons, lats

def short_fmt(val: float) -> str:
    if val >= 1e9: return f"{val/1e9:.2f}B"
    if val >= 1e6: return f"{val/1e6:.2f}M"
    if val >= 1e3: return f"{val/1e3:.1f}k"
    return f"{val:.0f}"

# =========================
# Plot: map arcs
# =========================
def plot_trade_flow(year, category, crop, country, flow_type="export", partner="All countries"):
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
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
        if partner_lc is not None:
            df = df[trade_df.loc[df.index, "tgt_lc"] == partner_lc]
        df = df.rename(columns={"Source_Countries":"source","Target_Countries":"target"})
        title = f"{country}: {crop} ({category}) – Exports {year}"
    else:
        df = trade_df[
            (trade_df["Year"] == year) &
            (trade_df["cat_lc"] == cat_lc) &
            (trade_df["item_lc"] == crop_lc) &
            (trade_df["tgt_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
        if partner_lc is not None:
            df = df[trade_df.loc[df.index, "src_lc"] == partner_lc]
        df = df.rename(columns={"Source_Countries":"source","Target_Countries":"target"})
        title = f"{country}: {crop} ({category}) – Imports {year}"

    if df.empty:
        st.info("No trade data found for this selection.")
        return

    df = df.groupby(["source", "target"], as_index=False)["value_kg_n"].sum()
    df = df.rename(columns={"value_kg_n": "weight"})
    df = df[df["weight"] > 0]

    df["source_map"] = df["source"].map(_canonical_country)
    df["target_map"] = df["target"].map(_canonical_country)
    df = df.dropna(subset=["source_map","target_map"])
    if df.empty:
        st.info("Countries not found in basemap for these flows.")
        return

    df["log_weight"] = np.log10(df["weight"] + 1)
    norm = plt.Normalize(vmin=df["log_weight"].min(), vmax=df["log_weight"].max())
    cmap = plt.colormaps.get_cmap("Spectral_r")

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.4)
    ax.add_feature(cfeature.LAND, facecolor="#efefef")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    for _, row in df.iterrows():
        src = country_coords.get(row["source_map"])
        tgt = country_coords.get(row["target_map"])
        if src is None or tgt is None:
            continue
        arc_x, arc_y = great_circle_points(src, tgt, npoints=50)
        color = cmap(norm(row["log_weight"]))
        lw = 0.5 + 2.6 * (row["log_weight"] / df["log_weight"].max())
        ax.plot(arc_x, arc_y, transform=ccrs.Geodetic(), color=color, linewidth=lw, alpha=0.85)
        idx = int(len(arc_x)*0.7)
        ax.annotate("", xy=(arc_x[idx], arc_y[idx]), xycoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    xytext=(arc_x[idx-1], arc_y[idx-1]), textcoords=ccrs.Geodetic()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical", shrink=0.55, ax=ax)
    cbar.set_label("Log-scaled Trade Flow (kg N)")
    plt.title(title, fontsize=14, weight=700)
    st.pyplot(fig, use_container_width=True)

# =========================
# Plot: Sankey
# =========================
def plot_sankey(df, country, year, crop, category, threshold, flow_type="export", palette_choice="Set3"):
    cat_lc = str(category).lower()
    crop_lc = str(crop).lower()
    country_lc = str(country).lower()

    if flow_type == "export":
        df_filtered = df[
            (df["Year"] == year) & (df["cat_lc"] == cat_lc) &
            (df["item_lc"] == crop_lc) & (df["src_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
        title_dir = "Exports"
    else:
        df_filtered = df[
            (df["Year"] == year) & (df["cat_lc"] == cat_lc) &
            (df["item_lc"] == crop_lc) & (df["tgt_lc"] == country_lc)
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
        title_dir = "Imports"

    if df_filtered.empty:
        st.info(f"No {title_dir.lower()} found for {country} in {year}.")
        return

    df_filtered = (df_filtered
                   .groupby(["Source_Countries","Target_Countries"], as_index=False)
                   .sum()
                   .rename(columns={"Source_Countries":"source",
                                    "Target_Countries":"target",
                                    "value_kg_n":"weight_n"}))

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

    palettes = {"Set3": pc.qualitative.Set3,
                "Dark2": pc.qualitative.Dark2,
                "Pastel1": pc.qualitative.Pastel1}
    selected_palette = palettes.get(palette_choice, pc.qualitative.Set3)

    if flow_type == "export":
        target_colors = {imp: selected_palette[i % len(selected_palette)] for i, imp in enumerate(importers)}
        node_colors = ["#636efa" if n in exporters else target_colors[n] for n in all_nodes]
        link_colors = [target_colors[t] for t in df_filtered["target"]]
    else:
        source_colors = {src: selected_palette[i % len(selected_palette)] for i, src in enumerate(exporters)}
        node_colors = [source_colors[n] if n in exporters else "#0ed6c1" for n in all_nodes]
        link_colors = [source_colors[s] for s in df_filtered["source"]]

    sources = df_filtered["source"].map(node_map)
    targets = df_filtered["target"].map(node_map)
    values  = df_filtered["weight_n"]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap", orientation="h",
        node=dict(
            pad=24, thickness=20, line=dict(color="#333", width=1.2),
            label=[f"{n}" for n in all_nodes],
            color=node_colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate=('<b>Exporter:</b> %{source.label}<br>'
                           '<b>Importer:</b> %{target.label}<br>'
                           '<b>Value:</b> %{value:,.0f} kg N<extra></extra>')
        )
    )])

    fig.update_layout(
        title_text=f"{country}: {crop} ({category}) – {title_dir} Sankey, {year}",
        font=dict(size=14), plot_bgcolor="#fff", paper_bgcolor="#fff",
        margin=dict(l=10, r=60, t=60, b=20), height=560
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Controls")
year_selected = st.sidebar.slider(
    "Year", int(trade_df["Year"].min()), int(trade_df["Year"].max()),
    int(trade_df["Year"].max()), step=1
)
category_selected = st.sidebar.selectbox("Category",
    sorted(trade_df["Category"].dropna().unique())
)
# If you have original_Item, allow browsing pairs first
if "original_Item" in trade_df.columns:
    with st.sidebar.expander("Browse by original_Item"):
        orig_choice = st.selectbox(
            "original_Item (optional filter)",
            ["(All)"] + sorted(trade_df["original_Item"].dropna().unique())
        )
else:
    orig_choice = "(All)"

# Crop list possibly filtered by original_Item choice
if orig_choice != "(All)" and "orig_lc" in trade_df.columns:
    crop_pool = trade_df.loc[trade_df["orig_lc"]==orig_choice.strip().lower(),"Item"].dropna().unique()
else:
    crop_pool = trade_df["Item"].dropna().unique()

crop_selected = st.sidebar.selectbox("Crop", sorted(crop_pool))

# Exporter search with case-insensitive match
country_search = st.sidebar.text_input("Exporter (type to search)", "India")
src_unique = trade_df["Source_Countries"].dropna().unique()
src_lower = [c.lower() for c in src_unique]
source_selected = src_unique[src_lower.index(country_search.strip().lower())] if country_search.strip().lower() in src_lower else "India"

target_selected = st.sidebar.selectbox(
    "Importer filter (for maps)", ["All countries"] + sorted(trade_df["Target_Countries"].dropna().unique())
)

threshold = st.sidebar.slider("Min share (Sankey link filter)", 0.0, 0.1, 0.01, 0.005)
pal_choice = st.sidebar.radio("Sankey palette", ("Set3","Dark2","Pastel1"), index=0)

st.sidebar.markdown("---")
with st.sidebar.expander("Download current selection"):
    # make a small export of current selection (exports)
    sel = trade_df[
        (trade_df["Year"]==year_selected) &
        (trade_df["cat_lc"]==str(category_selected).lower()) &
        (trade_df["item_lc"]==str(crop_selected).lower()) &
        (trade_df["src_lc"]==str(source_selected).lower())
    ][["Year","Category","Item","Source_Countries","Target_Countries","value_kg_n"]].copy()
    st.download_button(
        "Download CSV (current exports)", data=sel.to_csv(index=False).encode(),
        file_name=f"selection_exports_{year_selected}.csv", mime="text/csv"
    )

# =========================
# Header + KPIs
# =========================
st.title("🌍 Virtual Nitrogen Trade Explorer — Major Crops")

colA, colB, colC, colD = st.columns(4)
exports_total = trade_df[
    (trade_df["Year"]==year_selected) &
    (trade_df["cat_lc"]==str(category_selected).lower()) &
    (trade_df["item_lc"]==str(crop_selected).lower()) &
    (trade_df["src_lc"]==str(source_selected).lower())
]["value_kg_n"].sum()

imports_total = trade_df[
    (trade_df["Year"]==year_selected) &
    (trade_df["cat_lc"]==str(category_selected).lower()) &
    (trade_df["item_lc"]==str(crop_selected).lower()) &
    (trade_df["tgt_lc"]==str(source_selected).lower())
]["value_kg_n"].sum()

colA.metric("Year", f"{year_selected}")
colB.metric("Crop", crop_selected)
colC.metric("Exports (kg N)", short_fmt(exports_total))
colD.metric("Imports (kg N)", short_fmt(imports_total))
st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# Tabs: Exports / Imports / Pairs
# =========================
tab1, tab2, tab3 = st.tabs(["📤 Exports", "📥 Imports", "🔗 Item pairs"])

with tab1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Export Flows Map")
        plot_trade_flow(year_selected, category_selected, crop_selected, source_selected,
                        flow_type="export", partner=target_selected)
    with c2:
        st.subheader("Export Flows Sankey")
        plot_sankey(trade_df, source_selected, year_selected, crop_selected,
                    category_selected, threshold, flow_type="export", palette_choice=pal_choice)

with tab2:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Import Flows Map")
        # if "All countries" is selected, show imports of the exporter country itself
        plot_trade_flow(year_selected, category_selected, crop_selected,
                        target_selected if target_selected != "All countries" else source_selected,
                        flow_type="import")
    with c2:
        st.subheader("Import Flows Sankey")
        plot_sankey(trade_df, source_selected, year_selected, crop_selected,
                    category_selected, threshold, flow_type="import", palette_choice=pal_choice)

with tab3:
    st.subheader("Item ↔ original_Item pairs")
    if "original_Item" in trade_df.columns:
        pairs = (trade_df[["Item","original_Item"]]
                 .drop_duplicates()
                 .sort_values(["Item","original_Item"])
                 .reset_index(drop=True))
        st.caption("Unique pairs found in the dataset.")
        st.dataframe(pairs, use_container_width=True, height=420)
        st.download_button(
            "Download pairs as CSV",
            data=pairs.to_csv(index=False).encode(),
            file_name="item_original_pairs.csv",
            mime="text/csv"
        )
    else:
        st.info("The column **original_Item** is not present in this dataset, so pairs cannot be displayed.")
