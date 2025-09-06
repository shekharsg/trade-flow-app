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

# -------------------------
# Great-circle setup
# -------------------------
geod = Geod(ellps="WGS84")

@st.cache_data(show_spinner=False)
def load_data():
    trade_df = pd.read_csv("data/exp_import_max_value_n_ton.csv")
    return trade_df

trade_df = load_data()

# -------------------------
# World shapefile
# -------------------------
world = gpd.read_file("shapefiles/ne_110m_admin_0_countries.shp")
world_proj = world.to_crs(epsg=3857)
world["coords"] = world_proj.centroid.to_crs(world.crs)
country_coords = world.set_index("ADMIN")["coords"].to_dict()

def great_circle_points(src, tgt, npoints=50):
    points = geod.npts(src.x, src.y, tgt.x, tgt.y, npoints)
    lons = [src.x] + [p[0] for p in points] + [tgt.x]
    lats = [src.y] + [p[1] for p in points] + [tgt.y]
    return lons, lats

# -------------------------
# Short format helper
# -------------------------
def short_fmt(val):
    if val >= 1e6:
        return f"{val/1e6:.1f}M"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k"
    else:
        return f"{val:.0f}"

# -------------------------
# Map Plot Function (exporter/importer)
# -------------------------
def plot_trade_flow(year, category, crop, country, flow_type="export", partner="All countries"):
    if flow_type == "export":
        df = trade_df[
            (trade_df["Year"] == year) &
            (trade_df["Category"].str.lower() == category.lower()) &
            (trade_df["Item"].str.lower() == crop.lower()) &
            (trade_df["Source_Countries"].str.lower() == country.lower())
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
        if partner != "All countries":
            df = df[df["Target_Countries"].str.lower() == partner.lower()]
        df = df.rename(columns={"Source_Countries": "source", "Target_Countries": "target"})
        title = f"{country}: {crop} ({category}) Exports ({year})"
    else:  # importer perspective
        df = trade_df[
            (trade_df["Year"] == year) &
            (trade_df["Category"].str.lower() == category.lower()) &
            (trade_df["Item"].str.lower() == crop.lower()) &
            (trade_df["Target_Countries"].str.lower() == country.lower())
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
        if partner != "All countries":
            df = df[df["Source_Countries"].str.lower() == partner.lower()]
        df = df.rename(columns={"Source_Countries": "source", "Target_Countries": "target"})
        title = f"{country}: {crop} ({category}) Imports ({year})"

    if df.empty:
        st.warning("⚠️ No trade data found for this selection.")
        return

    df = df.groupby(["source", "target"], as_index=False)["value_kg_n"].sum()
    df = df.rename(columns={"value_kg_n": "weight"})
    df = df[df["weight"] > 0]

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
        if row["source"] in country_coords and row["target"] in country_coords:
            src = country_coords[row["source"]]
            tgt = country_coords[row["target"]]
            arc_x, arc_y = great_circle_points(src, tgt, npoints=50)
            color = cmap(norm(row["log_weight"]))
            lw = 0.5 + 2.5 * (row["log_weight"] / df["log_weight"].max())
            ax.plot(arc_x, arc_y, transform=ccrs.Geodetic(), color=color, linewidth=lw, alpha=0.8)
            # Arrow points towards target
            idx = int(len(arc_x)*0.7)
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
# Sankey function (unchanged)
# -------------------------
def plot_sankey(df, country, year, crop, category, threshold, flow_type="export", palette_choice="Set3"):
    if flow_type == "export":
        df_filtered = df[
            (df["Year"] == year) & 
            (df["Category"].str.lower() == category.lower()) & 
            (df["Item"].str.lower() == crop.lower()) & 
            (df["Source_Countries"].str.lower() == country.lower())
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]
    else:
        df_filtered = df[
            (df["Year"] == year) & 
            (df["Category"].str.lower() == category.lower()) & 
            (df["Item"].str.lower() == crop.lower()) & 
            (df["Target_Countries"].str.lower() == country.lower())
        ][["Source_Countries", "Target_Countries", "value_kg_n"]]

    if df_filtered.empty:
        st.warning(f"⚠️ No {'exports' if flow_type=='export' else 'imports'} found for {country} in {year}.")
        return

    df_filtered = df_filtered.groupby(["Source_Countries", "Target_Countries"], as_index=False).sum()
    df_filtered = df_filtered.rename(columns={
        "Source_Countries": "source", "Target_Countries": "target", "value_kg_n": "weight_n"})

    total_trade = df_filtered["weight_n"].sum()
    df_filtered = df_filtered[df_filtered["weight_n"] > threshold * total_trade]

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
        arrangement="snap", orientation="h",
        node=dict(
            pad=24, thickness=20, line=dict(color="#333", width=1.7),
            label=[f"{node}" for node in all_nodes],
            color=node_colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate=('<b>Exporter:</b> %{source.label}<br>' +
                           '<b>Importer:</b> %{target.label}<br>' +
                           '<b>Value:</b> %{value:,.0f} kg N<br>' +
                           f'<b>Year:</b> {year}<br><b>Crop:</b> {crop}<br><b>Category:</b> {category}<extra></extra>')
        )
    )])

    sankey_fig.update_layout(
        title_text=f"📊 {country}: {crop} ({category}) {'Exports' if flow_type=='export' else 'Imports'} Sankey {year}",
        font=dict(family="Open Sans, Arial", size=14, color="#222"),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        margin=dict(l=20, r=80, t=60, b=40), height=600
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filter & Customize Your View")

year_selected = st.sidebar.slider("Select Year", min_value=int(trade_df["Year"].min()),
                                  max_value=int(trade_df["Year"].max()), value=int(trade_df["Year"].max()),
                                  step=1, format="%d")

category_selected = st.sidebar.selectbox("Select Category", sorted(trade_df["Category"].dropna().unique()))
crop_selected = st.sidebar.selectbox("Select Crop", sorted(trade_df["Item"].unique()))

country_search = st.sidebar.text_input("Search Source Country (case-insensitive)", "India")
countries_lower = [c.lower() for c in trade_df["Source_Countries"].unique()]
if country_search.strip().lower() in countries_lower:
    source_selected = trade_df["Source_Countries"].unique()[countries_lower.index(country_search.strip().lower())]
else:
    source_selected = "India"

target_selected = st.sidebar.selectbox("Select Importing Country", ["All countries"] + sorted(trade_df["Target_Countries"].unique()))

threshold = st.sidebar.slider("Minimum trade flow fraction (filter small flows)", 0.0, 0.1, 0.01, 0.005)

pal_choice = st.sidebar.radio("Choose Sankey Color Palette",
                             ("Set3", "Dark2", "Pastel1"),
                             index=0)

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About this dashboard"):
    st.markdown(
        """
        This interactive dashboard visualizes nitrogen trade flows globally.
        - Use sliders and search inputs to explore trade by year, crop, and countries.
        - Sankey diagrams disclose top export/import flows by volume.
        - Hover on flows and nodes for detailed trade info.
        """
    )

# -------------------------
# Main page visualization
# -------------------------
st.title("🌍 Virtual N Global Trade Flow Explorer")

st.header("🗺️ Export Flows Map")
plot_trade_flow(year_selected, category_selected, crop_selected, source_selected, flow_type="export", partner=target_selected)

st.header("📤 Export Flows Sankey")
plot_sankey(trade_df, source_selected, year_selected, crop_selected, category_selected, threshold, flow_type="export", palette_choice=pal_choice)

st.header("🗺️ Import Flows Map")
plot_trade_flow(year_selected, category_selected, crop_selected,
                target_selected if target_selected != "All countries" else source_selected,
                flow_type="import")

st.header("📥 Import Flows Sankey")
plot_sankey(trade_df, source_selected, year_selected, crop_selected, category_selected, threshold, flow_type="import", palette_choice=pal_choice)
