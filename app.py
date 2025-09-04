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

@st.cache_data
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
# Short format
# -------------------------
def short_fmt(val):
    if val >= 1e6:
        return f"{val/1e6:.1f}M"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k"
    else:
        return f"{val:.0f}"

# -------------------------
# Map Plot
# -------------------------
def plot_trade_flow(year_selected, category_selected, crop_selected, source_selected, target_selected):
    df = trade_df[
        (trade_df["Year"] == year_selected) &
        (trade_df["Category"].str.lower() == category_selected.lower()) &
        (trade_df["Item"].str.lower() == crop_selected.lower()) &
        (trade_df["Source_Countries"].str.lower() == source_selected.lower())
    ][["Source_Countries", "Target_Countries", "value_kg_n"]]

    if target_selected != "All countries":
        df = df[df["Target_Countries"].str.lower() == target_selected.lower()]

    df = df.groupby(["Source_Countries", "Target_Countries"], as_index=False)["value_kg_n"].sum()
    df = df.rename(columns={"Source_Countries": "source", "Target_Countries": "target", "value_kg_n": "weight"})

    if df.empty:
        st.warning("⚠️ No trade data found for this selection.")
        return

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

            ax.plot(arc_x, arc_y, transform=ccrs.Geodetic(),
                    color=color, linewidth=lw, alpha=0.8)

            idx = int(len(arc_x) * 0.7)
            ax.annotate("",
                        xy=(arc_x[idx], arc_y[idx]),
                        xycoords=ccrs.Geodetic()._as_mpl_transform(ax),
                        xytext=(arc_x[idx-1], arc_y[idx-1]),
                        textcoords=ccrs.Geodetic()._as_mpl_transform(ax),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical", shrink=0.5, ax=ax)
    cbar.set_label("Log-scaled Trade Flow (kg N)")

    plt.title(f"{source_selected}: {crop_selected} ({category_selected}) Exports ({year_selected})", fontsize=13)
    st.pyplot(fig)

# -------------------------
# Sankey Export
# -------------------------
def plot_export_sankey(df, source_country, year, crop, category, threshold=0.01):
    df_filtered = df[
        (df["Year"] == year) &
        (df["Category"].str.lower() == category.lower()) &
        (df["Item"].str.lower() == crop.lower()) &
        (df["Source_Countries"].str.lower() == source_country.lower())
    ][["Source_Countries", "Target_Countries", "value_kg_n"]]

    if df_filtered.empty:
        st.warning(f"⚠️ No exports found for {source_country} in {year}.")
        return

    df_filtered = df_filtered.groupby(["Source_Countries", "Target_Countries"], as_index=False).sum()
    df_filtered = df_filtered.rename(columns={"Source_Countries": "source",
                                              "Target_Countries": "target",
                                              "value_kg_n": "weight_n"})
    total_trade = df_filtered["weight_n"].sum()
    df_filtered = df_filtered[df_filtered["weight_n"] > threshold * total_trade]

    exporters = df_filtered["source"].unique().tolist()
    importers = df_filtered["target"].unique().tolist()
    all_nodes = exporters + importers
    node_map = {node: i for i, node in enumerate(all_nodes)}

    palette = pc.qualitative.Set3 + pc.qualitative.Set2 + pc.qualitative.Pastel1
    target_colors = {imp: palette[i % len(palette)] for i, imp in enumerate(importers)}
    node_colors = ["#cccccc" if node in exporters else target_colors[node] for node in all_nodes]
    link_colors = [target_colors[t] for t in df_filtered["target"]]

    node_x = [0.25 if node in exporters else 0.55 for node in all_nodes]
    node_y = np.linspace(0.05, 0.95, len(all_nodes))

    sources = df_filtered["source"].map(node_map)
    targets = df_filtered["target"].map(node_map)
    values = df_filtered["weight_n"]

    sankey_fig = go.Figure(data=[go.Sankey(
        arrangement="snap", orientation="h",
        node=dict(pad=12, thickness=14, line=dict(color="black", width=0.8),
                  label=[""] * len(all_nodes), color=node_colors, x=node_x, y=node_y),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])

    exporter_total = df_filtered["weight_n"].sum()
    importer_totals = df_filtered.groupby("target")["weight_n"].sum().to_dict()

    for i, node in enumerate(all_nodes):
        x_pos, y_pos = node_x[i], node_y[i]
        if node in exporters:
            offset, align = 6, "left"
            text = f"{node} ({short_fmt(exporter_total)} kg N)"
        else:
            offset, align = -6, "right"
            text = f"{node} ({short_fmt(importer_totals.get(node, 0))} kg N)"
        sankey_fig.add_annotation(
            x=x_pos, y=y_pos, xshift=offset, text=text,
            showarrow=False, font=dict(size=9, color="black"),
            align=align, xanchor=align, yanchor="middle"
        )

    sankey_fig.update_layout(
        title_text=f"📤 {source_country}: {crop} ({category}) Exports Sankey ({year})",
        font=dict(size=10, color="black"),
        margin=dict(l=20, r=20, t=40, b=20), height=500
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

# -------------------------
# Sankey Import
# -------------------------
def plot_import_sankey(df, target_country, year, crop, category, threshold=0.01):
    df_filtered = df[
        (df["Year"] == year) &
        (df["Category"].str.lower() == category.lower()) &
        (df["Item"].str.lower() == crop.lower()) &
        (df["Target_Countries"].str.lower() == target_country.lower())
    ][["Source_Countries", "Target_Countries", "value_kg_n"]]

    if df_filtered.empty:
        st.warning(f"⚠️ No imports found for {target_country} in {year}.")
        return

    df_filtered = df_filtered.groupby(["Source_Countries", "Target_Countries"], as_index=False).sum()
    df_filtered = df_filtered.rename(columns={"Source_Countries": "source",
                                              "Target_Countries": "target",
                                              "value_kg_n": "weight_n"})
    total_trade = df_filtered["weight_n"].sum()
    df_filtered = df_filtered[df_filtered["weight_n"] > threshold * total_trade]

    exporters = df_filtered["source"].unique().tolist()
    importers = df_filtered["target"].unique().tolist()
    all_nodes = exporters + importers
    node_map = {node: i for i, node in enumerate(all_nodes)}

    palette = pc.qualitative.Set3 + pc.qualitative.Set2 + pc.qualitative.Pastel1
    source_colors = {src: palette[i % len(palette)] for i, src in enumerate(exporters)}
    node_colors = [source_colors[node] if node in exporters else "#cccccc" for node in all_nodes]
    link_colors = [source_colors[s] for s in df_filtered["source"]]

    node_x = [0.25 if node in exporters else 0.55 for node in all_nodes]
    node_y = np.linspace(0.05, 0.95, len(all_nodes))

    sources = df_filtered["source"].map(node_map)
    targets = df_filtered["target"].map(node_map)
    values = df_filtered["weight_n"]

    sankey_fig = go.Figure(data=[go.Sankey(
        arrangement="snap", orientation="h",
        node=dict(pad=12, thickness=14, line=dict(color="black", width=0.8),
                  label=[""] * len(all_nodes), color=node_colors, x=node_x, y=node_y),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])

    importer_total = df_filtered["weight_n"].sum()
    exporter_totals = df_filtered.groupby("source")["weight_n"].sum().to_dict()

    for i, node in enumerate(all_nodes):
        x_pos, y_pos = node_x[i], node_y[i]
        if node in exporters:
            offset, align = 6, "left"
            text = f"{node} ({short_fmt(exporter_totals.get(node, 0))} kg N)"
        else:
            offset, align = -6, "right"
            text = f"{node} ({short_fmt(importer_total)} kg N)"
        sankey_fig.add_annotation(
            x=x_pos, y=y_pos, xshift=offset, text=text,
            showarrow=False, font=dict(size=9, color="black"),
            align=align, xanchor=align, yanchor="middle"
        )

    sankey_fig.update_layout(
        title_text=f"📥 {target_country}: {crop} ({category}) Imports Sankey ({year})",
        font=dict(size=10, color="black"),
        margin=dict(l=20, r=20, t=40, b=20), height=500
    )
    st.plotly_chart(sankey_fig, use_container_width=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Filters")
year_selected = st.sidebar.selectbox("Year", sorted(trade_df["Year"].unique()), index=0)
category_selected = st.sidebar.selectbox("Category", sorted(trade_df["Category"].dropna().unique()))
crop_selected = st.sidebar.selectbox("Crop", sorted(trade_df["Item"].unique()))

countries_sorted = sorted(trade_df["Source_Countries"].unique())
default_index = countries_sorted.index("India") if "India" in countries_sorted else 0
source_selected = st.sidebar.selectbox("Country", countries_sorted, index=default_index)

target_selected = st.sidebar.selectbox("Importing Country", ["All countries"] + sorted(trade_df["Target_Countries"].unique()))

# -------------------------
# Main Page
# -------------------------
st.header("🌍 Virtual N Global Trade Flow")

plot_trade_flow(year_selected, category_selected, crop_selected, source_selected, target_selected)

st.subheader("📤 Export Flows")
plot_export_sankey(trade_df, source_selected, year_selected, crop_selected, category_selected)

st.subheader("📥 Import Flows")
plot_import_sankey(trade_df, source_selected, year_selected, crop_selected, category_selected)
