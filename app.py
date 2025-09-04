import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pyproj import Geod
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -------------------------
# Helper function: great-circle route
# -------------------------
geod = Geod(ellps="WGS84")

@st.cache_data
def load_data():
    trade_df = pd.read_csv("data/exp_import_max_value_n_ton.csv")
    return trade_df

trade_df = load_data()

# -------------------------
# Prepare world map & centroids
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
# Map Plotting Function
# -------------------------
def plot_trade_flow(year_selected, category_selected, crop_selected, source_selected, target_selected):
    df = trade_df[
        (trade_df["Year"] == year_selected) &
        (trade_df["Category"].str.lower() == category_selected.lower()) &
        (trade_df["Item"].str.lower() == crop_selected.lower()) &
        (trade_df["Source_Countries"].str.lower() == source_selected.lower())
    ][["Source_Countries", "Target_Countries", "value_kg_n", "value_tonne_raw"]]

    if target_selected != "All countries":
        df = df[df["Target_Countries"].str.lower() == target_selected.lower()]

    df = df.groupby(["Source_Countries", "Target_Countries"], as_index=False).sum()

    df = df.rename(columns={
        "Source_Countries": "source",
        "Target_Countries": "target",
        "value_kg_n": "weight_n",
        "value_tonne_raw": "weight_raw"
    })

    if df.empty:
        st.warning("⚠️ No trade data found for this selection.")
        return None

    # Remove zeros
    df = df[(df["weight_n"] > 0) & (df["weight_raw"] > 0)]
    df["log_weight"] = np.log10(df["weight_n"] + 1)

    # Totals
    total_raw_kg = df["weight_raw"].sum() * 1000  # tonne → kg
    total_n_kg = df["weight_n"].sum()

    st.subheader(f"📦 Total Physical Trade Volume: {total_raw_kg:,.0f} kg")
    st.subheader(f"🧪 Total Virtual N Trade Volume: {total_n_kg:,.0f} kg N")

    # Colors
    norm = plt.Normalize(vmin=df["log_weight"].min(), vmax=df["log_weight"].max())
    cmap = plt.colormaps.get_cmap("Spectral_r")

    # Plot map
    fig = plt.figure(figsize=(14, 7))
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
    cbar.set_label("Log-scaled Virtual N Flow (kg N)")

    plt.title(f"{source_selected}: {crop_selected} ({category_selected}) Exports ({year_selected})", fontsize=14)
    st.pyplot(fig)

    return df, total_raw_kg, total_n_kg

# -------------------------
# Sankey Diagram Function
# -------------------------
import plotly.colors as pc

def plot_sankey(df, source_country, year, crop, category, threshold=0.01):
    if df.empty:
        st.warning("⚠️ No data for Sankey diagram.")
        return

    # Sort and filter
    df_sorted = df.sort_values(by="weight_n", ascending=False)
    total_trade = df_sorted["weight_n"].sum()
    df_filtered = df_sorted[df_sorted["weight_n"] > threshold * total_trade]

    if df_filtered.empty:
        st.warning(f"⚠️ No importers above {threshold*100:.1f}% of trade.")
        return

    # Nodes
    exporters = df_filtered["source"].unique().tolist()
    importers = df_filtered["target"].unique().tolist()
    all_nodes = exporters + importers
    node_map = {node: i for i, node in enumerate(all_nodes)}

    # Colors
    palette = pc.qualitative.Set3 + pc.qualitative.Set2 + pc.qualitative.Pastel1
    target_colors = {imp: palette[i % len(palette)] for i, imp in enumerate(importers)}

    node_colors = ["lightgray" if node in exporters else target_colors[node] for node in all_nodes]

    sources = df_filtered["source"].map(node_map)
    targets = df_filtered["target"].map(node_map)
    values = df_filtered["weight_n"]

    # Link colors = target node colors
    link_colors = [target_colors[t] for t in df_filtered["target"]]

    # Node positions
    node_x = [0.0 if node in exporters else 1.0 for node in all_nodes]
    node_y = np.linspace(0, 1, len(all_nodes))

    # Sankey diagram (hide default labels)
    sankey_fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        orientation="h",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=1.2),
            label=[""] * len(all_nodes),
            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate="%{source.label} → %{target.label}<br>%{value:,.0f} kg N<extra></extra>"
        )
    )])

    # --- Add outside labels with volumes (short format) ---
    exporter_total = df_filtered["weight_n"].sum()
    importer_totals = df_filtered.groupby("target")["weight_n"].sum().to_dict()

    def short_fmt(val):
        if val >= 1e6:
            return f"{val/1e6:.1f}M"
        elif val >= 1e3:
            return f"{val/1e3:.1f}k"
        else:
            return f"{val:.0f}"

    for i, node in enumerate(all_nodes):
        x_pos = node_x[i]
        y_pos = node_y[i]
        align = "right" if x_pos == 0.0 else "left"
        offset = -40 if x_pos == 0.0 else 40

        if node in exporters:
            text = f"{node} ({short_fmt(exporter_total)} kg N)"
        else:
            val = importer_totals.get(node, 0)
            text = f"{node} ({short_fmt(val)} kg N)"

        sankey_fig.add_annotation(
            x=x_pos, y=y_pos,
            xshift=offset,
            text=text,
            showarrow=False,
            font=dict(size=12, color="black"),
            align=align,
            xanchor=align,
            yanchor="middle"
        )

    sankey_fig.update_layout(
        title_text=f"📊 {source_country}: {crop} ({category}) Exports Sankey ({year}) (Importers >{threshold*100:.0f}% only)",
        font=dict(size=12, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=60, b=60)
    )

    st.plotly_chart(sankey_fig, use_container_width=True)





# -------------------------
# Sidebar Widgets
# -------------------------
st.sidebar.header("Filters")
year_selected = st.sidebar.selectbox("Year", sorted(trade_df["Year"].unique()), index=0)
category_selected = st.sidebar.selectbox("Category", sorted(trade_df["Category"].dropna().unique()))
crop_selected = st.sidebar.selectbox("Crop", sorted(trade_df["Item"].unique()))
# Exporting country dropdown (default = India if available)
countries_sorted = sorted(trade_df["Source_Countries"].unique())
default_index = countries_sorted.index("India") if "India" in countries_sorted else 0

source_selected = st.sidebar.selectbox("Exporting Country", countries_sorted, index=default_index)
target_selected = st.sidebar.selectbox("Importing Country", ["All countries"] + sorted(trade_df["Target_Countries"].unique()))

# -------------------------
# Main Title
# -------------------------
st.title("🌍 Virtual N Global Trade Flow")

# -------------------------
# Run Visualization
# -------------------------
result = plot_trade_flow(year_selected, category_selected, crop_selected, source_selected, target_selected)

if result is not None:
    df_selection, total_raw_kg, total_n_kg = result
    st.markdown("---")
    plot_sankey(df_selection, source_selected, year_selected, crop_selected, category_selected)














