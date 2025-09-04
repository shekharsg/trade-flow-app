import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pyproj import Geod
import numpy as np
import streamlit as st

# -------------------------
# Helper function: great-circle route
# -------------------------
geod = Geod(ellps="WGS84")

@st.cache_data
def load_data():
    trade_df = pd.read_csv(
        "data/exp_import_max_value_n_ton.csv"  # keep data in a "data" folder
    )
    return trade_df

trade_df = load_data()

# -------------------------
# Prepare world map & centroids
# -------------------------
world = gpd.read_file("shapefiles/ne_110m_admin_0_countries.shp")
world_proj = world.to_crs(epsg=3857)
world["coords"] = world_proj.centroid.to_crs(world.crs)
country_coords = world.set_index("name")["coords"].to_dict()

def great_circle_points(src, tgt, npoints=50):
    points = geod.npts(src.x, src.y, tgt.x, tgt.y, npoints)
    lons = [src.x] + [p[0] for p in points] + [tgt.x]
    lats = [src.y] + [p[1] for p in points] + [tgt.y]
    return lons, lats

# -------------------------
# Plotting function
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
        st.warning("?? No trade data found for this selection.")
        return

    df = df[df["weight"] > 0]
    df["log_weight"] = np.log10(df["weight"] + 1)

    norm = plt.Normalize(vmin=df["log_weight"].min(), vmax=df["log_weight"].max())
    cmap = plt.colormaps.get_cmap("Spectral_r")

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
    cbar.set_label("Log-scaled Trade Flow (kg N)")

    plt.title(f"{source_selected}: {crop_selected} ({category_selected}) Exports ({year_selected})", fontsize=14)

    st.pyplot(fig)

# -------------------------
# Sidebar widgets
# -------------------------
st.sidebar.header("Filters")
year_selected = st.sidebar.selectbox("Year", sorted(trade_df["Year"].unique()), index=0)
category_selected = st.sidebar.selectbox("Category", sorted(trade_df["Category"].dropna().unique()))
crop_selected = st.sidebar.selectbox("Crop", sorted(trade_df["Item"].unique()))
source_selected = st.sidebar.selectbox("Exporting Country", sorted(trade_df["Source_Countries"].unique()), index=0)
target_selected = st.sidebar.selectbox("Importing Country", ["All countries"] + sorted(trade_df["Target_Countries"].unique()))

st.header("?? Global Trade Flow Explorer")
plot_trade_flow(year_selected, category_selected, crop_selected, source_selected, target_selected)

