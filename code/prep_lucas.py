"""prep_lucas.py -- build a clean LUCAS 2015 topsoil table for the experiment.

Source (user-provided, read-only): LUCAS 2015 topsoil CSV/shapefile + ancillary
environmental data. We predict log soil organic carbon (OC) from ENVIRONMENTAL
covariates only (bioclim BIO1-19, elevation, slope, aspect) -- never from other
soil chemistry, to avoid a circular soil-from-soil task. Coordinates come from the
shapefile (EPSG:4326), reprojected to the equal-area EPSG:3035 (metres). Regions
for worst-region coverage are fixed k-means (k=10) clusters on the coordinates.
Output: ../results/lucas_prepared.csv (Point_ID, x, y, region, logOC, predictors).
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from sklearn.cluster import KMeans

SRC = "/sessions/eloquent-magical-cannon/mnt/flood proj"
RES = os.path.join(os.path.dirname(__file__), "..", "results")
SHP = os.path.join(SRC, "LUCAS2015_topsoildata_20200323",
                   "LUCAS_Topsoil_2015_20200323-shapefile",
                   "LUCAS_Topsoil_2015_20200323.shp")
TS = os.path.join(SRC, "LUCAS2015_topsoildata_20200323", "LUCAS_Topsoil_2015_20200323.csv")
ANC = os.path.join(SRC, "LUCAS2015_AncillaryData_20201007.csv")
BIO = [f"BIO{i}" for i in range(1, 20)]
PRED = ["Elevation", "Slope", "Aspect_sin", "Aspect_cos"] + BIO
N_REGIONS = 10


def main():
    g = gpd.read_file(SHP)[["Point_ID", "geometry"]].to_crs(3035)
    g["x"] = g.geometry.x.values
    g["y"] = g.geometry.y.values
    coords = g[["Point_ID", "x", "y"]]

    ts = pd.read_csv(TS)[["Point_ID", "OC"]]
    ts["OC"] = pd.to_numeric(ts["OC"], errors="coerce")

    anc = pd.read_csv(ANC).rename(columns={"POI": "Point_ID"})
    anc = anc[["Point_ID", "Elevation", "Slope", "Aspect"] + BIO].copy()
    # circular aspect -> sin/cos (degrees)
    a = np.deg2rad(pd.to_numeric(anc["Aspect"], errors="coerce"))
    anc["Aspect_sin"] = np.sin(a)
    anc["Aspect_cos"] = np.cos(a)

    df = coords.merge(ts, on="Point_ID").merge(anc, on="Point_ID")
    df["logOC"] = np.log(df["OC"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["logOC", "x", "y"] + PRED)

    # fixed spatial regions for worst-region coverage
    df["region"] = KMeans(N_REGIONS, n_init=10, random_state=0).fit_predict(df[["x", "y"]].to_numpy())

    out = df[["Point_ID", "x", "y", "region", "logOC"] + PRED].reset_index(drop=True)
    out.to_csv(os.path.join(RES, "lucas_prepared.csv"), index=False)
    print("rows:", len(out), "| predictors:", len(PRED), "| regions:", out.region.nunique())
    print("logOC range:", round(out.logOC.min(), 2), round(out.logOC.max(), 2),
          "| OC median (g/kg):", round(float(np.exp(out.logOC.median())), 1))
    print("x span (km):", round((out.x.max()-out.x.min())/1e3), "y span (km):", round((out.y.max()-out.y.min())/1e3))
    print("region sizes:", sorted(out.region.value_counts().tolist()))


if __name__ == "__main__":
    main()
