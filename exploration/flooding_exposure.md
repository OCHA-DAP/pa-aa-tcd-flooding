---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-tcd-flooding
    language: python
    name: pa-aa-tcd-flooding
---

# Flooding exposure

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

WorldPop [source](https://hub.worldpop.org/geodata/summary?id=35719)

```python
import matplotlib.pyplot as plt
import pandas as pd

from src.datasources import worldpop, codab, floodscan
from src.constants import *
```

```python
# worldpop.aggregate_worldpop_to_adm2()
```

```python
# floodscan.clip_tcd_from_glb()
```

```python
# floodscan.calculate_exposure_raster()
```

```python
# floodscan.calculate_adm2_exposures()
```

```python
adm2 = codab.load_codab()
adm2_aoi = adm2[adm2["ADM1_PCODE"].isin(ADM1_FLOOD_EXTRA_PCODES)]
adm1 = adm2.dissolve(by="ADM1_PCODE").reset_index()
adm1_aoi = adm1[adm1["ADM1_PCODE"].isin(ADM1_FLOOD_EXTRA_PCODES)]
```

```python
adm2
```

```python
pop = worldpop.load_raw_worldpop()
pop_aoi = pop.rio.clip(adm2_aoi.geometry, all_touched=True)
pop_aoi = pop_aoi.where(pop_aoi > 0)
```

```python
fs_raster = floodscan.load_raw_tcd_floodscan()
fs_raster = fs_raster.rio.write_crs(4326)
fs_aoi = fs_raster.rio.clip(adm2_aoi.geometry, all_touched=True)

fs_aoi_year = fs_aoi.groupby("time.year").max()
fs_aoi_mean = fs_aoi_year.mean(dim="year")

fs_year = fs_raster.groupby("time.year").max()
fs_mean = fs_year.mean(dim="year")
```

```python
adm2_pop = worldpop.load_adm2_worldpop()
```

```python
exposure = floodscan.load_adm2_flood_exposures()
exposure = exposure.merge(adm2_pop, on="ADM2_PCODE")
exposure["frac_exposed"] = exposure["total_exposed"] / exposure["total_pop"]
```

```python
exposure
```

```python
exposure[exposure["ADM2_PCODE"] == NDJAMENA2].plot(x="year", y="total_exposed")
```

```python

```

```python
avg_exposure = (
    exposure.groupby("ADM2_PCODE").mean().reset_index().drop(columns=["year"])
)
int_cols = ["total_exposed", "total_pop"]
avg_exposure[int_cols] = avg_exposure[int_cols].astype(int)
avg_exposure_plot = adm2.merge(avg_exposure, on="ADM2_PCODE")
avg_exposure_plot_aoi = avg_exposure_plot[
    avg_exposure_plot["ADM1_PCODE"].isin(ADM1_FLOOD_EXTRA_PCODES)
]
```

```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

for j, variable in enumerate(["total_exposed", "frac_exposed"]):
    avg_exposure_plot.plot(
        column=variable, ax=axs[j], legend=True, cmap="Purples"
    )
    # for index, row in (
    #     avg_exposure_plot_aoi.sort_values(variable).iloc[-10:].iterrows()
    # ):
    #     centroid = row["geometry"].centroid

    #     axs[j].annotate(
    #         row["ADM2_EN"],
    #         xy=(centroid.x, centroid.y),
    #         xytext=(0, 0),
    #         textcoords="offset points",
    #         ha="center",
    #         va="center",
    #     )

    adm2.boundary.plot(ax=axs[j], linewidth=0.2, color="k")
    axs[j].axis("off")


axs[0].set_title("Population totale typiquement exposée aux inondations")
axs[1].set_title(
    "Fraction de la population typiquement exposée aux inondations"
)

plt.subplots_adjust(wspace=0)
```

```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

for j, variable in enumerate(["total_exposed", "frac_exposed"]):
    avg_exposure_plot_aoi.plot(
        column=variable, ax=axs[j], legend=True, cmap="Purples"
    )

    adm2_aoi.boundary.plot(ax=axs[j], linewidth=0.2, color="k")
    axs[j].axis("off")


axs[0].set_title("Population totale typiquement exposée aux inondations")
axs[1].set_title(
    "Fraction de la population typiquement exposée aux inondations"
)

for idx, row in adm2_aoi.iterrows():
    # Calculate the centroid of each geometry
    centroid = row.geometry.centroid
    x_shift, y_shift = (
        (-0.25, -0.1) if row["ADM2_FR"] == "N'Djaména" else (0, 0)
    )
    axs[0].annotate(
        row["ADM2_FR"],
        xy=(centroid.x + x_shift, centroid.y + y_shift),
        ha="center",
        va="center",
        fontsize=10,
    )

plt.subplots_adjust(wspace=0)
```

```python
cols = [
    # "ADM1_PCODE",
    "ADM1_FR",
    # "ADM2_PCODE",
    "ADM2_FR",
    # "total_pop",
    "total_exposed",
    "frac_exposed",
    # "geometry",
]
tot_label = "Pop. totale exposée"
frac_label = "Frac. de pop. exposée"
avg_exposure_plot[cols].sort_values("total_exposed", ascending=False).iloc[
    :10
].rename(
    columns={
        "total_exposed": tot_label,
        "frac_exposed": frac_label,
        "ADM1_FR": "Région",
        "ADM2_FR": "Département",
    }
).style.background_gradient(
    cmap="Purples"
).format(
    "{:,.0f}", subset=[tot_label]
).format(
    "{:.2f}", subset=[frac_label]
)
```

```python
cols = [
    # "ADM1_PCODE",
    "ADM1_FR",
    # "ADM2_PCODE",
    "ADM2_FR",
    # "total_pop",
    "total_exposed",
    "frac_exposed",
    # "geometry",
]
tot_label = "Pop. totale exposée"
frac_label = "Frac. de pop. exposée"
avg_exposure_plot_aoi[cols].sort_values("total_exposed", ascending=False).iloc[
    :
].rename(
    columns={
        "total_exposed": tot_label,
        "frac_exposed": frac_label,
        "ADM1_FR": "Région",
        "ADM2_FR": "Département",
    }
).style.background_gradient(
    cmap="Purples"
).format(
    "{:,.0f}", subset=[tot_label]
).format(
    "{:.2f}", subset=[frac_label]
)
```

```python
cols = [
    "ADM1_PCODE",
    "ADM1_FR",
    "ADM2_PCODE",
    "ADM2_FR",
    "total_pop",
    "total_exposed",
    "frac_exposed",
    # "geometry",
]
filename = "tcd_aoi_adm2_average_flood_exposed.csv"
avg_exposure_plot_aoi[cols].sort_values(
    "total_exposed", ascending=False
).to_csv(floodscan.PROC_FS_DIR / filename, index=False)
```

```python
exposure_raster = floodscan.load_raster_flood_exposures()
exposure_raster_aoi = exposure_raster.rio.clip(
    adm2_aoi.geometry, all_touched=True
)
```

```python
exposure_raster_aoi_mean = exposure_raster_aoi.mean(dim="year")
exposure_raster_aoi_mean = exposure_raster_aoi_mean.where(
    exposure_raster_aoi_mean > 5
)

exposure_raster_mean = exposure_raster.mean(dim="year")
exposure_raster_mean = exposure_raster_mean.where(exposure_raster_mean > 5)
```

```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))

# pop
pop.plot(ax=axs[0], cmap="Greys", vmax=500, add_colorbar=False)
axs[0].set_title("Population, 2020")

# flooding
fs_mean.where(fs_mean > 0.05).plot(ax=axs[1], cmap="Blues", add_colorbar=False)
axs[1].set_title("Fraction inondée typique, 1998-2023")

# exposure
exposure_raster_mean.plot(
    ax=axs[2], cmap="Purples", vmax=80, add_colorbar=False
)
axs[2].set_title(
    "Population totale typiquement exposée aux inondations, 1998-2023"
)

for ax in axs:
    adm2.boundary.plot(ax=ax, linewidth=0.2, color="k")
    ax.axis("off")

plt.subplots_adjust(wspace=0.2)
```

```python
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))

# pop
pop_aoi.plot(ax=axs[0], cmap="Greys", vmax=500, add_colorbar=False)
axs[0].set_title("Population, 2020")

# flooding
fs_aoi_mean.where(fs_aoi_mean > 0.05).plot(
    ax=axs[1], cmap="Blues", add_colorbar=False
)
axs[1].set_title("Fraction inondée typique, 1998-2023")

# exposure
exposure_raster_aoi_mean.plot(
    ax=axs[2], cmap="Purples", vmax=80, add_colorbar=False
)
axs[2].set_title(
    "Population totale typiquement exposée aux inondations, 1998-2023"
)
for idx, row in adm2_aoi.iterrows():
    # Calculate the centroid of each geometry
    centroid = row.geometry.centroid
    x_shift, y_shift = (
        (-0.25, -0.1) if row["ADM2_FR"] == "N'Djaména" else (0, 0)
    )
    axs[2].annotate(
        row["ADM2_FR"],
        xy=(centroid.x + x_shift, centroid.y + y_shift),
        ha="center",
        va="center",
        fontsize=8,
    )

for ax in axs:
    adm2_aoi.boundary.plot(ax=ax, linewidth=0.2, color="k")
    ax.axis("off")

plt.subplots_adjust(wspace=0.2)
```

```python

```
