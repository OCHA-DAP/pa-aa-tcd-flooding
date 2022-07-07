# Flooded fraction in the region of interest

This notebook repeats the analysis of `01_tcd_foodscan_country.md` but with a focus
on the priority areas which are N'Djamena and Mayo Kebbi Est

Note: this notebook is a bit less ordered

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import numpy as np
import altair as alt
import pandas as pd
import panel.widgets as pnw
import hvplot.xarray
import matplotlib.pyplot as plt
```

```python
from src.constants import gdf_adm1
from src.utils import (
    load_floodscan,
    get_return_periods_dataframe,
    load_compare_datasources,
)
```

```python
%load_ext rpy2.ipython
```

```R tags=[]
library(tidyverse)
```

```R
plotFloodedFraction <- function (df,y_col,facet_col,title){
df %>%
ggplot(
aes_string(
x = "time",
y = y_col
)
) +
stat_smooth(
geom = "area",
span = 1/4,
fill = "#ef6666"
) +
scale_x_date(
date_breaks = "3 months",
date_labels = "%b"
) +
facet_wrap(
as.formula(paste("~", facet_col)),
scales="free_x",
ncol=5
) +
ylab("Flooded percentage")+
xlab("Month")+
labs(title = title)+
theme_minimal()
}
```

```python
# admin1's of interest
adm1_list = ["N'Djamena", "Mayo-Kebbi Est"]
```

```python
adm_col = "admin1Name"
```

```python
gdf_adm1["include"] = np.where(gdf_adm1[adm_col].isin(adm1_list), True, False)
adms = (
    alt.Chart(gdf_adm1)
    .mark_geoshape(stroke="black")
    .encode(
        color=alt.Color("include", scale=alt.Scale(range=["grey", "red"])),
        tooltip=["admin1Name"],
    )
    .properties(width=400, height=400, title="Admins of focus")
)
adms
```

```python
fs_clip = load_floodscan()
```

```python
da_clip = fs_clip.SFED_AREA
```

```python
gdf_reg = gdf_adm1[gdf_adm1[adm_col].isin(adm1_list)]
```

```python
da_reg = da_clip.rio.clip(gdf_reg.geometry)
```

```python
# plot raster data with time slider
# can select subset of data to make it easier navigatible
# with slider
(
    da_reg
    #  .sel(time=da_clip.time.dt.year==2014)
    .interactive.sel(time=pnw.DiscreteSlider).plot(vmin=0, vmax=1, cmap="GnBu")
)
```

```python
df_floodscan_reg = da_clip.aat.compute_raster_stats(
    gdf=gdf_adm1[gdf_adm1.include],
    feature_col=adm_col,
    stats_list=["mean"],
    all_touched=False,
)
df_floodscan_reg["year"] = df_floodscan_reg.time.dt.year
df_floodscan_reg["month"] = df_floodscan_reg.time.dt.month
```

```python
# should document but for now removing 2022 as it is not a full year
# but does have very high values till feb, so computations get
#  a bit skewed with that
df_floodscan_reg = df_floodscan_reg[df_floodscan_reg.year <= 2021]
```

## Plot patterns

We plot the smoothed data per year (with ggplot cause it is awesome).

We can see that:

- 2014, 2016, 2017, 2020, and 2021 indeed had clearly the highest peak.
- This is the line with the signal on country level
- We also see some differences, e.g. the 2019 peak is lower.

N'Djamena

```python
df_floodscan_ndjam = df_floodscan_reg[df_floodscan_reg[adm_col] == "N'Djamena"]
```

```R magic_args="-i df_floodscan_ndjam -w 40 -h 20 --units cm"
df_plot <- df_floodscan_ndjam %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_admin1Name = mean_admin1Name*100)
plotFloodedFraction(df_plot,'mean_admin1Name','year',"Flooded fraction of N'Djamena")
```

Mayo-Kebbi Est

```python
df_floodscan_mke = df_floodscan_reg[
    df_floodscan_reg[adm_col] == "Mayo-Kebbi Est"
]
```

```R magic_args="-i df_floodscan_mke -w 40 -h 20 --units cm"
df_plot <- df_floodscan_mke %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_admin1Name = mean_admin1Name*100)
plotFloodedFraction(
    df_plot,'mean_admin1Name','year',"Flooded fraction of Mayo-Kebbi Est")
```

## Return periods

Next we compute the return period and check which years had one or more
 peaks above the return period.
We use the analytical method to compute the return period

```python
# compute rolling mean to determine peak as raw
# data is quite spikey
df_floodscan_reg["mean_rolling"] = (
    df_floodscan_reg.sort_values([adm_col, "time"])
    .groupby(adm_col, as_index=False)[f"mean_{adm_col}"]
    .rolling(10, min_periods=10)
    .mean()
    .mean_admin1Name
)
```

```python
df_floodscan_piv = df_floodscan_reg.pivot(
    index=["time", "year"], columns="admin1Name", values="mean_rolling"
).reset_index()
```

```python
def get_df_floodscan_rp(df, adm1):
    return get_return_periods_dataframe(
        df[["year", adm1]]
        .sort_values(adm1, ascending=False)
        .drop_duplicates(["year"]),
        method="empirical",
        rp_var=adm1,
        round_rp=False,
        years=[1.5, 2, 3, 4, 5],
    )
```

```python
# use the peak per year, i.e. only one event per year
years = [1.5, 2, 3, 4, 5]
df_floodscan_year = df_floodscan_piv.groupby("year", as_index=False).max()
for adm in adm1_list:
    df_rp_adm = get_df_floodscan_rp(df_floodscan_year, adm)
    display(df_rp_adm)
    for y in years:
        df_floodscan_year[f"{adm}_rp_{y}"] = np.where(
            df_floodscan_year[adm] >= df_rp_adm.loc[y].values[0], True, False
        )
```

```python
# display the values to get a feeling for them
for adm in adm1_list:
    df_max_5 = (
        df_floodscan_year.loc[
            df_floodscan_year[f"{adm}_rp_4"], ["time", "year", adm]
        ].sort_values(adm)
        #         .head()
    )
    display(df_max_5)

    # #TODO: arghh somehow it is not finding the dates while it did so
    # #before.. must have smth to do with date formats
    # #Not very important though
    # da_clip_peak = da_clip.sel(time=da_clip.time.isin(list(df_max_5.time)))
    # g=da_clip_peak.plot(col='time',
    #                    )
    # for ax in g.axes.flat:
    #     ax.axis("off")
    # g.fig.suptitle(
    # f"Flooded fraction during peak for 1 in 5 year return period years",y=1.1
    # );
```

## Compare years

Plot 1 in 4 year return period years for the country, and the two regions of interest

```python
rp = 4
df_sources = pd.DataFrame()
df_sources["year"] = range(1998, 2022)
```

```python
for adm in adm1_list:
    df_sources[f"{adm}"] = np.where(
        df_sources.year.isin(
            df_floodscan_year.loc[df_floodscan_year[f"{adm}_rp_{rp}"]].year
        ),
        True,
        False,
    )
```

```python
df_alls = load_compare_datasources()
```

```python
df_sources = df_sources.merge(
    df_alls[["year", "FloodScan Country"]], on="year"
).rename(columns={"FloodScan Country": "Country"})
```

```python
df_long = df_sources.melt(
    "year", var_name="source", value_name=f"rp_{rp}_years"
)
value_color_mapping = {
    True: "red",
    False: "#D3D3D3",
}
# show 1 in x year return period years for both data sources
alt.Chart(df_long).mark_rect().encode(
    x="year:N",
    y=alt.Y(
        "source:N",
        sort=[
            "Country",
            "Mayo-Kebbi Est",
            "N'Djamena",
        ],
    ),
    color=alt.Color(
        f"rp_{rp}_years:N",
        scale=alt.Scale(
            range=list(value_color_mapping.values()),
            domain=list(value_color_mapping.keys()),
        ),
        legend=alt.Legend(title=f"1 in {rp} year rp"),
    ),
).properties(title=f"Floodscan 1 in {rp} year return period years")
```
