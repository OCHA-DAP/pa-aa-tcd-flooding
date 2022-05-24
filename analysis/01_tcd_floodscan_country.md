### Flooded fraction in the country and its correlation to impact

This is an exploration to understanding if anticipating an increased risk in TCD is possible. 
We start by getting a historical overview on the floods in the country from FloodScan
We compare this to the limited impact data we got. 

This is an exploration of a quick understanding of increased risk, not a full AA framework
3 notions: 

- recommended focus on riverine flooding 
- priority areas/provinces: N'Djamena and Mayo Kebbi Est
- expected peak period: september-october (sometimes until november)

This notebook focuses on the whole country and `02_tcd_floodscan_roi.md` zooms in on the priority provinces



```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import altair as alt
import numpy as np
import pandas as pd
import panel.widgets as pnw
import hvplot.xarray
import matplotlib.pyplot as plt
```

```python
from aatoolbox import CodAB

from src.constants import gdf_adm0, iso3
from src.utils import (
    load_floodscan,
    load_emdat_exploration,
    load_floodscan_stats,
    load_cerf,
    load_ifrc,
    get_return_periods_dataframe,
)
```

```python
# needed to plot dataframes with Altair of more than 5000 rows
alt.data_transformers.enable("data_server")
```

```python
%load_ext rpy2.ipython
```

```R tags=[]
library(tidyverse)
```

#### define functions

```R
plotFloodedFraction <- function (df,y_col,facet_col){
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
ncol=4
) +
ylab("Flooded fraction")+
xlab("Month")+
theme_minimal()
}
```

```python
bound_col = "admin0Pcod"
```

```python
fs_clip = load_floodscan()
```

```python
da_clip = fs_clip.SFED_AREA
```

```python
# TODO: move to .py file
# df_floodscan_country=compute_raster_statistics(
#         gdf=gdf_adm0,
#         bound_col=bound_col,
#         raster_array=da_clip,
#         lon_coord="lon",
#         lat_coord="lat",
#         stats_list=["median","mean","max","count","sum"], #std, count
#         #computes value where 20% of the area is above that value
#         percentile_list=[80],
#         #Decided to only use centres, but can change that
#         all_touched=False,
#     )
# df_floodscan_country['year']=df_floodscan_country.time.dt.year
# # df_floodscan_country.to_csv(country_data_exploration_dir/'floodscan'/f'{iso3}_floodscan_adm0_stats.csv')
```

```python
df_floodscan_country = load_floodscan_stats(adm_level=0)
```

```python
df_floodscan_country[f"mean_{bound_col}_perc"] = (
    df_floodscan_country[f"mean_{bound_col}"] * 100
)
```

We can plot the data over all years. We plot both the raw data and smoothed data. 
We see a yearly pattern where some years the peak is higher than others (though a max of 0.8% of the country is flooded). 

We see that the shape of the peaks also differ. Some are high but short while others the max value is lower but they consist for a longer period. 
What the difference in impact of these patterns is, I am unsure about.

```python
plt_orig = (
    alt.Chart()
    .mark_line()
    .encode(
        x="time:T",
        y=alt.Y(f"mean_{bound_col}_perc:Q", title="daily value"),
    )
    .properties(
        width=1000,
        height=300,
    )
)
plt_mavg = (
    alt.Chart()
    .mark_line()
    .encode(
        x="time:T",
        y=alt.Y("mean_rolling:Q", title="10-day mean"),
    )
)
alt.layer(plt_orig, data=df_floodscan_country).properties(
    width=1000, title="3-week rolling sum of cases per HZ"
)
```

```python
# only select data up to 2021 as 2022 is not complete and messes up the plot
df_floodscan_country_2021 = df_floodscan_country[
    df_floodscan_country.year <= 2021
]
```

```R magic_args="-i df_floodscan_country_2021 -w 40 -h 20 --units cm"
df_plot <- df_floodscan_country_2021 %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'))
plotFloodedFraction(df_plot,'mean_admin0Pcod_perc','year')
```

Next we compute the return period and check which years had a peak above the return period. 
It is discussable whether only looking at the peak is the best method.

```python
df_floodscan_country["mean_rolling"] = (
    df_floodscan_country.sort_values("time")[f"mean_{bound_col}"]
    .rolling(10, min_periods=10)
    .mean()
)
df_floodscan_country["month"] = pd.DatetimeIndex(
    df_floodscan_country["time"]
).month
df_floodscan_country_rainy = df_floodscan_country.loc[
    (df_floodscan_country["month"] >= 7)
    | (df_floodscan_country["month"] <= 11)
]
```

```python
# get one row per adm2-year combination that saw the highest mean value
df_floodscan_peak = df_floodscan_country_rainy.sort_values(
    "mean_rolling", ascending=False
).drop_duplicates(["year"])
```

```python
years = np.arange(1.5, 20.5, 0.5)
```

```python
df_rps_ana = get_return_periods_dataframe(
    df_floodscan_peak,
    rp_var="mean_rolling",
    years=years,
    method="analytical",
    round_rp=False,
)
df_rps_emp = get_return_periods_dataframe(
    df_floodscan_peak,
    rp_var="mean_rolling",
    years=years,
    method="empirical",
    round_rp=False,
)
```

```python
fig, ax = plt.subplots()
ax.plot(df_rps_ana.index, df_rps_ana["rp"], label="analytical")
ax.plot(df_rps_emp.index, df_rps_emp["rp"], label="empirical")
ax.legend()
ax.set_xlabel("Return period [years]")
ax.set_ylabel("Fraction flooded");
```

```python
df_floodscan_peak[
    df_floodscan_peak.mean_rolling >= df_rps_ana.loc[3, "rp"]
].sort_values("year")
```

```python
df_floodscan_peak[
    df_floodscan_peak.mean_rolling >= df_rps_ana.loc[5, "rp"]
].sort_values("year")
```

We can conclude that: 
- Every year a small part of the country gets flooded
- 1998, 2001, 2010, and 2012 saw the highest peak. 
- The peak is generally between Aug and Sep, which is in the second half of the rainy season
- We do see differences in shapes of the peaks during different years
- Flooding should potentially be looked at at a more local scale. 


### Geospatial distribution
We next plot the rasters at the points there was the most flooding. 

From here we can see that the most flooding occurs towards the south. 


```python
timest_rp5 = list(
    df_floodscan_peak[
        df_floodscan_peak.mean_rolling >= df_rps_ana.loc[5, "rp"]
    ].time
)
```

```python
# #TODO: arghh somehow it is not finding the dates while it did so
# #before.. must have smth to do with date formats
# da_clip_peak = da_clip.sel(time=da_clip.time.isin(timest_rp5))
# g=da_clip_peak.plot(col='time',
# #                     cmap="GnBu"
#                    )
# for ax in g.axes.flat:
#     ax.axis("off")
# g.fig.suptitle(f"Flooded fraction during peak for 1 in 5 year return period years",y=1.1);
```

```python
# gif of the timeseries
# the first loop it is whacky but after that it is beautiful
time = pnw.Player(name="time", start=0, end=122, step=7, loop_policy="loop")

# select a year else it takes ages
da_clip.sel(
    time=(da_clip.time.dt.year == 2020)
    & (da_clip.time.dt.month.isin([7, 8, 9, 10, 11]))
).interactive(loc="bottom").isel(time=time).plot(
    #     cmap="GnBu",
    vmin=0,
    vmax=1,
)
```

```python
# gif of the timeseries
# the first loop it is whacky but after that it is beautiful
time = pnw.Player(name="time", start=0, end=122, step=7, loop_policy="loop")

# select a year else it takes ages
da_clip.sel(
    time=(da_clip.time.dt.year == 2012)
    & (da_clip.time.dt.month.isin([7, 8, 9, 10, 11]))
).interactive(loc="bottom").isel(time=time).plot(
    #     cmap="GnBu",
    vmin=0,
    vmax=1,
)
```

### Comparison to impact data
Now that we have the floodscan data, we can check if this corresponds with impact data. 

The sources we have are: 
- CERF allocations
- an internal list shared by IFRC
- EM-DAT


### CERF allocations
CERF has allocation data since 2006. We see that a big allocation on floods was done in 2012, which coincides with the large peak 
we saw in the FloodScan data. 

Smaller allocations were done in 2009 and 2011 that were on displacement but mention floods. The 2011 one could
possibly relate to the flooding we saw in 2010. 

```python
df_cerf = load_cerf()
df_cerf["date"] = df_cerf.dateUSGSignature.dt.to_period("M")
df_country = df_cerf[df_cerf.countryCode == iso3.upper()]
cols = df_country.select_dtypes(include=[np.object]).columns
# remove accents (cause French)
df_country[cols] = df_country[cols].apply(
    lambda x: x.str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)
mask = df_country.apply(
    lambda row: row.astype(str).str.contains("flood", case=False).any(), axis=1
)
df_countryd = df_country.loc[mask]
# group year-month combinations together
df_countrydgy = (
    df_countryd[["date", "totalAmountApproved"]].groupby("date").sum()
)
```

```python
df_countrydgy
```

```python
with pd.option_context("display.max_colwidth", None):
    display(df_countryd[["date", "emergencyTypeName", "projectTitle"]])
```

### IFRC
IFRC shared an internal list of impact years. So we look at these as well. We assume the data is recorded since 2012

From this data we can see that 2012 was clearly the most affected year.
2020 saw the second-largest number of affected people. This might be caused by flooding in the capital #TODO: double-check

Moreover, we can see that 7/10 years are recorded in the data as having some flood inpact (though 2014 and 2016 saw relatively small impact)

It is unclear if e.g. 2013 refers to the impact caused by the 2012 flood. 

```python
df_ifrc = load_ifrc()
# some headers span two rows (gotta love excel)
df_ifrc.columns = df_ifrc.columns.map("".join)
df_ifrc = df_ifrc.rename(
    columns={
        "AnnéeUnnamed: 0_level_1": "year",
        "ProvinceUnnamed: 1_level_1": "province",
    }
).drop(
    ["personnesblessées.1", "personnesblessées.2", "personnesblessées.3"],
    axis=1,
)
df_ifrc["Nombre de décès"] = (
    df_ifrc["Nombre de décès"].replace(" ", 0).replace(np.nan, 0).astype(int)
)
df_ifrc["Têtes de bétail perdues"] = (
    df_ifrc["Têtes de bétail perdues"]
    .replace(" ", 0)
    .replace(np.nan, 0)
    .astype(int)
)
```

```python
df_ifrc_year = df_ifrc.groupby("year", as_index=False).sum()
display(df_ifrc_year)
```

```python
df_ifrc_year_long = df_ifrc_year.melt(
    "year", var_name="indicator", value_name="value"
)
alt.Chart(df_ifrc_year_long).mark_bar().encode(x="year:N", y="value:Q").facet(
    facet="indicator:N", columns=4
).resolve_scale(y="independent")
```

### EM-DAT
Another commonly used source is em-dat

```python
# TODO: move this to .py file
def emdat_clean_transform(df, country_iso):
    """
    Copied/stolen from 510!
    https://github.com/rodekruis/IBF-model/blob/2b2f98c392bc0f997272c86d1f391d33dea3f768/flood/impact-data/utils.py
    emdat_clean_transform
    Simple script that maps EM-DAT data (.xlsx) into IBF-system format
    Parameters
    ----------
    input : str
        name of input file (.xlsx)
    output : str
        name of output file (.csv)
    """

    # filter on country
    df = df[df["ISO"].str.lower() == country_iso]

    # change some column names
    dict_columns = {
        "Disaster Type": "disaster_type",
        "Disaster Subtype": "comments",
        "Location": "location",
        "Total Deaths": "people_dead",
        "No Injured": "people_injured",
        "No Affected": "people_affected",
    }
    df = df.rename(columns=dict_columns)
    df["disaster_type"] = df["disaster_type"].str.lower()

    # parse start and end date
    df = df.rename(
        columns={
            "Start Year": "year",
            "Start Month": "month",
            "Start Day": "day",
        }
    )
    for ix, row in df.iterrows():
        if pd.isna(row["day"]) and not pd.isna(row["month"]):
            df.at[ix, "day"] = 1
    df["start_date_event"] = pd.to_datetime(
        df[["year", "month", "day"]], errors="coerce"
    )
    df = df.drop(columns=["year", "month", "day"])
    df = df.rename(
        columns={"End Year": "year", "End Month": "month", "End Day": "day"}
    )
    for ix, row in df.iterrows():
        if pd.isna(row["day"]) and not pd.isna(row["month"]):
            try:
                df.at[ix, "day"] = calendar.monthrange(
                    int(row["year"]), int(row["month"])
                )[1]
            except:
                pass
        elif (
            pd.isna(row["day"])
            and pd.isna(row["month"])
            and not pd.isna(row["start_date_event"])
        ):
            try:
                df.at[ix, "month"] = row["start_date_event"].month
                df.at[ix, "day"] = calendar.monthrange(
                    int(row["start_date_event"].year),
                    int(row["start_date_event"].month),
                )[1]
            except:
                pass
    df["end_date_event"] = pd.to_datetime(
        df[["year", "month", "day"]], errors="coerce"
    )
    df = df.drop(columns=["year", "month", "day"])
    df["date_event"] = (
        df["start_date_event"]
        + (df["end_date_event"] - df["start_date_event"]) / 2
    )

    for col in df.columns:
        if col not in dict_columns.values() and col not in [
            "start_date_event",
            "end_date_event",
            "date_event",
        ]:
            df = df.drop(columns=[col])

    df["data_source"] = "EM-DAT"
    df["data_source_url"] = "https://www.emdat.be/"
    #     df.to_csv(output_path)

    return df
```

```python
df_raw = load_emdat_exploration()
```

```python
df_emdat = emdat_clean_transform(df_raw, iso3)
```

```python
df_emdat["start_year"] = df_emdat.start_date_event.dt.year
```

```python
df_emdat_year = df_emdat.groupby("start_year", as_index=False).sum()
display(df_emdat_year)
```

```python
df_emdat_year_long = df_emdat_year.melt(
    "start_year", var_name="indicator", value_name="value"
)
alt.Chart(df_emdat_year_long).mark_bar().encode(
    x="start_year:N", y="value:Q"
).facet(facet="indicator:N", columns=4).resolve_scale(y="independent")
```

<!-- #region -->
From the EM-DAT data we see again the impact of the 2012 flood confirmed. 
However, we see also some different patterns. 

For example the impact of 2021 is substanially larger than that of 2020, which was the other way around in the IFRC data. 
In return, 2019 has barely any impact in the EM-DAT data while it has in the IFRC data


We can also look at the data before 2012. We see significant impact in 2010 and 2001, which corresponds to the floodscan data. 
In 1998 there was no impact recorded though, which is suspicious. 

Another suspicion is that there was a substantial number of deaths recorded in 2006 but no people affected. 
<!-- #endregion -->

Questions:
- Why 2020 is known as big flood, which is also partly shown in impact data, but not being seen in floodscan data (nor glofas see later notebook). 
- zoom in to specific areas of impact (see next notebook a little bit as well)


2020 resources:

- [IFRC doc](https://adore.ifrc.org/Download.aspx?FileId=348595) detailing impact and which provinces
- [Images of flooding from floodlist](https://floodlist.com/africa/chad-floods-ndjamena-november-2020)
- [UNOSAT flood extent](https://unosat-maps.web.cern.ch/TD/FL20200826TCD/UNOSAT_A3_Natural_Landscape_FL20200826TCD_20200902_20200906_Chad.pdf)

```python

```
