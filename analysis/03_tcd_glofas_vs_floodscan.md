# Comparing flood events with historical streamflow

This notebook looks at the correlation between peaks in historical
 streamflow (from GloFAS reanalysis) and the timing of past flood events
  identified from floodscan.

Just loading the data and some simple plots but real analysis still has to
 be done

Some questions:

- What is/are the best reporting point to use?
- In the impact data and from talks it is stated that 2020 is a bad year
  but from my first looks we don't see that back in GloFAS or Floodscan.
  What might be the reason?
- How is the correspondence between floodscan and GloFAS?
- How is the correspondence between these two and the impact data?

![afbeelding.png](https://drive.google.com/uc?export=view&id=1bwjql5wk8kcBX6EGEOaFldo__-16Pp84)

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import altair as alt
import pandas as pd
from aatoolbox import GlofasReanalysis
from aatoolbox.utils import raster
import matplotlib.pyplot as plt

from src import constants
from src.utils import load_floodscan, get_return_periods_dataframe
```

```python
STATIONS = {"N'Djamena": "Ndjamena Fort Lamy", "Mayo-Kebbi Est": "Mailao"}
```

```python
# needed to plot dataframes with Altair of more than 5000 rows
alt.data_transformers.enable("data_server")
```

```python
glofas_reanalysis = GlofasReanalysis(
    country_config=constants.country_config,
    geo_bounding_box=constants.geo_bounding_box,
)

ds_glofas = glofas_reanalysis.load()
```

```python
ds_glofas
```

```python
# What is the time vs valid_time?
```

```python
df_glofas = (
    ds_glofas.to_dataframe()
    .reset_index()
    .drop(["step", "surface", "valid_time"], axis=1)
)[["time"] + list(STATIONS.values())]
df_glofas["year"] = df_glofas.time.dt.year
```

```python
df_glofas
```

```python
# plot timeseries of two reporting points
df_long = df_glofas.drop("year", axis=1).melt(
    "time", var_name="station", value_name="discharge"
)
plt_orig = (
    alt.Chart()
    .mark_line()
    .encode(
        x="time:T",
        y="discharge:Q",
    )
    .properties(
        width=1000,
        height=300,
    )
)

alt.layer(plt_orig, data=df_long).properties(
    width=1000, title="River discharge at Ndjamena_Fort_Lamy"
).facet("station:N", columns=1).resolve_scale(y="independent")
```

## Compare GLOFAS and floodscan

```python
gdf_adm1 = constants.gdf_adm1[
    constants.gdf_adm1["admin1Name"].isin(STATIONS.keys())
]
gdf_adm1
```

```python
ds_floodscan = load_floodscan()
```

```python
ds_floodscan
```

```python
df_floodscan_orig = ds_floodscan["SFED_AREA"].aat.compute_raster_stats(
    gdf=gdf_adm1, feature_col="admin1Name", stats_list=["mean"]
)
adm_col = "admin1Name"
# compute rolling mean
df_floodscan_orig["mean_rolling"] = (
    df_floodscan_orig.sort_values([adm_col, "time"])
    .groupby(adm_col, as_index=False)[f"mean_{adm_col}"]
    .rolling(10, min_periods=10)
    .mean()
    .mean_admin1Name
)
df_floodscan = df_floodscan_orig.pivot(
    index="time", columns="admin1Name", values="mean_rolling"
).reset_index()
df_floodscan["year"] = df_floodscan.time.dt.year
```

```python
start_slice = "1998-01-01"
end_slice = "2021-12-31"


def filter_event_dates(df_event, start, end):
    return df_event[
        (df_event["time"] < str(end)) & (df_event["time"] > str(start))
    ].reset_index()


def get_df_glofas_rp(station, years=None):
    return get_return_periods_dataframe(
        df_glofas[["year", station]]
        .sort_values(station, ascending=False)
        .drop_duplicates(["year"]),
        method="empirical",
        rp_var=station,
        years=years,
    )


def get_df_floodscan_rp(adm1, years=None):
    return get_return_periods_dataframe(
        df_floodscan[["year", adm1]]
        .sort_values(adm1, ascending=False)
        .drop_duplicates(["year"]),
        method="empirical",
        rp_var=adm1,
        round_rp=False,
        years=years,
    )


for adm1, station in STATIONS.items():

    fig, axs = plt.subplots(
        1,
        figsize=(15, 6),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.suptitle(f"Historical streamflow at {station}")

    da_plt = ds_glofas[station].sel(time=slice(start_slice, end_slice))
    df_floodscan_adm = filter_event_dates(
        df_floodscan[["time", adm1]],
        start_slice,
        end_slice,
    )

    observations = da_plt.values
    x = da_plt.time
    ax = axs[0, 0]
    ax.plot(da_plt.time, da_plt.values, c="k", lw=0.75, alpha=0.75)
    ax.set_ylabel("Discharge [m$^3$ s$^{-1}$]")
    ax.plot([], [], label="GloFas", color=f"black")
    ax.plot([], [], label="Floodscan", color=f"blue")
    ax2 = ax.twinx()
    ax2.plot(df_floodscan_adm.time, df_floodscan_adm[adm1])
    ax2.set_ylabel("Flooded fraction")

    # plt rp lines
    df_glofas_rp = get_df_glofas_rp(station)
    df_floodscan_rp = get_df_floodscan_rp(adm1)
    for rp_glofas, rp_floodscan in zip(
        df_glofas_rp["rp"], df_floodscan_rp["rp"]
    ):
        ax.axhline(rp_glofas, c="k", lw=0.5)
        ax2.axhline(rp_floodscan)

    ax.figure.legend()
```

It looks like there could potentially be some correspondence.

Next we compute the years during which the highest peaks were observed for
Glofas and Floodscan. With the goal to see how well they correspond.

```python
import xarray as xr
import numpy as np
from typing import List


def get_dates_list_from_data_array(
    da: xr.DataArray, threshold: float, min_duration: int = 1
) -> List[np.datetime64]:
    """
    Given a data array of a smoothly varying quantity over time,
    get the dates of an event occurring where the quantity crosses
    some threshold for a specified duration. If the duration is more than
    one timestep, then the event date is defined as the timestep when
    the duration is reached.
    :param da: Data array with the main quantity
    :param threshold: Threshold >= which an event is defined
    :param min_duration: Number of timesteps above the quantity to be
    considered an event
    :return: List of event dates
    """
    groups = get_groups_above_threshold(
        observations=da.to_masked_array(),
        threshold=threshold,
        min_duration=min_duration,
    )
    return [da.time[group[0] + min_duration - 1].data for group in groups]


def get_dates_list_from_dataframe(
    df: pd.DataFrame, threshold: float, cname: str, min_duration: int = 1
) -> List[np.datetime64]:
    groups = get_groups_above_threshold(
        observations=df[cname],
        threshold=threshold,
        min_duration=min_duration,
    )
    return [
        np.datetime64(df.time[group[0] + min_duration - 1]) for group in groups
    ]


def get_groups_above_threshold(
    observations: np.ndarray,
    threshold: float,
    min_duration: int = 1,
    additional_condition: np.ndarray = None,
) -> List:
    """
    Get indices where consecutive values are equal to or above a
    threshold :param observations: The array of values to search for
    groups (length N) :param threshold: The threshold above which the
    values must be :param min_duration: The minimum group size (default
    1) :param additional_condition: (optional) Any additional condition
    the values must satisfy (array-like of bools, length N) :return:
    list of arrays with indices
    """
    condition = observations >= threshold
    if additional_condition is not None:
        condition = condition & additional_condition
    groups = np.where(np.diff(condition, prepend=False, append=False))[
        0
    ].reshape(-1, 2)
    return [group for group in groups if group[1] - group[0] >= min_duration]


def get_detection_stats(
    true_event_dates: np.ndarray,
    forecasted_event_dates: np.ndarray,
    days_before_buffer: int,
    days_after_buffer: int,
) -> dict:
    """
    Give a list of true and forecasted event dates, calculate how many
    true / false positives and false negatives occurred
    :param true_event_dates: A list of dates when the true events occurred
    :param forecasted_event_dates: A list of dates when the events were
    forecasted to occur
    :param days_before_buffer: How many days before the forecasted date the
    true event can occur. Usually set to the lead time or a small number
    (even 0)
    :param days_after_buffer: How many days after the forecasted date the
    true event can occur. Can usually be a generous number
    like 30, since forecasting too early isn't usually an issue
    :return: dictionary with parameters
    """
    df_detected = pd.DataFrame(
        0, index=np.array(true_event_dates), columns=["detected"]
    )
    FP = 0
    # Loop through the forecasted event
    for forecasted_event in forecasted_event_dates:
        # Calculate the offset from the true dates
        days_offset = (true_event_dates - forecasted_event) / np.timedelta64(
            1, "D"
        )
        # Calculate which true events were detected by this forecast event
        detected = (days_offset >= -1 * days_before_buffer) & (
            days_offset <= days_after_buffer
        )
        df_detected.loc[detected, "detected"] += 1
        # If there were no detections at all, it's a FP
        if not sum(detected):
            FP += 1
    return {
        # TP is the number of true events that were detected
        "TP": sum(df_detected["detected"] > 0),
        # FN is the number of true events that were not detected
        "FN": sum(df_detected["detected"] == 0),
        "FP": FP,
    }
```

Below we're checking the correspondance between GloFAS and
Floodscan "events", i.e. when
RP thresholds are crossed. The main free parameters
are the DAYS_BEFORE/AFTER_BUFFERs,
as these determine the window that is allowed for
overlap to be considered a TP.
I used 30 days on both sides since we're just
interested in general correspondance between
the GloFAS model and floodscan.

```python
DAYS_BEFORE_BUFFER = 30
DAYS_AFTER_BUFFER = 30

df_station_stats = pd.DataFrame(columns=["station", "rp", "TP", "FP", "FN"])

for adm1, station in STATIONS.items():

    forecast = ds_glofas[station].sel(time=slice(start_slice, end_slice))
    model = filter_event_dates(
        df_floodscan[["time", adm1]],
        start_slice,
        end_slice,
    )

    df_glofas_rp = get_df_glofas_rp(station)
    df_floodscan_rp = get_df_floodscan_rp(adm1)

    for rp in [1.5, 2.0, 3.0, 5.0]:
        model_dates = get_dates_list_from_dataframe(
            model, df_floodscan_rp.loc[rp, "rp"], adm1
        )
        forecast_dates = get_dates_list_from_data_array(
            forecast, df_glofas_rp.loc[rp, "rp"]
        )
        detection_stats = get_detection_stats(
            true_event_dates=model_dates,
            forecasted_event_dates=forecast_dates,
            days_before_buffer=DAYS_BEFORE_BUFFER,
            days_after_buffer=DAYS_AFTER_BUFFER,
        )
        df_station_stats = df_station_stats.append(
            {**{"station": station, "rp": rp}, **detection_stats},
            ignore_index=True,
        )
```

```python
df_station_stats
```

```python

  (alt.Chart(df_station_stats.melt(["station", "rp"]))
  .mark_line()
    .encode(
        x="rp:O",
        y="value",
        color="station",
        strokeDash='variable'
    )
  ).properties(width=400)


```

The above plot shows TP, FP, and FN. Basically you want
TP to be >> FP or FN for the model to be working well.
Indeed TP is above the others for low RPs in Ndjamena,
and for both low and 1-in-5 year RP in the southern region.

```python
def get_more_detection_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute precision, recall, F1, POD and FAR
    :param df: Dataframe with columns TP, FP and FN
    :return: Dataframe with additional stats columns
    """
    # Convert everything to float to avoid zero division errors
    for q in ["TP", "FP", "FN"]:
        df[q] = df[q].astype("float")
    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["recall"] = df["TP"] / (df["TP"] + df["FN"])
    df["F1"] = 2 / (1 / df["precision"] + 1 / df["recall"])
    df["POD"] = df["recall"]
    df["FAR"] = 1 - df["precision"]
    for q in ["TP", "FP", "FN"]:
        df[q] = df[q].astype("int")
    return df


df_prf1 = get_more_detection_stats(df_station_stats.copy())
df_prf1 = df_prf1[["station", "rp", "precision", "recall", "F1"]]
```

```python
(
    alt.Chart(df_prf1.melt(["station", "rp"]))
    .mark_line()
    .encode(x="rp:O", y="value", color="station", strokeDash="variable")
).properties(width=400)
```

Good performance, especially for 1 in 5 year RP in Mailao.

## Event timeline comparison

Make a plot of the events to get an idea
if there are many repetitions per year, and if so,
how spaced out they are

```python
# Choose an RP -- let's do 1 in 4 y
rp = 4

# Get the events and plot
for adm1, station in STATIONS.items():

    forecast = ds_glofas[station].sel(time=slice(start_slice, end_slice))
    model = filter_event_dates(
        df_floodscan[["time", adm1]],
        start_slice,
        end_slice,
    )
    df_glofas_rp = get_df_glofas_rp(station, years=[rp])
    df_floodscan_rp = get_df_floodscan_rp(adm1, years=[rp])

    model_dates = get_dates_list_from_dataframe(
        model, df_floodscan_rp.loc[rp, "rp"], adm1
    )
    forecast_dates = get_dates_list_from_data_array(
        forecast, df_glofas_rp.loc[rp, "rp"]
    )
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(
        model_dates, [1] * len(model_dates), "o", label="floodscan", alpha=0.5
    )
    ax.plot(
        forecast_dates,
        [1.1] * len(forecast_dates),
        "^",
        label="glofas",
        alpha=0.5,
    )
    ax.set_title(adm1)
    ax.legend()
    ax.grid(visible=True, which="both")
```

Unfortunately there are a lof repeat events per year.
They generally seem clustereed though, and this
problem could likely be resolved by smoothing floodscan.
The only big gap looks like in GloFAS at Mayo-Kebbi Est
in 1998 and 1998. Check the data:

```python
forecast_dates
```

Indeed there is a difference of > month in 1998 and > 15 days
in 1999.

Make a plot showing the timeline with exceedance, normalized to the
the RPs

```python
# plot the return periods by year for the two sources
df_sources = pd.DataFrame()
df_sources["year"] = range(1998, 2022)
for adm1, station in STATIONS.items():

    forecast = ds_glofas[station].sel(time=slice(start_slice, end_slice))
    model = filter_event_dates(
        df_floodscan[["time", adm1]],
        start_slice,
        end_slice,
    )
    df_glofas_rp = get_df_glofas_rp(station, years=[rp]).loc[rp, "rp"]
    df_floodscan_rp = get_df_floodscan_rp(adm1, years=[rp]).loc[rp, "rp"]
    model_dates = get_dates_list_from_dataframe(model, df_floodscan_rp, adm1)
    forecast_dates = get_dates_list_from_data_array(forecast, df_glofas_rp)
    df_sources[f"floodscan_{adm1}"] = np.where(
        df_sources.year.isin(
            set([pd.to_datetime(d).year for d in model_dates])
        ),
        True,
        False,
    )
    df_sources[f"glofas_{station}"] = np.where(
        df_sources.year.isin(
            set([pd.to_datetime(d).year for d in forecast_dates])
        ),
        True,
        False,
    )
df_long = df_sources.melt(
    "year", var_name="source", value_name=f"rp_{rp}_years"
)
value_color_mapping = {
    True: "red",
    False: "#D3D3D3",
}

(
    alt.Chart(df_long)
    .mark_rect()
    .encode(
        x="year:N",
        y=alt.Y(
            "source:N",
            sort=[
                "floodscan_N'Djamena",
                "floodscan_Mayo-Kebbi Est",
                "glofas_Ndjamena Fort Lamy",
                "glofas_Mailao",
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
    )
    .properties(
        title=f"1 in {rp} year return period years of FloodScan and GloFas"
    )
)
```

```python
# Get the events and plot
for adm1, station in STATIONS.items():

    forecast = ds_glofas[station].sel(time=slice(start_slice, end_slice))
    model = filter_event_dates(
        df_floodscan[["time", adm1]],
        start_slice,
        end_slice,
    )
    df_glofas_rp = get_df_glofas_rp(station, years=[rp]).loc[rp, "rp"]
    df_floodscan_rp = get_df_floodscan_rp(adm1, years=[rp]).loc[rp, "rp"]
    model_dates = get_dates_list_from_dataframe(model, df_floodscan_rp, adm1)
    forecast_dates = get_dates_list_from_data_array(forecast, df_glofas_rp)
    # Figure out which years need to be plotted
    years = sorted(
        list(
            set(
                [pd.to_datetime(d).year for d in forecast_dates]
                + [pd.to_datetime(d).year for d in model_dates]
            )
        )
    )
    fig, axs = plt.subplots(2, len(years) // 2, figsize=(15, 8), facecolor="w")
    fig.suptitle(adm1)
    for year, ax in zip(years, axs.flatten()):
        # Filter based on year, from july to october
        plot_start = f"{year}-07-01"
        plot_end = f"{year}-10-31"
        gf = forecast.sel(time=slice(plot_start, plot_end)) / df_glofas_rp
        fs = filter_event_dates(model, plot_start, plot_end)
        fs[adm1] = fs[adm1] / df_floodscan_rp
        # Plot general
        ax.plot(gf.time, gf, label="glofas")
        ax.plot(fs["time"], fs[adm1], label="floodscan", c="g")
        # Plot RP exceedence
        gfe = gf.where(gf > 1, np.nan)
        fse = fs.copy()
        fse[adm1] = np.where(fse[adm1] < 1, np.nan, fse[adm1])
        ax.plot(gfe.time, gfe, "r", lw=2)
        ax.plot(fse["time"], fse[adm1], "orange", lw=2, marker=".")
        # Cleaning axes
        ax.set_title(year)
        ax.set_ylim(-0.05, 1.5)
        ax.axhline(1, c="k")
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.legend()
```
