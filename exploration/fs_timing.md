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

# Floodscan timing
<!-- markdownlint-disable MD013 -->
Checking timing of Floodscan in N'Djamena and Mayo-Kebbi Est vs observational and GloFAS reanalysis at N'Djamena station.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ocha_stratus as stratus
import pandas as pd

from src.datasources import glofas, dre
from src.constants import *
```

```python
df_dre = dre.open_dre_obsv(from_blob=True)
```

```python
# exclude past 2023 because timing seems off
df_dre = df_dre[df_dre["Date"].dt.year < 2023]
```

```python
names = {NDJAMENA1: "N'Djamena", MAYOKEBBIEST1: "Mayo-Kebbi Est"}
```

```python
query = f"""
SELECT pcode, valid_date, mean
FROM public.floodscan
WHERE pcode IN {(NDJAMENA1, MAYOKEBBIEST1)}
AND band = 'SFED'
"""
df_fs = pd.read_sql(
    query, stratus.get_engine(stage="prod"), parse_dates=["valid_date"]
)
df_fs["name"] = df_fs["pcode"].replace(names)
```

```python
df_fs
```

```python
df_gf = glofas.load_glofas_reanalysis(station_name="ndjamena")
```

```python
df_gf
```

```python
df_daily = (
    df_fs.pivot(columns="name", index="valid_date", values="mean")
    .reset_index()
    .merge(df_gf.rename(columns={"time": "valid_date"}))
    .merge(df_dre.rename(columns={"Date": "valid_date"}), how="left")
)
```

```python
df_daily
```

```python
for col in df_daily:
    if col == "valid_date":
        continue
    df_daily[f"{col}_rel"] = df_daily[col] / df_daily[col].max()
```

```python
df_daily
```

```python
for year, group in df_daily.groupby(df_daily["valid_date"].dt.year):
    fig, ax = plt.subplots()
    group.set_index("valid_date")[[x for x in group if "rel" in x]].plot(ax=ax)
    ax.set_ylim((0, 1))
```

```python
df_daily["plot_date"] = df_daily["valid_date"].map(
    lambda x: pd.Timestamp(
        year=1904,
        month=x.month,
        day=x.day,
    )
)
```

```python
df_daily_mean = (
    df_daily.groupby(df_daily["plot_date"])[
        [x for x in df_daily if "rel" in x]
    ]
    .mean()
    .reset_index()
)
```

```python
df_daily_mean
```

```python
rel_cols
```

```python
labels = {
    "dis24_rel": "GloFAS reanalysis",
    "level_cm_rel": "Observed river level",
}
```

```python
colors = ["steelblue", "darkorange", "green", "purple"]

# Get the list of columns to plot (assumes order is consistent)
rel_cols = [col for col in df_daily.columns if "rel" in col]

# Map columns to colors by position
col_colors = dict(zip(rel_cols, colors))

fig, ax = plt.subplots(dpi=200)

for year, group in df_daily.groupby(df_daily["valid_date"].dt.year):
    group = group.set_index("plot_date")
    for col in rel_cols:
        ax.plot(
            group.index,
            group[col],
            color=col_colors[col],
            linewidth=1,
            alpha=0.1,
        )
for col in rel_cols:
    ax.plot(
        df_daily_mean["plot_date"],
        df_daily_mean[col],
        color=col_colors[col],
        linewidth=2,
        label=labels.get(col, col).removesuffix("_rel"),
    )

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax.set_ylim((0, 1))
ax.set_xlim(
    (df_daily_mean["plot_date"].min(), df_daily_mean["plot_date"].max())
)

[ax.spines[x].set_visible(False) for x in ["top", "right"]]
ax.legend()
```

To answer the question about timing, here is a plot that shows:

- purple: Mayo-Kebbi Est (new province) Floodscan extent
- orange: N'Djamena (old province) Floodscan extent
- green: GloFAS reanalysis (N'Djamena station)
- purple: observed river level (N'Djamena station)

The plot shows all years since 1998 as fine lines, and the day-wise mean as the bold line.
We can see a few things:

1. The river has pretty much a single big peak each year
2. GloFAS gets the shape roughly right, but about a month early
3. Mayo-Kebbi Est flood extent matches the GloFAS reanalysis nicely
    - I suspect this is because it's upstream of N'Djamena, so this kind of "cancels out" the hydrological model miscalibration
    - I'll get the average timings to put in the framework, so people know that we don't get the extra 30-ish days leadtime in the new province
4. The elephant in the room- the N'Djamena flood extent has a bunch of big peaks earlier in the year, way before the river level peaks.
    - We already knew this, since this went into the framework last year
    - If you look closely, there are some peaks in Oct-Nov. Based on previous analysis, these align with riverine flooding years. So, something about Floodscan's bias (and low resolution) means that we barely pick up on this, but we do pick up one the earlier peaks (from flash flooding? or something else).
