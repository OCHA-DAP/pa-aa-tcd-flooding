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

# Floodscan validation

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasources import floodscan, codab, glofas
from src.constants import NDJAMENA2
from src.raster import upsample_dataarray
```

```python
# June 1
SEASON_START_DAYOFYEAR = 152
# Oct 1
SEASON_MID_DAYOFYEAR = 274
# Dec 1
SEASON_END_DAYOFYEAR = 335
```

```python
floodscan.process_ndjamena_daily_floodscan()
```

```python
fs_mean = floodscan.load_ndjamena_daily_floodscan()
```

```python
fs_mean[
    (fs_mean["time"] >= "2022-06-01") & (fs_mean["time"] <= "2022-10-01")
].set_index("time")[[x for x in fs_mean.columns if "roll" in x]].plot()
```

```python
rea = glofas.load_reanalysis()
```

```python
compare = rea.merge(fs_mean)
compare["dayofyear"] = compare["time"].dt.dayofyear
compare = compare[
    (compare["dayofyear"] >= SEASON_START_DAYOFYEAR)
    # (compare["dayofyear"] >= SEASON_MID_DAYOFYEAR)
    & (compare["dayofyear"] <= SEASON_END_DAYOFYEAR)
]
compare["roll7_late"] = compare.apply(
    lambda row: (
        row["roll7"] if row["dayofyear"] >= SEASON_MID_DAYOFYEAR else np.nan
    ),
    axis=1,
)
```

```python
compare.corr()["dis24"]
```

```python
compare[compare["time"] == "2022-10-01"]
```

```python
peaks = compare.groupby(compare["time"].dt.year).max().reset_index(drop=True)
peaks["year"] = peaks["time"].dt.year
```

```python
fs_col = "roll7_late"

fig, ax = plt.subplots(dpi=200)
peaks.plot.scatter(x="dis24", y=fs_col, ax=ax)
for year, row in peaks.set_index("year").iterrows():
    flip_years = []
    ha = "right" if year in flip_years else "left"
    ax.annotate(
        f" {year} ",
        (row["dis24"], row[fs_col]),
        color="k",
        fontsize=8,
        va="center",
        ha=ha,
    )
    ax.set_ylabel("FS after Oct.")
    ax.set_xlabel("GloFAS reanalysis")
    ax.set_title("Yearly peaks comparison")
```

```python
fs_col = "roll7"

fig, ax = plt.subplots(dpi=200)
peaks.plot.scatter(x="dis24", y=fs_col, ax=ax)
for year, row in peaks.set_index("year").iterrows():
    flip_years = []
    ha = "right" if year in flip_years else "left"
    ax.annotate(
        f" {year} ",
        (row["dis24"], row[fs_col]),
        color="k",
        fontsize=8,
        va="center",
        ha=ha,
    )
    ax.set_ylabel("FS any time")
    ax.set_xlabel("GloFAS reanalysis")
    ax.set_title("Yearly peaks comparison")
```

```python
21 / 5
```

```python
def highlight_true(val):
    if isinstance(val, bool) and val is True:
        return "background-color: crimson"
    return ""


df_plot = peaks.copy()
df_plot["rank"] = df_plot["dis24"].rank(ascending=False).astype(int)

cols = ["dis24", "roll7", "roll7_late"]

target_rp = 5

for x in cols:
    thresh = df_plot[x].quantile(1 - 1 / target_rp)
    df_plot[f"{x}_trig"] = df_plot[x] > thresh

df_plot = df_plot.sort_values("dis24", ascending=False)
df_plot = df_plot.rename(
    columns={
        "dis24_trig": "GF reanalysis",
        "roll7_trig": "FS",
        "roll7_late_trig": "FS (after Oct)",
    }
)
df_plot.set_index("rank")[
    ["year", "GF reanalysis", "FS", "FS (after Oct)"]
].style.map(highlight_true)
```

```python
fs_col = "roll7"

df_plot = compare.copy()
for x in ["dis24", fs_col]:
    thresh = peaks[x].quantile(1 - 1 / target_rp)
    df_plot[f"{x}_norm"] = df_plot[x] / thresh

max_y = df_plot[[f"{fs_col}_norm", "dis24_norm"]].max().max()

for year, group in df_plot.groupby(compare["time"].dt.year):
    fig, ax = plt.subplots()
    group.plot(x="time", y=["dis24_norm", f"{fs_col}_norm"], ax=ax)
    ax.set_ylim(0, max_y)
    ax.axhspan(
        1,
        max_y,
        color="red",
        alpha=0.05,
    )
    ax.legend(["GloFAS reanalysis", "Floodscan"])
    plt.show()
    plt.close()
```

```python
fs_q
```

```python
seasonal = (
    compare.groupby("dayofyear")[["roll7", "dis24"]].mean().reset_index()
)
```

```python
pd.to_datetime("2000-01-01")
```

```python
seasonal["eff_date"] = pd.to_datetime(
    seasonal["dayofyear"], format="%j", errors="coerce"
).apply(lambda x: x.replace(year=2000))
```

```python
ax = seasonal.plot(x="eff_date", y="roll7")
ax.set_ylim(bottom=0)
ax.set_ylabel("Seasonally average flooded fraction")
ax.set_xlabel("Date of year")
ax.set_title("Seasonal average flooded fraction of N'Djamena (2003-2023)")
```

```python
ax = seasonal.plot(x="eff_date", y="dis24")
ax.set_ylim(bottom=0)
ax.set_ylabel("Seasonally average flooded fraction")
ax.set_xlabel("Date of year")
ax.set_title("Seasonal average flooded fraction of N'Djamena (2003-2023)")
```

```python
means = compare.groupby(compare["time"].dt.year).mean().reset_index(drop=True)
means["year"] = means["time"].dt.year
means.plot.scatter(x="dis24", y="roll7")
```

```python

```
