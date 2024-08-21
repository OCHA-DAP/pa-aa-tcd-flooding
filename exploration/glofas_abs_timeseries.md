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

# GloFAS absolute timeseries

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import *
from src.datasources import glofas
```

```python
ref_ens = glofas.load_reforecast_ensembles()
ref = (
    ref_ens.groupby(["time", "valid_time"])[["dis24", "leadtime"]]
    .mean()
    .reset_index()
)
ref["leadtime"] = ref["leadtime"].astype(int)
max_complete_lt = 45
ref = ref[ref["leadtime"] <= max_complete_lt]
```

```python
ref
```

```python
rp_a = 5

rea = glofas.load_reanalysis()
rea = rea[rea["time"].dt.year.isin(ref["time"].dt.year.unique())]
rea_peaks = rea.loc[rea.groupby(rea["time"].dt.year)["dis24"].idxmax()]
q_rp_a = rea_peaks["dis24"].quantile(1 - 1 / rp_a)
rea_peaks["trigger"] = rea_peaks["dis24"] > q_rp_a
rea_peaks["year"] = rea_peaks["time"].dt.year
rea_peaks["cerf"] = rea_peaks["year"].isin(CERF_YEARS)
rea_peaks["rank"] = rea_peaks["dis24"].rank(ascending=False)
rea_peaks["rp"] = len(rea_peaks) / rea_peaks["rank"]
rea_peaks.sort_values("rank")
```

```python
rp_f = 5
lt_min = 5
LT_MAX = 30

val_col = "dis24"

dfs = []
dfs_threshs = []

for lt in ref["leadtime"].unique():
    if lt < lt_min or lt >= LT_MAX:
        continue

    dff = ref[(ref["leadtime"] <= lt) & (ref["leadtime"] >= lt_min)]
    df_in = dff.loc[dff.groupby(dff["time"].dt.year)[val_col].idxmax()]
    df_in["lt_max"] = lt
    thresh = df_in[val_col].quantile(1 - 1 / rp_f)
    df_in["trigger"] = df_in[val_col] >= thresh
    # print(lt, thresh)
    dfs_threshs.append({"lt_max": lt, "thresh": thresh})
    dfs.append(df_in)

ref_threshs = pd.DataFrame(dfs_threshs)
ref_peaks = pd.concat(dfs, ignore_index=True)
ref_peaks["year"] = ref_peaks["time"].dt.year
```

```python
compare = (
    ref.drop(columns=["time"])
    .rename(columns={"valid_time": "time"})
    .merge(rea, on="time", suffixes=("_f", "_a"))
    .merge(ref_threshs.rename(columns={"lt_max": "leadtime"}))
)
compare["P"] = compare["dis24_a"] > q_rp_a
compare["PP"] = compare["dis24_f"] > compare["thresh"]
compare["TP"] = compare["P"] & compare["PP"]
compare["TN"] = ~compare["P"] & ~compare["PP"]
compare["FP"] = ~compare["P"] & compare["PP"]
compare["FN"] = compare["P"] & ~compare["PP"]
```

```python
metrics = (
    compare.drop(columns=["time", "dis24_f", "dis24_a", "thresh"])
    .groupby("leadtime")
    .sum()
    .reset_index()
)
metrics["TPR"] = metrics["TP"] / metrics["P"]
metrics["PPV"] = metrics["TP"] / metrics["PP"]
```

```python
metrics.plot(x="leadtime", y=["TPR", "PPV"])
```

```python
ref.groupby("leadtime")["dis24"].mean().plot()
```

```python
ref.groupby("valid_time").size().max()
```

```python
ref = (
    ref.groupby("valid_time")
    .apply(identify_complete_valid_times, include_groups=False)
    .reset_index(level=0)
)
```

```python
ref["complete"].mean()
```

```python
def identify_complete_valid_times(group):
    group["complete"] = len(group) == 15
    return group
```

```python
ref[ref["complete"]].groupby("leadtime")["dis24"].mean().plot()
```

```python

```
