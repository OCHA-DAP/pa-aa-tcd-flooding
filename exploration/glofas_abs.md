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

# GloFAS absolute values

Instead of probabilities (which will be used for monitoring)

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
```

```python
ref = (
    ref_ens.groupby(["time", "valid_time"])[["dis24", "leadtime"]]
    .mean()
    .reset_index()
)
ref["leadtime"] = ref["leadtime"].astype(int)
```

```python
ref.groupby("leadtime").size()
```

```python
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

val_col = "dis24"

dfs = []
dfs_threshs = []

for lt in ref["leadtime"].unique():
    if lt < lt_min or lt >= 30:
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
ref_threshs
```

```python
compare = rea_peaks.merge(ref_peaks, on="year", suffixes=["_a", "_f"])
for indicator in ["cerf", "trigger_a"]:
    compare[f"TP_{indicator}"] = compare[indicator] & compare["trigger_f"]
    compare[f"FP_{indicator}"] = ~compare[indicator] & compare["trigger_f"]
    compare[f"TN_{indicator}"] = ~compare[indicator] & ~compare["trigger_f"]
    compare[f"FN_{indicator}"] = compare[indicator] & ~compare["trigger_f"]

compare["days_error"] = (compare["time_a"] - compare["valid_time"]).dt.days
compare = compare.sort_values(["year", "lt_max"])
```

```python
dicts = []
for lt, group in compare.groupby("lt_max"):
    TPR = group["TP_trigger_a"].sum() / group["trigger_a"].sum()
    PPV = group["TP_trigger_a"].sum() / group["trigger_f"].sum()
    TPR_C = group["TP_cerf"].sum() / group["cerf"].sum()
    PPV_C = group["TP_cerf"].sum() / group["trigger_f"].sum()
    dicts.append(
        {"TPR": TPR, "PPV": PPV, "TPR_C": TPR_C, "PPV_C": PPV_C, "lt_max": lt}
    )

metrics = pd.DataFrame(dicts)
metrics
```

```python
max_lt = 15

# 3yr
rp_a_3 = rea_peaks["dis24"].quantile(1 - 1 / 3)
rp_a_target = rea_peaks["dis24"].quantile(1 - 1 / rp_a)

rp_f = ref_threshs.set_index("lt_max").loc[max_lt, "thresh"]
compare_lt = compare[compare["lt_max"] == max_lt].copy()
fig, ax = plt.subplots(dpi=300)
compare_lt.plot(
    y="dis24_a",
    x="dis24_f",
    ax=ax,
    marker=".",
    color="k",
    linestyle="",
    legend=False,
)

ax.axvline(x=rp_f, color="dodgerblue", linestyle="-", linewidth=0.3)
ax.axvspan(
    rp_f,
    8500,
    ymin=0,
    ymax=1,
    color="dodgerblue",
    alpha=0.1,
)
if rp_a <= 3:
    ax.axhline(y=rp_a_3, color="red", linestyle="-", linewidth=0.3)
    ax.axhspan(
        rp_a_3,
        8500,
        color="red",
        alpha=0.05,
        linestyle="None",
    )

ax.axhline(y=rp_a_target, color="red", linestyle="-", linewidth=0.3)
ax.axhspan(
    rp_a_target,
    8500,
    color="red",
    alpha=0.05,
    linestyle="None",
)

for year, row in compare_lt.set_index("year").iterrows():
    flip_years = [2004, 2006, 0, 2016]
    ha = "right" if year in flip_years else "left"
    ax.annotate(
        f" {year} ",
        (row["dis24_f"], row["dis24_a"]),
        color="k",
        fontsize=8,
        va="center",
        ha=ha,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Reanalysis (m$^3$/s)")
ax.set_xlabel(f"Prévisions (m$^3$/s), délai {lt_min}-{max_lt} jours")
ax.set_ylim(top=8500)
ax.set_xlim(right=8500)
ax.set_title(
    f"Fleuve Chari à N'Djamena\nPics annuels GloFAS (2003-2022),\n"
    f"période de retour {rp_a} ans"
)
```

```python
fig, ax = plt.subplots(dpi=300)
compare_lt.plot.scatter(x="dis24_a", y="days_error", ax=ax)
ax.set_ylabel("Days reforecast peak precedes\nreanalysis peak")
ax.set_xlabel("Observational yearly peak (cm)")
```

```python
ref_peaks[ref_peaks["lt_max"] == max_lt]
```

```python

```
