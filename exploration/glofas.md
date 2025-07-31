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

# GloFAS

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
# glofas.process_reanalysis()
```

```python
# glofas.process_reforecast_ensembles(skip_lt_groups=["lt1104-1104"])
```

```python
glofas.process_reforecast_frac()
```

```python
ens = glofas.load_reforecast_ensembles()
ens["leadtime"] -= 1
```

```python
ens
```

```python
ens_mean = (
    ens.groupby(["time", "valid_time", "leadtime"])["dis24"]
    .mean()
    .reset_index()
    .rename(columns={"dis24": "dis24_mean"})
)
ens_mean
```

```python
ref = glofas.load_reforecast_frac()
# We subtract one from the leadtime because in the
# GloFAS interface, we presume the first box corresponds to
# the smallest leadtime.
# So, the valid_time here is one day later than the actual valid time,
# which is reported in the GloFAS interface.
ref["leadtime"] -= 1
```

```python
ref
```

```python
ref.groupby("leadtime").size()
```

```python
n_years = 20
```

```python
max_complete_lt = 44
ref = ref[ref["leadtime"] <= max_complete_lt]
```

```python
rea = glofas.load_reanalysis()
```

```python
rp_a = 4
rp_a_eff = (n_years - 1) / ((n_years + 1) / rp_a - 1)

rea = glofas.load_reanalysis()
rea = rea[rea["time"].dt.year.isin(ref["time"].dt.year.unique())]
rea_peaks = rea.loc[rea.groupby(rea["time"].dt.year)["dis24"].idxmax()]
q_rp_a = rea_peaks["dis24"].quantile(1 - 1 / rp_a_eff)
rea_peaks["trigger"] = rea_peaks["dis24"] >= q_rp_a
rea_peaks["year"] = rea_peaks["time"].dt.year
rea_peaks["cerf"] = rea_peaks["year"].isin(CERF_YEARS)
rea_peaks["rank"] = rea_peaks["dis24"].rank(ascending=False)
rea_peaks["rp"] = len(rea_peaks) / rea_peaks["rank"]
rea_peaks.sort_values("rank", ascending=False)
```

```python
rea_peaks.plot(x="year", y="dis24")
```

## Readiness

```python
n_years = 20
rp_ceiling = 3.5
rp_effective = (n_years - 1) / ((n_years + 1) / rp_ceiling - 1)
print(rp_ceiling)
print(rp_effective)
```

```python
21 / 6
```

```python
rp_f = rp_effective
lt_min = 0
thresh_yr = 2
maxmin = "_mean"

val_col = f"{thresh_yr}yr_thresh{maxmin}"

dfs = []
dfs_threshs = []
dfs_first_trigger = []
dfs_last_trigger = []

for lt in ref["leadtime"].unique():
    if lt < lt_min or lt >= 30:
        continue

    dff = ref[(ref["leadtime"] <= lt) & (ref["leadtime"] >= lt_min)]
    df_in = dff.loc[dff.groupby(dff["time"].dt.year)[val_col].idxmax()]
    df_in["lt_max"] = lt
    thresh = df_in[val_col].quantile(1 - 1 / rp_f) - 0.00001
    df_in["trigger"] = df_in[val_col] >= thresh
    # print(lt, thresh)
    df_first_trigger = dff.loc[
        dff[dff[val_col] >= thresh]
        .groupby(dff["time"].dt.year)["time"]
        .idxmin()
    ]
    df_last_trigger = dff.loc[
        dff[dff[val_col] >= thresh]
        .groupby(dff["time"].dt.year)["time"]
        .idxmax()
    ]
    df_first_trigger["lt_max"] = lt
    df_last_trigger["lt_max"] = lt
    dfs_threshs.append({"lt_max": lt, "thresh": thresh})
    dfs.append(df_in)
    dfs_first_trigger.append(df_first_trigger)
    dfs_last_trigger.append(df_last_trigger)

ref_threshs = pd.DataFrame(dfs_threshs)
ref_peaks = pd.concat(dfs, ignore_index=True)
ref_peaks["year"] = ref_peaks["time"].dt.year
ref_first_triggers = pd.concat(dfs_first_trigger, ignore_index=True)
ref_last_triggers = pd.concat(dfs_last_trigger, ignore_index=True)

compare = rea_peaks.merge(ref_peaks, on="year", suffixes=["_a", "_f"])
for indicator in ["cerf", "trigger_a"]:
    compare[f"TP_{indicator}"] = compare[indicator] & compare["trigger_f"]
    compare[f"FP_{indicator}"] = ~compare[indicator] & compare["trigger_f"]
    compare[f"TN_{indicator}"] = ~compare[indicator] & ~compare["trigger_f"]
    compare[f"FN_{indicator}"] = compare[indicator] & ~compare["trigger_f"]

compare = compare.sort_values(["year", "lt_max"])

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
display(metrics.merge(ref_threshs))
```

```python
max_lt = 15

# 3yr
rp_a_3 = rea_peaks["dis24"].quantile(1 - 1 / rp_a_eff)
rp_a_target = rea_peaks["dis24"].quantile(1 - 1 / rp_a_eff)

rp_f = ref_threshs.set_index("lt_max").loc[max_lt, "thresh"] * 100
compare_lt = compare[compare["lt_max"] == max_lt].copy()
compare_lt["percent"] = compare_lt[val_col] * 100
fig, ax = plt.subplots(dpi=300)
compare_lt.plot(
    y="dis24",
    x="percent",
    ax=ax,
    marker=".",
    color="k",
    linestyle="",
    legend=False,
)

ax.axvline(x=rp_f, color="dodgerblue", linestyle="-", linewidth=0.3)
ax.axvspan(
    rp_f,
    100,
    ymin=0,
    ymax=1,
    color="dodgerblue",
    alpha=0.1,
)

ax.axhline(y=rp_a_target, color="red", linestyle="-", linewidth=0.3)
ax.axhspan(
    rp_a_target,
    8000,
    color="red",
    alpha=0.05,
    linestyle="None",
)

for year, row in compare_lt.set_index("year").iterrows():
    flip_years = [2018, 2011, 2008]
    ha = "right" if year in flip_years else "left"
    ax.annotate(
        f" {year} ",
        (row["percent"], row["dis24"]),
        color="k",
        fontsize=8,
        va="center",
        ha=ha,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Réanalyse (m$^3$/s)")
ax.set_xlabel(
    f"Prévisions (% supérieur à période de retour "
    f"{thresh_yr} ans, délai {lt_min}-{max_lt} jours)"
)
ax.set_ylim(top=7000)
ax.set_xlim(right=100, left=0)
ax.set_title("Fleuve Chari à N'Djamena\nPics annuels GloFAS (2003-2022)")
```

```python
ref_first_triggers[ref_first_triggers["lt_max"] == max_lt]
```

```python
ref_last_triggers[ref_last_triggers["lt_max"] == max_lt]
```

```python
ref[ref["time"] == "2010-08-28"].merge(ens_mean).sort_values(
    ["5yr_thresh", "dis24_mean"], ascending=False
)
```

## Action

```python
n_years = 20
rp_ceiling = 4
rp_effective = (n_years - 1) / ((n_years + 1) / rp_ceiling - 1)
print(rp_ceiling)
print(rp_effective)
```

```python
rp_f = rp_effective
lt_min = 0
thresh_yr = 2
maxmin = ""

val_col = f"{thresh_yr}yr_thresh{maxmin}"

dfs = []
dfs_threshs = []
dfs_first_trigger = []
dfs_last_trigger = []

for lt in ref["leadtime"].unique():
    if lt < lt_min or lt >= 30:
        continue

    dff = ref[(ref["leadtime"] <= lt) & (ref["leadtime"] >= lt_min)]
    df_in = dff.loc[dff.groupby(dff["time"].dt.year)[val_col].idxmax()]
    df_in["lt_max"] = lt
    thresh = df_in[val_col].quantile(1 - 1 / rp_f)
    df_in["trigger"] = df_in[val_col] >= thresh
    # print(lt, thresh)
    df_first_trigger = dff.loc[
        dff[dff[val_col] >= thresh]
        .groupby(dff["time"].dt.year)["time"]
        .idxmin()
    ]
    df_last_trigger = dff.loc[
        dff[dff[val_col] >= thresh]
        .groupby(dff["time"].dt.year)["time"]
        .idxmax()
    ]
    df_first_trigger["lt_max"] = lt
    df_last_trigger["lt_max"] = lt
    dfs_threshs.append({"lt_max": lt, "thresh": thresh})
    dfs.append(df_in)
    dfs_first_trigger.append(df_first_trigger)
    dfs_last_trigger.append(df_last_trigger)

ref_threshs = pd.DataFrame(dfs_threshs)
ref_peaks = pd.concat(dfs, ignore_index=True)
ref_peaks["year"] = ref_peaks["time"].dt.year
ref_first_triggers = pd.concat(dfs_first_trigger, ignore_index=True)
ref_last_triggers = pd.concat(dfs_last_trigger, ignore_index=True)

compare = rea_peaks.merge(ref_peaks, on="year", suffixes=["_a", "_f"])
for indicator in ["cerf", "trigger_a"]:
    compare[f"TP_{indicator}"] = compare[indicator] & compare["trigger_f"]
    compare[f"FP_{indicator}"] = ~compare[indicator] & compare["trigger_f"]
    compare[f"TN_{indicator}"] = ~compare[indicator] & ~compare["trigger_f"]
    compare[f"FN_{indicator}"] = compare[indicator] & ~compare["trigger_f"]

compare = compare.sort_values(["year", "lt_max"])

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
display(metrics.merge(ref_threshs))
```

```python
max_lt = 10

# 3yr
rp_a_target = rea_peaks["dis24"].quantile(1 - 1 / rp_a_eff)

rp_f = ref_threshs.set_index("lt_max").loc[max_lt, "thresh"] * 100
compare_lt = compare[compare["lt_max"] == max_lt].copy()
compare_lt["percent"] = compare_lt[val_col] * 100
fig, ax = plt.subplots(dpi=300)
compare_lt.plot(
    y="dis24",
    x="percent",
    ax=ax,
    marker=".",
    color="k",
    linestyle="",
    legend=False,
)

ax.axvline(x=rp_f, color="dodgerblue", linestyle="-", linewidth=0.3)
ax.axvspan(
    rp_f,
    100,
    ymin=0,
    ymax=1,
    color="dodgerblue",
    alpha=0.1,
)
ax.annotate(
    f"Seuil période de retour {rp_a}-ans",
    (1, rp_a_target),
    fontsize=8,
    color="crimson",
)
ax.annotate(
    f"Seuil période de retour {rp_ceiling}-ans",
    (rp_f, compare_lt["dis24"].min()),
    fontsize=8,
    color="dodgerblue",
    rotation=90,
    ha="right",
)

ax.axhline(y=rp_a_target, color="crimson", linestyle="-", linewidth=0.3)
ax.axhspan(
    rp_a_target,
    8000,
    color="crimson",
    alpha=0.05,
    linestyle="None",
)

for year, row in compare_lt.set_index("year").iterrows():
    flip_years = [2006]
    ha = "right" if year in flip_years else "left"
    ax.annotate(
        f" {year} ",
        (row["percent"], row["dis24"]),
        color="k",
        fontsize=8,
        va="center",
        ha=ha,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Réanalyse (m$^3$/s)")
ax.set_xlabel(
    f"Prévisions (% supérieur à période de retour "
    f"{thresh_yr} ans, délai {lt_min}-{max_lt} jours)"
)
ax.set_ylim(top=7000)
ax.set_xlim(right=100, left=0)
ax.set_title("Fleuve Chari à N'Djamena\nPics annuels GloFAS (2003-2022)")
```

```python
ref_first_triggers[ref_first_triggers["lt_max"] == max_lt]
```

```python
ref_last_triggers[ref_last_triggers["lt_max"] == max_lt]
```

```python
ref[ref["time"] == "2010-09-04"].merge(ens_mean).sort_values(
    ["5yr_thresh", "dis24_mean"], ascending=False
)
```

```python
cols = [x for x in compare.columns if compare[x].dtype != bool]
compare[compare["lt_max"] == max_lt][cols]
```

```python
ref
```

```python
dicts = []
for lt_max, group in compare.groupby("lt_max"):
    corr_in = group.corr()
    dicts.append(
        {
            "lt_max": lt_max,
            "2yr_thresh": corr_in.loc["dis24", "2yr_thresh"],
            "5yr_thresh": corr_in.loc["dis24", "5yr_thresh"],
        }
    )

df_corr = pd.DataFrame(dicts)
df_corr
```

```python
df_corr.set_index("lt_max").plot()
```

```python
rea
```

```python

```
