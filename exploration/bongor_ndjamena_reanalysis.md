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

# Bongor-N'Djamena reanalysis
<!-- markdownlint-disable MD013 -->
Comparison of reanalysis betwee Bongor and N'Djamena stations

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import ocha_stratus as stratus
import pandas as pd

from src.datasources import glofas
from src.constants import *
```

```python
# df_n = glofas.load_reanalysis(local=True)
df_n = glofas.load_glofas_reanalysis("ndjamena")
```

```python
df_n
```

```python
df_b = glofas.load_glofas_reanalysis("bongor")
```

```python
df_b
```

```python
df_compare = df_b.merge(df_n, on="time", suffixes=("_b", "_n"))
```

```python
df_compare[df_compare["time"].dt.year == 2024].plot(x="time")
```

```python
df_peaks = (
    df_compare.groupby(df_compare["time"].dt.year)
    .max()
    .drop(columns="time")
    .reset_index()
)
```

```python
df_peaks
```

```python
df_peaks.corr()
```

```python
df_peaks.plot.scatter(x="dis24_b", y="dis24_n")
```

```python
df_peaks.plot(x="time")
```

```python
fig, ax = plt.subplots(figsize=(7, 7))

df_plot = df_peaks[df_peaks["time"] >= 2000]

df_plot.plot(x="dis24_b", y="dis24_n", linewidth=0, ax=ax, legend=False)

for year, row in df_plot.set_index("time").iterrows():
    ax.annotate(
        year, (row["dis24_b"], row["dis24_n"]), ha="center", va="center"
    )

rp = 3.5
x_thresh = df_plot["dis24_b"].quantile(1 - 1 / rp)
y_thresh = df_plot["dis24_n"].quantile(1 - 1 / rp)

ax.axhline(y_thresh)
ax.axvline(x_thresh)

ax.set_xlabel("Bongor")
ax.set_ylabel("N'Djamena")
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
```

```python
def calculate_one_group_rp(group, col_name: str = "q", ascending: bool = True):
    """Calculate the empirical RP for a single group.

    Parameters
    ----------
    group : pd.DataFrame
        The group for which to calculate the RP.
    col_name : str, optional
        The name of the column for which to calculate the RP, by default "q".
    ascending : bool, optional
        Whether to rank the column in ascending order, by default True.
        Should be False for cases where a high number is severe
        (e.g. precipitation for flooding), and True for cases where a low
        number is severe (e.g. precipitation for drought).

    Returns
    -------
    pd.DataFrame
        The input group with the RP columns added.
    """
    group[f"{col_name}_rank"] = group[col_name].rank(ascending=ascending)
    group[f"{col_name}_rp"] = (len(group) + 1) / group[f"{col_name}_rank"]
    return group
```

```python
# df_peaks_recent = df_peaks[df_peaks["time"] >= 2000].copy()
df_peaks_recent = df_peaks[
    (df_peaks["time"] >= 2003) & (df_peaks["time"] <= 2024)
].copy()
```

```python
total_years = df_peaks_recent["time"].nunique()
target_rp = 3.5
target_year_count = int((total_years + 1) / target_rp)
actual_rp = (total_years + 1) / target_year_count
target_year_count, actual_rp
```

```python
for x in ["b", "n"]:
    df_peaks_recent = calculate_one_group_rp(
        df_peaks_recent, f"dis24_{x}", ascending=False
    )
```

```python
df_peaks_recent["min_rp"] = df_peaks_recent[
    [f"dis24_{x}_rp" for x in ["b", "n"]]
].min(axis=1)
df_peaks_recent["max_rp"] = df_peaks_recent[
    [f"dis24_{x}_rp" for x in ["b", "n"]]
].max(axis=1)
```

```python
def determine_trig(col):
    thresh = df_peaks_recent.sort_values(col, ascending=False).iloc[
        target_year_count - 1
    ][col]
    return df_peaks_recent[col] >= thresh
```

```python
df_peaks_recent["or_trig"] = determine_trig("max_rp")
df_peaks_recent["and_trig"] = determine_trig("min_rp")
df_peaks_recent["n_only_trig"] = determine_trig("dis24_n_rp")
df_peaks_recent["b_only_trig"] = determine_trig("dis24_b_rp")
```

```python
df_peaks_recent["indep_trig"] = df_peaks_recent[
    ["b_only_trig", "n_only_trig"]
].any(axis=1)
```

```python
df_peaks_recent.sort_values("dis24_n_rp", ascending=False)
```

```python
df_fs
```

```python
df_fs_peaks = (
    df_fs.groupby(["pcode", df_fs["valid_date"].dt.year])["mean"]
    .max()
    .reset_index()
)
```

```python
df_fs_peaks.pivot(
    index="valid_date", columns="pcode", values="mean"
).reset_index()
```

```python
df_peaks_recent = df_peaks_recent.merge(
    df_fs_peaks.pivot(index="valid_date", columns="pcode", values="mean")
    .reset_index()
    .rename(columns={"valid_date": "time"})
)
```

```python
df_peaks_recent
```

```python
trig_color = "crimson"
nontrig_color = "k"
```

```python
xcol, ycol = NDJAMENA1, MAYOKEBBIEST1
```

```python
def plot_exposure_trig(trig_col, trig_text):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)

    df_peaks_recent.plot(x=xcol, y=ycol, ax=ax, linewidth=0, legend=False)

    for year, row in df_peaks_recent.set_index("time").iterrows():
        color = trig_color if row[trig_col] else nontrig_color
        ax.annotate(year, row[[xcol, ycol]], color=color)

    ax.set_xlabel("Province N'Djamena")
    ax.set_ylabel("Province Mayo-Kebbi Est")
    fig.suptitle("Fraction inondée maximale par année", y=0.95)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    fig.text(
        0.5,
        0.90,
        f"Déclencheur : {trig_text}",
        ha="center",
        color=trig_color,
        fontstyle="italic",
        alpha=0.7,
    )
    return fig, ax
```

```python
fig, ax = plot_exposure_trig("n_only_trig", "Station N'Djamena")
```

```python
fig, ax = plot_exposure_trig("b_only_trig", "Station Bongor")
```

```python
fig, ax = plot_exposure_trig("or_trig", "Stations Bongor OU N'Djamena")
```

```python
fig, ax = plot_exposure_trig("and_trig", "Stations Bongor ET N'Djamena")
```

```python
fig, ax = plot_exposure_trig(
    "indep_trig", "Stations Bongor OU N'Djamena (indépdendant)"
)
```

```python

```

```python
df_fs_peaks.sort_values("mean", ascending=False)
```

```python

```

```python
df_peaks.merge(
    df_fs_peaks.rename(columns={"valid_date": "time", "mean": "fs_mean"}),
    how="left",
).corr()
```

```python
df_peaks
```
