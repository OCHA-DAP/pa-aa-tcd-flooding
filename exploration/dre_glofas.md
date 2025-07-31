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

# DRE-GloFAS comparison

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.datasources import dre, glofas
```

```python
MIN_YEAR = 2003
```

```python
df_dre = dre.open_dre_obsv()
df_dre = df_dre.rename(columns={"Date": "time"})
df_dre["dayofyear"] = df_dre["time"].dt.dayofyear
df_dre["eff_date"] = pd.to_datetime(df_dre["dayofyear"], format="%j")
df_dre = df_dre[df_dre["time"].dt.year >= MIN_YEAR]
df_dre = df_dre.dropna()
df_dre = df_dre.drop_duplicates()
# df_dre = df_dre.drop_duplicates(keep=False, subset="time")
```

```python
for year in df_dre["time"].dt.year.unique():
    fig, ax = plt.subplots()
    df_dre[df_dre["time"].dt.year == year].set_index("time").plot(ax=ax)
    plt.show()
    plt.close()
```

```python
DROP_YEARS = [2006, 2008, 2023, 2024]
```

```python
df_dre = df_dre[~df_dre["time"].dt.year.isin(DROP_YEARS)]
```

```python
peaks_dre = df_dre.loc[
    df_dre.groupby(df_dre["time"].dt.year)["level_cm"].idxmax()
]
peaks_dre["year"] = peaks_dre["time"].dt.year
peaks_dre["rank"] = peaks_dre["level_cm"].rank(ascending=False)
peaks_dre["rp"] = len(peaks_dre) / peaks_dre["rank"]
peaks_dre = peaks_dre.sort_values("rank")
```

```python
len(peaks_dre)
```

```python
df_gf = glofas.load_reanalysis()
df_gf["dayofyear"] = df_gf["time"].dt.dayofyear
df_gf["eff_date"] = pd.to_datetime(df_gf["dayofyear"], format="%j")
df_gf = df_gf[~df_gf["time"].dt.year.isin(DROP_YEARS)]
```

```python
df_gf
```

```python
peaks_gf = df_gf.loc[df_gf.groupby(df_gf["time"].dt.year)["dis24"].idxmax()]
peaks_gf["year"] = peaks_gf["time"].dt.year
peaks_gf["rank"] = peaks_gf["dis24"].rank(ascending=False)
peaks_gf["rp"] = len(peaks_gf) / peaks_gf["rank"]
peaks_gf = peaks_gf.sort_values("rank")
```

```python
len(peaks_gf)
```

```python
peaks = peaks_gf.merge(peaks_dre, on="year", suffixes=("_gf", "_dre"))
peaks["days_early"] = peaks["dayofyear_dre"] - peaks["dayofyear_gf"]
peaks = peaks.sort_values("rank_dre")
```

```python
peaks
```

```python
19 / 5
```

```python
len(peaks)
```

```python
rp = 4

fig, ax = plt.subplots(dpi=300)
peaks.plot(
    x="level_cm",
    y="dis24",
    ax=ax,
    marker=".",
    color="k",
    linestyle="",
    legend=False,
)

ax.axhline(
    y=peaks["dis24"].quantile(1 - 1 / rp),
    color="crimson",
    linestyle="-",
    linewidth=0.3,
)
ax.axhspan(
    peaks["dis24"].quantile(1 - 1 / rp),
    10000,
    color="crimson",
    alpha=0.05,
    linestyle="None",
)
ax.annotate(
    f"Seuil période de retour {rp}-ans",
    (peaks["level_cm"].min() - 20, peaks["dis24"].quantile(1 - 1 / rp)),
    fontsize=8,
    color="crimson",
)
ax.axvline(
    x=peaks["level_cm"].quantile(1 - 1 / rp),
    color="limegreen",
    linestyle="-",
    linewidth=0.3,
)
ax.axvspan(
    peaks["level_cm"].quantile(1 - 1 / rp),
    1000,
    ymin=0,
    ymax=1,
    color="limegreen",
    alpha=0.1,
)
ax.annotate(
    f"Seuil période de retour {rp}-ans",
    (peaks["level_cm"].quantile(1 - 1 / rp), peaks["dis24"].min() + 2000),
    fontsize=8,
    color="limegreen",
    rotation=90,
    ha="right",
)

for year, row in peaks.set_index("year").iterrows():
    flip_years = [2011]
    ha = "right" if year in flip_years else "left"
    ax.annotate(
        f" {year} ",
        (row["level_cm"], row["dis24"]),
        color="k",
        fontsize=8,
        va="center",
        ha=ha,
    )

ax.set_ylim(top=7000)
ax.set_xlim(right=850)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Réanalyse (m$^3$/s)")
ax.set_xlabel("Niveau du fleuve observationel (cm)")
ax.set_title(
    "Fleuve Chari à N'Djamena\nPics annuels GloFAS vs. observationnel (2003-2022)"
)
```

```python
dre_thresh = peaks["level_cm"].quantile(1 - 1 / 5)
print(f"{dre_thresh=}")
df_dre.loc[
    df_dre[df_dre["level_cm"] >= dre_thresh]
    .groupby(df_dre["time"].dt.year)["time"]
    .idxmin()
]
```

```python
len(peaks)
```

```python
peaks = peaks.sort_values("rp_dre")
np.interp(750, peaks["level_cm"], peaks["rp_dre"]) * (len(peaks) + 1) / len(
    peaks
)
```

```python
df_dre[df_dre["time"].dt.year == 2022].plot(x="time", y="level_cm")
```

```python
fig, ax = plt.subplots(dpi=300)
peaks.plot.scatter(x="level_cm", y="days_early", ax=ax)
ax.set_ylabel("Days reanalysis peak precedes\nobservational peak")
ax.set_xlabel("Observational yearly peak (cm)")
```

```python
peaks["days_early"].mean()
```

```python
df = df_gf.merge(df_dre)
```

```python
for year in df["time"].dt.year.unique():
    fig, ax = plt.subplots()
    df[df["time"].dt.year == year].set_index("time").plot(
        y=["dis24", "level_cm"], ax=ax
    )
    plt.show()
    plt.close()
```

```python
df.corr().loc["dis24", "level_cm"]
```

```python
peaks.plot.scatter(x="level_cm", y="dis24")
```

```python

```
