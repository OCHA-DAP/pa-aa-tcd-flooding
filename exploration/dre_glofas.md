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

# DRE vs. GloFAS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
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
peaks.plot.scatter(x="level_cm", y="dis24")
```

```python
peaks.plot.scatter(x="level_cm", y="days_early")
```

```python
df = df_gf.merge(df_dre)
```

```python
for year in df["time"].dt.year.unique():
    fig, ax = plt.subplots()
    df[df["time"].dt.year == year].set_index("time").plot(ax=ax)
    plt.show()
    plt.close()
```

```python
peaks = (
    df.groupby(df["time"].dt.year)[["dis24", "level_cm"]].max().reset_index()
)
```

```python
peaks.plot.scatter(x="level_cm", y="dis24")
```

```python

```
