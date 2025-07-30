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

# DRE

Direction des ressources en eau

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src.datasources import dre
```

```python
df = dre.open_dre_obsv()
```

```python
df.dtypes
```

```python
df.set_index("Date").plot()
```

```python
MIN_YEAR = 2003
MAX_YEAR = 2022
```

```python
peaks = df.loc[
    df[(df["Date"].dt.year >= MIN_YEAR) & (df["Date"].dt.year <= MAX_YEAR)]
    .groupby(df["Date"].dt.year)["level_cm"]
    .idxmax()
]
peaks["year"] = peaks["Date"].dt.year
peaks["rank"] = peaks["level_cm"].rank(ascending=False)
peaks["rp"] = len(peaks) / peaks["rank"]
peaks = peaks.sort_values("rank")
```

```python
peaks
```

```python

```
