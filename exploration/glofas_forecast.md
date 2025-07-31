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

# GloFAS forecast

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import pandas as pd

from src.datasources import glofas
from src.constants import *
```

```python
filename = "glofas-forecast-test-2024-09-24.grib"
da = xr.load_dataset(glofas.GF_TEST_DIR / filename)["dis24"]
```

```python
df = (
    da.to_dataframe()["dis24"]
    .reset_index()
    .drop(columns=["latitude", "longitude"])
)
```

```python
lt2frac = {
    "2024-09-24": [
        0,
        69,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        92,
        76,
        69,
        63,
        71,
        80,
        84,
        82,
        82,
        82,
        80,
        80,
        75,
        75,
        73,
        71,
        69,
        67,
        65,
        65,
        57,
    ],
    "2024-09-23": [
        0,
        0,
        59,
        80,
        73,
        63,
        49,
        43,
        37,
        37,
        37,
        37,
        35,
        41,
        53,
        59,
        67,
        65,
        67,
        69,
        69,
        71,
        67,
        67,
        63,
        57,
        57,
        49,
        47,
        43,
    ],
}
```

```python
df
```

```python
dicts = []
dates = ["2024-09-23", "2024-09-24"]
for date in dates:
    for lt, q in enumerate(lt2frac.get(date)):
        # if lt > 15:
        #     continue
        dff = df[(df["time"] == date) & (df["step"].dt.days == lt + 1)]
        dicts.append(
            {
                "date": date,
                "lt": lt + 1,
                "thresh": dff["dis24"].quantile(1 - q / 100),
            }
        )
```

```python
df_thresh = pd.DataFrame(dicts)
df_thresh
```

```python
df_thresh["thresh"].plot()
```

```python
df_thresh["thresh"].max()
```

```python
df_thresh["thresh"].min()
```

```python
df_thresh["thresh"].mean()
```

```python
df_thresh["thresh"].median()
```

```python
df_thresh["thresh"].plot()
```

```python
thresh_2yr = df_thresh["thresh"].median()
```

```python
df["2yr_thresh"] = df["dis24"] > thresh_2yr
df["5yr_thresh"] = df["dis24"] > NDJAMENA_5YRRP
```

```python
ens = (
    df.groupby(["time", "step"])[[x for x in df.columns if "yr_thresh" in x]]
    .mean()
    .reset_index()
)
```

```python
ens_pivot = ens.pivot(index="time", columns="step")
```

```python
ens_pivot["2yr_thresh"][
    [x for x in ens_pivot["2yr_thresh"].columns if x.days <= 15]
].sort_index(ascending=False)
```

```python
ens_pivot["2yr_thresh"][
    [x for x in ens_pivot["2yr_thresh"].columns if x.days > 15]
].sort_index(ascending=False)
```

```python

```
