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

from src.constants import *
from src.datasources import glofas
```

```python
glofas.process_reanalysis()
```

```python
rea = glofas.load_reanalysis()
```

```python
rea_peaks = rea.loc[rea.groupby(rea["time"].dt.year)["dis24"].idxmax()]
rea_peaks["year"] = rea_peaks["time"].dt.year
```

```python
rea_peaks.plot(x="year", y="dis24")
```
