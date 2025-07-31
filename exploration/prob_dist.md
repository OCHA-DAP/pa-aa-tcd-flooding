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

# Prob distribution
<!-- markdownlint-disable MD013 -->

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

from src.constants import *
from src.datasources import glofas
```

```python
ref = glofas.load_reforecast_frac()
ref["leadtime"] -= 1
```

```python
ref
```

```python
max_lt = 10
ref_peaks = (
    ref[ref["leadtime"] <= max_lt]
    .groupby("time")["5yr_thresh"]
    .max()
    .reset_index()
)
```

```python
ref_peaks["5yr_thresh"].value_counts()
```

```python
for year in ref_peaks["time"].dt.year.unique():
    print(year)
    dff = ref_peaks[ref_peaks["time"].dt.year == year]
    print(f"actual max {dff['5yr_thresh'].max()}")

    # Assuming your DataFrame is called df and the relevant column is "5yr_thresh"
    data = dff["5yr_thresh"].copy()

    # Adjust values exactly at the boundaries
    data[data <= 0] = 0.0001
    data[data >= 1] = 0.9999
    try:
        # Fit a Beta distribution to your data
        a, b, loc, scale = beta.fit(
            data, floc=0, fscale=1
        )  # Fixing loc=0 and scale=1 for bounded data [0, 1]

        # Calculate the estimated population size based on your sample percentage X
        X = 0.1  # Replace this with your actual sample percentage as a decimal
        population_size = len(data) / X

        # Estimate the expected maximum for the population
        expected_maximum = beta.ppf(
            1 - 1 / population_size, a, b, loc=loc, scale=scale
        )

        # Output the result
        print(f"Expected Maximum for the Population: {expected_maximum}")
    except Exception as e:
        print(e)
```

```python

```
