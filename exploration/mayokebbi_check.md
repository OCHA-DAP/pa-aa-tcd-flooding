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

# Check Mayo-Kebbi

Make sure that Mayo-Kebbi flood exposure is correlated enough with N'Djamena

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.constants import *
```

```python
def detrend_column(
    df: pd.DataFrame,
    col: str,
    index_col: str = "valid_date",
    min_index=None,
    max_index=None,
) -> pd.DataFrame:
    """
    Detrend a column in a DataFrame using linear regression (via NumPy).

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame. Must contain a datetime column.
    col : str
        The name of the column to detrend.
    time_col : str
        The name of the datetime column. Default is "valid_date".

    Returns:
    --------
    pd.DataFrame
        Copy of the input DataFrame with a new column: <col>_detrended
    """
    if min_index is None:
        min_index = df[index_col].min()
    if max_index is None:
        max_index = df[index_col].max()

    df_sorted = df.sort_values(index_col).copy()
    df_model = df_sorted[
        (df_sorted[index_col] >= min_index)
        & (df_sorted[index_col] <= max_index)
    ]

    x = df_model[index_col]
    y = df_model[col].values

    # Linear regression fit
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    trend = a * df_sorted[index_col] + b
    detrended = df_sorted[col] - trend
    detrended += y.mean()  # Shift to preserve original mean

    df_sorted[f"{col}_detrended"] = detrended

    return df_sorted
```

```python
query = """
SELECT *
FROM app.floodscan_exposure
WHERE pcode = '{pcode}'
"""
```

```python
dfs = []
for pcode in [NDJAMENA1, MAYOKEBBIEST1]:
    df_in = pd.read_sql(
        query.format(pcode=pcode),
        stratus.get_engine(stage="prod"),
        parse_dates=["valid_date"],
    )
    dfs.append(df_in)

df_exp = pd.concat(dfs, ignore_index=True)
```

```python
df_exp.dtypes
```

```python
df_exp_yearly = (
    df_exp.groupby(["pcode", df_exp["valid_date"].dt.year])["sum"]
    .max()
    .reset_index()
)
```

```python
df_plot.plot()
```

```python
df_plot = df_exp_yearly.pivot(
    values="sum", columns="pcode", index="valid_date"
)
```

```python
df_plot
```

```python
df_plot_detrend = detrend_column(df_plot.reset_index(), NDJAMENA1)
df_plot_detrend = detrend_column(df_plot_detrend, MAYOKEBBIEST1).set_index(
    "valid_date"
)
```

```python
df_plot_detrend
```

```python
fig, ax = plt.subplots(figsize=(7, 7))

df_plot_detrend.plot(
    x=NDJAMENA1 + "_detrended",
    y=MAYOKEBBIEST1 + "_detrended",
    ax=ax,
    linewidth=0,
    legend=False,
)

for year, row in df_plot_detrend.iterrows():
    ax.annotate(
        year,
        (row[NDJAMENA1 + "_detrended"], row[MAYOKEBBIEST1 + "_detrended"]),
    )
```

```python
fig, ax = plt.subplots(figsize=(7, 7))

df_plot_detrend.plot(
    x=NDJAMENA1,
    y=MAYOKEBBIEST1,
    ax=ax,
    linewidth=0,
    legend=False,
)

for year, row in df_plot_detrend.iterrows():
    ax.annotate(
        year,
        (row[NDJAMENA1], row[MAYOKEBBIEST1]),
    )
```

```python
df_plot_detrend.corr()
```

```python
df_plot_detrend.plot()
```
