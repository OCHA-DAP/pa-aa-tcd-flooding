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

# DRE Bongor
<!-- markdownlint-disable MD013 -->
Observational data from DRE, Bongor station

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ocha_stratus as stratus

from src.datasources import dre, glofas
from src.constants import *
```

```python
df = dre.open_dre_obsv("bongor", from_blob=True)
```

```python
df
```

```python
df.plot(x="Date", y="level_cm")
```

```python
df[df.duplicated(subset="Date")]
```

```python
df.dtypes
```

```python
# Ensure datetime
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["DayOfYear"] = df["Date"].dt.dayofyear

# Full date range
full_range = pd.DataFrame(
    {"Date": pd.date_range(df["Date"].min(), df["Date"].max())}
)
full_range["Year"] = full_range["Date"].dt.year
full_range["DayOfYear"] = full_range["Date"].dt.dayofyear

# Merge and create 'Present' boolean
df_present = df[["Date", "level_cm"]].copy()
df_present["Present"] = ~df_present["level_cm"].isna()
merged = full_range.merge(
    df_present[["Date", "Present"]], on="Date", how="left"
)

# Fill missing dates as not present
merged["Present"] = merged["Present"].fillna(False)

# Group to ensure uniqueness
grouped = (
    merged.groupby(["Year", "DayOfYear"])["Present"]
    .any()
    .unstack()
    .astype(float)
)

# Plot
fig, ax = plt.subplots(figsize=(15, 8))
im = ax.imshow(grouped, aspect="auto", cmap="Greys", interpolation="none")

ax.set_title("Data Availability (1 = Present, 0 = Missing)", fontsize=14)
ax.set_xlabel("Day of Year")
ax.set_ylabel("Year")
ax.set_yticks(np.arange(0, len(grouped.index), 5))
ax.set_yticklabels(grouped.index[::5])
ax.set_xticks([0, 59, 120, 181, 243, 304, 364])
ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Dec"])
plt.colorbar(im, ax=ax, label="Data Present")

plt.tight_layout()
plt.show()
```

```python
df_peaks = df.loc[df.dropna().groupby(df["Date"].dt.year)["level_cm"].idxmax()]
```

```python
df_peaks
```

```python
df_peaks["date_1900"] = df_peaks["Date"].apply(
    lambda x: pd.Timestamp(year=1900, month=x.month, day=x.day)
)
```

```python
df_peaks.plot(x="Year", y="date_1900")
```

```python
df_peaks.plot(x="Year", y="level_cm")
```

```python
df_peaks.plot.scatter(x="date_1900", y="level_cm")
```

```python
df_rea = glofas.load_glofas_reanalysis("bongor")
```

```python
query = f"""
SELECT pcode, valid_date, mean
FROM public.floodscan
WHERE pcode = '{MAYOBONEYE2}'
AND band = 'SFED'
"""
df_fs = pd.read_sql(
    query, stratus.get_engine(stage="prod"), parse_dates=["valid_date"]
)
```

```python
df_fs
```

```python
df_daily = df_rea.merge(df.rename(columns={"Date": "time"}), how="left").merge(
    df_fs.rename(columns={"valid_date": "time", "mean": "fs_mean"}), how="left"
)
```

```python
df_daily
```

```python
# Ensure datetime
df_daily["time"] = pd.to_datetime(df_daily["time"])
df_daily["year"] = df_daily["time"].dt.year
df_daily["doy"] = df_daily["time"].dt.dayofyear

# Normalize each variable across all data (relative to column max)
df_daily["dis24_norm"] = df_daily["dis24"] / df_daily["dis24"].max()
df_daily["level_cm_norm"] = df_daily["level_cm"] / df_daily["level_cm"].max()
df_daily["fs_mean_norm"] = df_daily["fs_mean"] / df_daily["fs_mean"].max()

# Select years to plot
years = sorted(df_daily["year"].unique())

# Create subplots
fig, axes = plt.subplots(
    nrows=len(years), ncols=1, figsize=(12, 2.5 * len(years)), sharex=True
)

if len(years) == 1:
    axes = [axes]  # ensure iterable

# Plot for each year
for ax, year in zip(axes, years):
    data = df_daily[df_daily["year"] == year]

    ax.plot(
        data["doy"], data["dis24_norm"], label="dis24", color="blue", alpha=0.6
    )
    ax.plot(
        data["doy"],
        data["level_cm_norm"],
        label="level_cm",
        color="green",
        alpha=0.6,
    )
    ax.plot(
        data["doy"],
        data["fs_mean_norm"],
        label="fs_mean",
        color="orange",
        alpha=0.6,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(str(year), rotation=0, labelpad=30)

axes[-1].set_xlabel("Day of Year")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
```

```python
df_fs
```

```python
df_fs_peaks = (
    df_fs.groupby(df_fs["valid_date"].dt.year)["mean"].max().reset_index()
)
```

```python
df_fs_peaks[df_fs_peaks["valid_date"] >= 2003].sort_values(
    "mean", ascending=False
)
```
