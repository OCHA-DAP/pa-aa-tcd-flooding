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

# Download Mayo-Kebbi reanalysis

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import xarray as xr

from src.datasources import glofas
```

```python
da = xr.open_dataset("temp/glofas_pixel_check.grib")
```

```python
da.max(dim=["time"])["dis24"].plot()
```

```python
da.max(dim=["time"])["dis24"].sel(
    latitude=slice(10.29, 10.2), longitude=slice(15.4, 15.5)
).plot()
```

```python
da.max(dim=["time"])["dis24"].sel(
    latitude=slice(10.29, 10.2), longitude=slice(15.4, 15.5)
)
```

```python
glofas.download_glofas_reanalysis_to_blob("bongor")
```

```python
glofas.process_glofas_reanalysis("bongor")
```

```python
df = glofas.load_glofas_reanalysis("bongor")
```

```python
df.plot(x="time", y="dis24")
```

```python
glofas.download_glofas_reanalysis_year_to_blob(2024, "ndjamena")
```

```python
glofas.download_glofas_reanalysis_year_to_blob(2023, "ndjamena")
```

```python
glofas.download_glofas_reanalysis_to_blob("ndjamena")
```

```python
glofas.process_glofas_reanalysis("ndjamena")
```

```python
df_test = glofas.load_glofas_reanalysis("ndjamena")
```

```python
df_test.plot(x="time", y="dis24")
```

```python
da_test = glofas.load_glofas_reanalysis_year("raw", "ndjamena", 2023)
```

```python

```

```python
da_test["dis24"].plot()
```

```python
df_test = da_test.to_dataframe()
```

```python
df_test.loc[df_test["dis24"].idxmax()]
```

```python

```
