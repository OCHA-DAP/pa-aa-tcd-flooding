### Flooded fraction in the region of interest
This notebook repeats the analysis of `01_tcd_foodscan_country.md` but with a focus
on the priority areas which are N'Djamena and Mayo Kebbi Est

Note: this notebook is a bit less ordered

```python
%load_ext autoreload
%autoreload 2
%load_ext jupyter_black
```

```python
import numpy as np
import altair as alt
import pandas as pd
import panel.widgets as pnw
import hvplot.xarray
```

```python
from src.constants import gdf_adm1
from src.utils import load_floodscan, get_return_periods_dataframe
```

```python
%load_ext rpy2.ipython
```

```R tags=[]
library(tidyverse)
```

#### define functions

```R
plotFloodedFraction <- function (df,y_col,facet_col,title){
df %>%
ggplot(
aes_string(
x = "time",
y = y_col
)
) +
stat_smooth(
geom = "area",
span = 1/4,
fill = "#ef6666"
) +
scale_x_date(
date_breaks = "3 months",
date_labels = "%b"
) +
facet_wrap(
as.formula(paste("~", facet_col)),
scales="free_x",
ncol=5
) +
ylab("Flooded fraction")+
xlab("Month")+
labs(title = title)+
theme_minimal()
}
```

```python
#admin1's of interest
adm1_list=["N'Djamena","Mayo-Kebbi Est"]
```

```python
adm_col="admin1Name"
```

```python
gdf_adm1['include']=np.where(gdf_adm1[adm_col].isin(adm1_list),True,False)
adms=alt.Chart(gdf_adm1).mark_geoshape(stroke="black").encode(
    color=alt.Color("include",scale=alt.Scale(range=["grey","red"])),
    tooltip=["admin1Name"]
).properties(width=400,height=400,title="Admins of focus")# and SSD rivers")
# gdf_rivers=gpd.read_file(country_data_public_exploration_dir/"rivers"/"ssd_main_rivers_fao_250k"/"ssd_main_rivers_fao_250k.shp")
# rivers=alt.Chart(gdf_rivers).mark_geoshape(stroke="blue",filled=False).encode(tooltip=['CLASS'])
adms#+rivers
```

```python
fs_clip = load_floodscan()
```

```python
da_clip=fs_clip.SFED_AREA
```

```python
gdf_reg=gdf_adm2[gdf_adm2[adm_col].isin(adm1_list)]
```

```python
da_reg=da_clip.rio.clip(gdf_reg.geometry)
```

```python
#plot raster data with time slider
#can select subset of data to make it easier navigatible
#with slider
(da_reg
#  .sel(time=da_clip.time.dt.year==2014)
 .interactive.sel(time=pnw.DiscreteSlider).plot(
vmin=0,vmax=1,cmap="GnBu"))

```

```python
df_floodscan_reg=da_clip.aat.compute_raster_stats(
        gdf=gdf_adm1[gdf_adm1.include],
        feature_col=adm_col,
        stats_list=["median","min","mean","max","sum","count"],
        #computes value where 20% of the area is above that value
        percentile_list=[80],
        all_touched=False,
    )
df_floodscan_reg['year']=df_floodscan_reg.time.dt.year
df_floodscan_reg['month'] = df_floodscan_reg.time.dt.month
```

We can plot the data over all years. 
We see a yearly pattern where some years the peak is higher than others (though a max of 1.75% of the country is flooded). 

We see that some peaks have very high outliers, while others are wider. Which to classify as a flood, I am unsure about. With the method of std, we are now looking at the high outliers. 

```python
df_floodscan_reg['mean_rolling']=df_floodscan_reg.sort_values([adm_col,'time']).groupby(adm_col,as_index=False)[f"mean_{adm_col}"].rolling(10,min_periods=10).mean().mean_admin1Name
```

```python
df_floodscan_reg_rainy = df_floodscan_reg.loc[(df_floodscan_reg['month'] >= 7) & (df_floodscan_reg['month'] <= 10)]
```

```python
#should document but for now removing 2022 as it is not a full year
#but does have very high values till feb, so computations get a bit skewed with that
df_floodscan_reg=df_floodscan_reg[df_floodscan_reg.year<=2021]
```

Next we compute the return period and check which years had a peak above the return period. 
It is discussable whether only looking at the peak is the best method.. 

```python
#get one row per adm2-year combination that saw the highest mean value
df_floodscan_peak=df_floodscan_reg.sort_values('mean_rolling', ascending=False).drop_duplicates(['year'])
```

```python
years = np.arange(1.5, 20.5, 0.5)
```

```python
df_rps_ana=get_return_periods_dataframe(df_floodscan_peak, rp_var="mean_rolling",years=years, method="analytical",round_rp=False)
df_rps_emp=get_return_periods_dataframe(df_floodscan_peak, rp_var="mean_rolling",years=years, method="empirical",round_rp=False)
```

```python
fig, ax = plt.subplots()
ax.plot(df_rps_ana.index, df_rps_ana["rp"], label='analytical')
ax.plot(df_rps_emp.index, df_rps_emp["rp"], label='empirical')
ax.legend()
ax.set_xlabel('Return period [years]')
ax.set_ylabel('Fraction flooded');
```

We now use the empirical method, but could also use the analytical method. For the return periods of our interest, this will return less years. 

```python
df_floodscan_peak['rp3']=np.where(df_floodscan_peak.mean_rolling>=df_rps_emp.loc[3,'rp'],True,False)
df_floodscan_peak['rp5']=np.where(df_floodscan_peak.mean_rolling>=df_rps_emp.loc[5,'rp'],True,False)
```

```python
df_floodscan_peak[df_floodscan_peak.rp3].sort_values('mean_rolling')
```

```python
df_floodscan_peak[df_floodscan_peak.rp5].sort_values('mean_rolling')
```

```python
da_clip.sel(time=list(df_floodscan_peak[(df_floodscan_peak.year==2020)].time)[0]).plot()
```

```python
timest_rp5=list(df_floodscan_peak[df_floodscan_peak.rp5].sort_values('year').time)
da_clip_peak=da_clip.sel(time=da_clip.time.isin(timest_rp5))
```

```python
# #TODO: arghh somehow it is not finding the dates while it did so
# #before.. must have smth to do with date formats
# #Not very important though
# g=da_clip_peak.plot(col='time',
#                    )
# for ax in g.axes.flat:
#     ax.axis("off")
# g.fig.suptitle(f"Flooded fraction during peak for 1 in 5 year return period years",y=1.1);
```

```python
# #TODO: arghh somehow it is not finding the dates while it did so
# #before.. must have smth to do with date formats
# g=da_clip_peak.rio.clip(gdf_reg.geometry).plot(col='time')
# for ax in g.axes.flat:
#     ax.axis("off")
# g.fig.suptitle(f"Flooded fraction during peak for 1 in 5 year return period years",y=1.1);
```

Next we plot the smoothed data per year (with ggplot cause it is awesome). 

We can see that: 

- 2014, 2016, 2017, 2020, and 2021 indeed had clearly the highest peak. 
- This is the line with the signal on country level
- We also see some differences, e.g. the 2019 peak is lower. 


N'Djamena

```python
df_floodscan_ndjam=df_floodscan_reg[df_floodscan_reg[adm_col]=="N'Djamena"]
```

```R magic_args="-i df_floodscan_ndjam -w 40 -h 20 --units cm"
df_plot <- df_floodscan_ndjam %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_admin1Name = mean_admin1Name*100)
plotFloodedFraction(df_plot,'mean_admin1Name','year',"Flooded fraction of ROI")
```

Mayo-Kebbi Est

```python
df_floodscan_mke=df_floodscan_reg[df_floodscan_reg[adm_col]=="Mayo-Kebbi Est"]
```

```R magic_args="-i df_floodscan_mke -w 40 -h 20 --units cm"
df_plot <- df_floodscan_mke %>%
mutate(time = as.Date(time, format = '%Y-%m-%d'),mean_admin1Name = mean_admin1Name*100)
plotFloodedFraction(df_plot,'mean_admin1Name','year',"Flooded fraction of ROI")
```
