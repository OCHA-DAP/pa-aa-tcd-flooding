import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.stats import genextreme as gev

from src import constants
from src.datasource_extensions import (
    CERF,
    CodABExt,
    CompareSources,
    Emdat,
    FloodScan,
    FloodScanStats,
    IfrcImpact,
)

logger = logging.getLogger(__name__)


# TODO: maybe there is a cleaner way to do this (e.g. using rioxarray?)
# question: should this preprocessing go here or in datasource_extensions?
def load_floodscan() -> xr.Dataset:
    floodscan = FloodScan(country_config=constants.country_config)
    da = floodscan.load()
    da.SFED_AREA.attrs.pop("grid_mapping")
    da.NDT_SFED_AREA.attrs.pop("grid_mapping")
    da.LWMASK_AREA.attrs.pop("grid_mapping")
    return da.rio.write_crs("EPSG:4326", inplace=True)


def get_path_compare_datasources() -> Path:
    compsourc = CompareSources(country_config=constants.country_config)
    return compsourc.get_exploration_filepath()


def load_compare_datasources() -> pd.DataFrame:
    compsourc = CompareSources(country_config=constants.country_config)
    return compsourc.load()


def load_emdat_exploration() -> pd.DataFrame:
    emdat = Emdat(country_config=constants.country_config)
    return emdat.load_exploration()


def load_emdat() -> pd.DataFrame:
    emdat = Emdat(country_config=constants.country_config)
    return emdat.load()


def load_floodscan_stats(adm_level: int) -> pd.DataFrame:
    fs_stats = FloodScanStats(
        country_config=constants.country_config, adm_level=adm_level
    )
    return fs_stats.load()


def load_cerf() -> pd.DataFrame:
    cerf = CERF(country_config=constants.country_config)
    return cerf.load()


def load_ifrc() -> pd.DataFrame:
    ifrc = IfrcImpact(country_config=constants.country_config)
    return ifrc.load()


# TODO: remove once codabs are fixed
def load_adm2() -> gpd.GeoDataFrame:
    codab_adm2 = CodABExt(country_config=constants.country_config, adm_level=2)
    return codab_adm2.load()


def load_adm1() -> gpd.GeoDataFrame:
    codab_adm1 = CodABExt(country_config=constants.country_config, adm_level=1)
    return codab_adm1.load()


# copied from pa-anticipatory-action
def get_return_periods_dataframe(
    df: pd.DataFrame,
    rp_var: str,
    years: list = None,
    method: str = "analytical",
    show_plots: bool = False,
    extend_factor: int = 1,
    round_rp: bool = True,
) -> pd.DataFrame:
    """
    Function to get the return periods, either empirically or
    analytically See the `glofas/utils.py` to do this with a xarray
    dataset instead of a dataframe
    :param df: Dataframe with data to compute rp on
    :param rp_var: column name to compute return period on
    :param years: Return period years to compute
    :param method: Either "analytical" or "empirical"
    :param show_plots: If method is analytical, can show the histogram and GEV
    distribution overlaid
    :param extend_factor: If method is analytical, can extend the interpolation
    range to reach higher return periods
    :param round_rp: if True, round the rp values, else return original values
    :return: Dataframe with return period years as index and stations as
    columns
    """
    if years is None:
        years = [1.5, 2, 3, 5]
    df_rps = pd.DataFrame(columns=["rp"], index=years)
    if method == "analytical":
        f_rp = get_return_period_function_analytical(
            df_rp=df,
            rp_var=rp_var,
            show_plots=show_plots,
            extend_factor=extend_factor,
        )
    elif method == "empirical":
        f_rp = get_return_period_function_empirical(
            df_rp=df,
            rp_var=rp_var,
        )
    else:
        logger.error(f"{method} is not a valid keyword for method")
        return None
    df_rps["rp"] = f_rp(years)
    if round_rp:
        df_rps["rp"] = np.round(df_rps["rp"])
    return df_rps


def get_return_period_function_analytical(
    df_rp: pd.DataFrame,
    rp_var: str,
    show_plots: bool = False,
    plot_title: str = "",
    extend_factor: int = 1,
):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column with the quantity to be evaluated
    :param show_plots: Show the histogram with GEV distribution overlaid
    :param plot_title: The title of the plot
    :param extend_factor: Extend the interpolation range in case you want to
    calculate a relatively high return period
    :return: Interpolated function that gives the quantity for a
    given return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    rp_var_values = df_rp[rp_var]
    shape, loc, scale = gev.fit(
        rp_var_values,
        loc=rp_var_values.median(),
        scale=rp_var_values.median() / 2,
    )
    x = np.linspace(
        rp_var_values.min(),
        rp_var_values.max() * extend_factor,
        100 * extend_factor,
    )
    if show_plots:
        fig, ax = plt.subplots()
        ax.hist(rp_var_values, density=True, bins=20)
        ax.plot(x, gev.pdf(x, shape, loc, scale))
        ax.set_title(plot_title)
        plt.show()
    y = gev.cdf(x, shape, loc, scale)
    y = 1 / (1 - y)
    return interp1d(y, x)


def get_return_period_function_empirical(df_rp: pd.DataFrame, rp_var: str):
    """
    :param df_rp: DataFrame where the index is the year, and the rp_var
    column contains the maximum value per year
    :param rp_var: The column
    with the quantity to be evaluated
    :return: Interpolated function
    that gives the quantity for a give return period
    """
    df_rp = df_rp.sort_values(by=rp_var, ascending=False)
    n = len(df_rp)
    df_rp["rank"] = np.arange(n) + 1
    df_rp["exceedance_probability"] = df_rp["rank"] / (n + 1)
    df_rp["rp"] = 1 / df_rp["exceedance_probability"]
    return interp1d(df_rp["rp"], df_rp[rp_var])
