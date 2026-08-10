import os
from pathlib import Path
from typing import Literal

import cdsapi
import numpy as np
import ocha_stratus as stratus
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from src import cds_utils
from src.constants import (
    NDJAMENA_2YRRP,
    NDJAMENA_2YRRP_MAX,
    NDJAMENA_2YRRP_MEAN,
    NDJAMENA_2YRRP_MEDIAN,
    NDJAMENA_2YRRP_MIN,
    NDJAMENA_5YRRP,
    NDJAMENA_LAT,
    NDJAMENA_LON,
    PROJECT_PREFIX,
)

GF_STATIONS = {
    "bongor": {
        # note that coords are based on selecting pixel from raw GloFAS data,
        # not from the GloFAS interface
        "lon": 15.43,
        "lat": 10.22,
    },
    "katoa": {
        "lon": 15.073,
        "lat": 10.834,
    },
    "ndjamena": {
        "lon": NDJAMENA_LON,
        "lat": NDJAMENA_LAT,
    },
}

DATA_DIR = Path(os.getenv("AA_DATA_DIR_NEW", ""))
GF_REA_RAW_DIR = (
    DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "cems-glofas-historical"
)
GF_REF_RAW_DIR = (
    DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "cems-glofas-reforecast"
)
GF_F_RAW_DIR = (
    DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "cems-glofas-forecast"
)
GF_PROC_DIR = DATA_DIR / "public" / "processed" / "tcd" / "glofas"
GF_TEST_DIR = DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "test"
PITCH = 0.005
NDJAMENA_N, NDJAMENA_S, NDJAMENA_E, NDJAMENA_W = (
    NDJAMENA_LAT + PITCH,
    NDJAMENA_LAT - PITCH,
    NDJAMENA_LON + PITCH,
    NDJAMENA_LON - PITCH,
)


def get_blob_name(
    data_type: Literal["raw", "processed"],
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    station_name: str,
    year: int = None,
) -> str:
    if year is None and data_type == "raw":
        raise ValueError("Year must be provided for raw data")
    if data_type == "raw":
        return f"{PROJECT_PREFIX}/{data_type}/glofas/{dataset}/glofas_{data_type}_{dataset}_{station_name}_{year}.grib"  # noqa
    return f"{PROJECT_PREFIX}/{data_type}/glofas/glofas_{dataset}_{station_name}.parquet"  # noqa


def get_coords(station_name):
    station = GF_STATIONS[station_name]
    glofas_lon, glofas_lat = get_glofas_grid_coords(
        station["lon"], station["lat"]
    )
    pitch = 0.001
    N = glofas_lat + pitch
    S = glofas_lat
    E = glofas_lon + pitch
    W = glofas_lon
    return [N, W, S, E]


def get_glofas_grid_coords(lon, lat):
    grid_lat = np.arange(-90.025, 90, 0.05)
    grid_lon = np.arange(-180.025, 180, 0.05)
    nearest_lat_idx = (np.abs(grid_lat - lat)).argmin()
    nearest_lon_idx = (np.abs(grid_lon - lon)).argmin()
    return round(grid_lon[nearest_lon_idx], 3), round(
        grid_lat[nearest_lat_idx], 3
    )


def download_glofas_reanalysis_year_to_blob(
    year: int, station_name: str, pitch: float = 0.001, clobber: bool = False
):
    station = GF_STATIONS[station_name]
    glofas_lon, glofas_lat = get_glofas_grid_coords(
        station["lon"], station["lat"]
    )
    N = glofas_lat + pitch
    S = glofas_lat
    E = glofas_lon + pitch
    W = glofas_lon
    dataset = "cems-glofas-historical"
    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [f"{year}"],
        "hmonth": [f"{x:02}" for x in range(1, 13)],
        "hday": [f"{x:02}" for x in range(1, 32)],
        "data_format": "grib2",
        "download_format": "unarchived",
        "area": [N, W, S, E],
    }
    blob_name = get_blob_name("raw", "reanalysis", station_name, year)
    # check if blob exists
    if not clobber and stratus.list_container_blobs(
        name_starts_with=blob_name
    ):
        print(f"{blob_name} already exists in blob storage")
        return
    return cds_utils.download_raw_cds_api_to_blob(dataset, request, blob_name)


def load_glofas_reanalysis_year(
    data_type: Literal["raw", "processed"], station_name: str, year: int
):
    blob_name = get_blob_name(data_type, "reanalysis", station_name, year)
    if data_type == "raw":
        local_filepath = "temp" / Path(blob_name)
        if local_filepath.exists():
            return xr.load_dataset(local_filepath)
        else:
            blob_data = stratus.load_blob_data(blob_name)
            print(f"Downloading {blob_name} to {local_filepath}")
            if not local_filepath.parent.exists():
                os.makedirs(local_filepath.parent)
            with open(local_filepath, "wb") as file:
                file.write(blob_data)
            return xr.load_dataset(local_filepath)
    elif data_type == "processed":
        return stratus.load_parquet_from_blob(blob_name)


def process_glofas_reanalysis(station_name: str):
    raw_blob_dir = "/".join(
        get_blob_name("raw", "reanalysis", station_name, year=0).split("/")[
            :-1
        ]
    )
    blob_names = [
        x
        for x in stratus.list_container_blobs(name_starts_with=raw_blob_dir)
        if x.endswith(".grib") and station_name in x
    ]
    dfs = []
    for blob_name in tqdm(blob_names):
        year = int(blob_name.split(".")[0].split("_")[-1])
        ds = load_glofas_reanalysis_year("raw", station_name, year)
        da = ds["dis24"]
        df_in = da.to_dataframe().reset_index()[["time", "dis24"]]
        dfs.append(df_in)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time")
    blob_name = get_blob_name("processed", "reanalysis", station_name)
    stratus.upload_parquet_to_blob(df, blob_name)


def download_glofas_reanalysis_to_blob(
    station_name: str,
    clobber: bool = False,
    min_year: int = 1979,
    max_year: int = 2024,
):
    for year in tqdm(range(min_year, max_year + 1)):
        download_glofas_reanalysis_year_to_blob(
            year, station_name, clobber=clobber
        )


def load_glofas_reanalysis(station_name: str):
    blob_name = get_blob_name("processed", "reanalysis", station_name)
    return stratus.load_parquet_from_blob(blob_name)


def process_reanalysis():
    """Process reanalysis for N'Djamena station only"""
    if not GF_PROC_DIR.exists():
        GF_PROC_DIR.mkdir(parents=True)
    files = [x for x in os.listdir(GF_REA_RAW_DIR) if x.endswith(".grib")]
    dfs = []
    for file in tqdm(files):
        da_in = xr.load_dataset(GF_REA_RAW_DIR / file, engine="cfgrib")[
            "dis24"
        ]
        df_in = (
            da_in.sel(
                latitude=NDJAMENA_LAT, longitude=NDJAMENA_LON, method="nearest"
            )
            .to_dataframe()
            .reset_index()[["time", "dis24"]]
        )
        dfs.append(df_in)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time")
    filename = "ndjamena_glofas_reanalysis.csv"
    df.to_csv(GF_PROC_DIR / filename, index=False)


def load_reanalysis(local: bool = False):
    """Load N'Djamena GloFAS reanalysis data."""
    filename = "ndjamena_glofas_reanalysis.csv"
    if local:
        filepath = Path("temp") / filename
    else:
        filepath = GF_PROC_DIR / filename
    return pd.read_csv(filepath, parse_dates=["time"])


def download_reanalysis():
    if not GF_REA_RAW_DIR.exists():
        GF_REA_RAW_DIR.mkdir(parents=True)
    years = range(2003, 2024)
    client = cdsapi.Client()
    dataset = "cems-glofas-historical"
    for year in tqdm(years):
        filename = f"ndjamena_glofas_reanalysis_{year}.grib"
        target = GF_REA_RAW_DIR / filename
        if target.exists():
            print(f"already downloaded for {year}")
            continue
        request = {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "hyear": [f"{year}"],
            "hmonth": [f"{x:02}" for x in range(1, 13)],
            "hday": [f"{x:02}" for x in range(1, 32)],
            "data_format": "grib2",
            "download_format": "unarchived",
            "area": [NDJAMENA_N, NDJAMENA_W, NDJAMENA_S, NDJAMENA_E],
        }
        client.retrieve(dataset, request, target)


def download_forecast_ensembles():
    if not GF_F_RAW_DIR.exists():
        GF_F_RAW_DIR.mkdir(parents=True)
    c = cdsapi.Client()
    dataset = "cems-glofas-forecast"
    leadtimes = [x * 24 for x in range(1, 31)]
    days = [1, 5, 8, 12, 15, 19, 22, 26, 29]
    extend_pitch = 0.005
    for leadtime in tqdm(leadtimes):
        filename = f"ndjamena_forecast_ens_2023_lt{leadtime}.grib"
        save_path = GF_F_RAW_DIR / filename
        if save_path.exists():
            print(f"Skipping {leadtime}, already exists")
            continue
        try:
            c.retrieve(
                dataset,
                {
                    "system_version": ["operational"],
                    "hydrological_model": ["lisflood"],
                    "product_type": ["ensemble_perturbed_forecasts"],
                    "variable": "river_discharge_in_the_last_24_hours",
                    "year": ["2023"],
                    "month": [f"{x:02}" for x in range(6, 12)],
                    "day": [f"{x:02}" for x in days],
                    "leadtime_hour": [str(leadtime)],
                    "data_format": "grib2",
                    "download_format": "unarchived",
                    "area": [
                        NDJAMENA_N + extend_pitch,
                        NDJAMENA_W - extend_pitch,
                        NDJAMENA_S - extend_pitch,
                        NDJAMENA_E + extend_pitch,
                    ],
                },
                save_path,
            )
        except Exception as e:
            print(f"Failed to download {leadtime}")
            print(e)


def download_reforecast_ensembles():
    """
    Download reforecast ensembles for N'Djamena station.
    Note that because of CDS API limitations, have to split requests by
    leadtime chunks and years.
    """
    if not GF_REF_RAW_DIR.exists():
        GF_REF_RAW_DIR.mkdir(parents=True)
    c = cdsapi.Client()

    years = range(2003, 2024)

    leadtimes = [x * 24 for x in range(1, 47)]
    max_leadtime_chunk = 5
    # split leadtimes into chunks
    # max_leadtime_chunk size is determined manually by iterating over chunk
    # sizes in the CDS online interface and the using largest one that
    # doesn't result in too large of a request
    leadtime_chunks = [
        leadtimes[x : x + max_leadtime_chunk]
        for x in range(0, len(leadtimes), max_leadtime_chunk)
    ]

    for leadtime_chunk in tqdm(leadtime_chunks):
        lt_chunk_str = f"{leadtime_chunk[0]}-{leadtime_chunk[-1]}"
        for year in tqdm(years):
            save_path = (
                GF_REF_RAW_DIR
                / f"ndjamena_reforecast_ens_{year}_lt{lt_chunk_str}.grib"
            )
            if save_path.exists():
                print(f"Skipping {year} {lt_chunk_str}, already exists")
                continue
            try:
                c.retrieve(
                    "cems-glofas-reforecast",
                    {
                        "system_version": ["version_4_0"],
                        "hydrological_model": ["lisflood"],
                        "product_type": ["ensemble_perturbed_reforecast"],
                        "variable": "river_discharge_in_the_last_24_hours",
                        "hyear": [f"{year}"],
                        # only taking relevant months (June to November)
                        "hmonth": [f"{x:02}" for x in range(6, 12)],
                        "hday": [f"{x:02}" for x in range(1, 32)],
                        "leadtime_hour": [str(x) for x in leadtime_chunk],
                        "data_format": "grib",
                        "download_format": "unarchived",
                        "area": [
                            NDJAMENA_N,
                            NDJAMENA_W,
                            NDJAMENA_S,
                            NDJAMENA_E,
                        ],
                    },
                    save_path,
                )

            except Exception as e:
                print(f"Failed to download {year} {lt_chunk_str}")
                print(e)


def process_reforecast_ensembles(skip_lt_groups=None, verbose: bool = False):
    """Combine various leadtime chunk and year files from download into
    single parquet file.
    """
    filenames = [x for x in os.listdir(GF_REF_RAW_DIR) if "ens" in x]
    if skip_lt_groups is None:
        skip_lt_groups = []
    filenames = [
        x
        for x in filenames
        if x.split("_")[-1].split(".")[0] not in skip_lt_groups
    ]

    dfs = []
    for filename in tqdm(filenames):
        filepath = GF_REF_RAW_DIR / filename
        if verbose:
            print(f"Processing {filename}")
        ds_in = xr.open_dataset(
            filepath,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "decode_timedelta": True},
        )
        df_in = (
            ds_in.sel(
                latitude=NDJAMENA_LAT, longitude=NDJAMENA_LON, method="nearest"
            )
            .to_dataframe()[["dis24", "valid_time"]]
            .reset_index()
        )
        df_in["leadtime"] = df_in["step"].dt.days
        df_in = df_in.drop(columns=["step"])
        dfs.append(df_in)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["time", "leadtime"])
    filename = "ndjamena_glofas_reforecast_ens.parquet"
    df.to_parquet(GF_PROC_DIR / filename)


def load_reforecast_ensembles(local: bool = False):
    filename = "ndjamena_glofas_reforecast_ens.parquet"
    if local:
        filepath = Path("temp") / filename
    else:
        filepath = GF_PROC_DIR / filename
    return pd.read_parquet(filepath)


def process_reforecast_frac():
    """Calculate fraction of ensemble members exceeding 2 and 5 year return"""
    df = pd.read_parquet(
        GF_PROC_DIR / "ndjamena_glofas_reforecast_ens.parquet"
    )

    df["2yr_thresh"] = df["dis24"] > NDJAMENA_2YRRP
    df["2yr_thresh_max"] = df["dis24"] > NDJAMENA_2YRRP_MAX
    df["2yr_thresh_min"] = df["dis24"] > NDJAMENA_2YRRP_MIN
    df["2yr_thresh_mean"] = df["dis24"] > NDJAMENA_2YRRP_MEAN
    df["2yr_thresh_median"] = df["dis24"] > NDJAMENA_2YRRP_MEDIAN
    df["5yr_thresh"] = df["dis24"] > NDJAMENA_5YRRP

    ens = (
        df.groupby(["time", "leadtime", "valid_time"])[
            [x for x in df.columns if "yr_thresh" in x]
        ]
        .mean()
        .reset_index()
    )
    filename = "ndjamena_glofas_reforecast_frac.parquet"
    ens.to_parquet(GF_PROC_DIR / filename)


def load_reforecast_frac():
    filename = "ndjamena_glofas_reforecast_frac.parquet"
    return pd.read_parquet(GF_PROC_DIR / filename)
