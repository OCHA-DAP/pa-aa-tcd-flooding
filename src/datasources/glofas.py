import os
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from src.constants import NDJAMENA_LAT, NDJAMENA_LON

DATA_DIR = Path(os.getenv("AA_DATA_DIR_NEW"))
GF_REA_RAW_DIR = (
    DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "cems-glofas-historical"
)
GF_REF_RAW_DIR = (
    DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "cems-glofas-reforecast"
)
GF_PROC_DIR = DATA_DIR / "public" / "processed" / "tcd" / "glofas"
GF_TEST_DIR = DATA_DIR / "public" / "raw" / "tcd" / "glofas" / "test"
PITCH = 0.005
N, S, E, W = (
    NDJAMENA_LAT + PITCH,
    NDJAMENA_LAT - PITCH,
    NDJAMENA_LON + PITCH,
    NDJAMENA_LON - PITCH,
)


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


def load_reanalysis():
    filename = "ndjamena_glofas_reanalysis.csv"
    return pd.read_csv(GF_PROC_DIR / filename, parse_dates=["time"])


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
            "area": [N, W, S, E],
        }
        client.retrieve(dataset, request, target)


def download_reforecast_ensembles():
    if not GF_REF_RAW_DIR.exists():
        GF_REF_RAW_DIR.mkdir(parents=True)
    c = cdsapi.Client()

    years = range(2003, 2023)

    leadtimes = [x * 24 for x in range(1, 47)]
    max_leadtime_chunk = 5
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
                        "hmonth": [f"{x:02}" for x in range(6, 12)],
                        "hday": [f"{x:02}" for x in range(1, 32)],
                        "leadtime_hour": [str(x) for x in leadtime_chunk],
                        "data_format": "grib",
                        "download_format": "unarchived",
                        "area": [N, W, S, E],
                    },
                    save_path,
                )

            except Exception as e:
                print(f"Failed to download {year} {lt_chunk_str}")
                print(e)
