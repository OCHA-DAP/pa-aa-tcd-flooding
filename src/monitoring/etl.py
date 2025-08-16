import ocha_stratus as stratus
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from sqlalchemy import text

from src import cds_utils
from src.constants import GLOFAS_THRESH, GLOFAS_WARNING_THRESH, PROJECT_PREFIX

load_dotenv()

DB_SCHEMA = "projects"
DB_TABLE = "pa_aa_tcd_flooding_monitoring"


def get_blob_name(data_type, station_name, date):
    filename = (
        f"glofas_{station_name}_{data_type}_{date.strftime('%Y-%m-%d')}.grib"
    )
    return f"{PROJECT_PREFIX}/raw/glofas/monitoring/{filename}"


def get_glofas_forecast(
    forecast_blob_name,
    coords,
    issued_date,
    keep_local_copy=True,
    overwrite=False,
):
    container = stratus.get_container_client("projects", "dev")
    if (
        container.get_blob_client(forecast_blob_name).exists()
        and not overwrite
    ):
        print(f"File already exists: {forecast_blob_name}. Skipping download")
        return
    forecast_dataset = "cems-glofas-forecast"
    max_days = 14
    forecast_request = {
        "system_version": ["operational"],
        "hydrological_model": ["lisflood"],
        "product_type": ["ensemble_perturbed_forecasts"],
        "variable": "river_discharge_in_the_last_24_hours",
        "year": [str(issued_date.year)],
        "month": [str(issued_date.month).zfill(2)],
        "day": [str(issued_date.day).zfill(2)],
        "leadtime_hour": [str(24 * x) for x in range(1, max_days + 1)],
        "data_format": "grib2",
        "download_format": "unarchived",
        "area": coords,
    }

    cds_utils.download_raw_cds_api_to_blob(
        forecast_dataset,
        forecast_request,
        forecast_blob_name,
        keep_local_copy=keep_local_copy,
    )


def process_glofas(blob_name, data_type, station_name):
    ds = xr.open_dataset(
        f"temp/{blob_name}",
        engine="cfgrib",
        decode_timedelta=True,
        backend_kwargs={
            "indexpath": "",
        },
    )
    # Take the ensemble mean if forecast
    if data_type == "glofas_forecast":
        ds = ds["dis24"].mean(dim="number")
    df = (
        # we are keeping the GloFAS convention as valid time being the next day
        # because this is how the historical analysis was done
        ds.assign_coords(valid_time=ds["valid_time"] - pd.Timedelta(hours=0))
        .to_dataframe()
        .reset_index()
    )
    df["valid_date"] = pd.to_datetime(df["valid_time"])
    df["src"] = f"{data_type}_{station_name}"
    df = df.rename(columns={"dis24": "value", "time": "issued_date"})
    return df[["issued_date", "valid_date", "value", "src"]]


def get_database_forecast(monitoring_date):
    engine = stratus.get_engine(stage="dev")
    with engine.connect() as con:
        df = pd.read_sql(
            text(
                f"""
            select * from {DB_SCHEMA}.{DB_TABLE}
            where monitoring_date = :monitoring_date
            order by valid_date
            """
            ),
            con=con,
            params={"monitoring_date": monitoring_date},
        )
    if len(df) == 0:
        raise Exception(f"No data saved for {monitoring_date}")
    return df


def check_results(monitoring_date, activation=True):
    if activation:
        glofas_thresh = GLOFAS_THRESH
    else:
        glofas_thresh = GLOFAS_WARNING_THRESH

    df = get_database_forecast(monitoring_date)
    assert df.monitoring_date.nunique() == 1

    df_forecast = df[df.src.str.contains("glofas_forecast")].reset_index()
    for col in ["issued_date", "valid_date"]:
        df_forecast[col] = pd.to_datetime(df_forecast[col])
    df_forecast["lead_days"] = (
        df_forecast["valid_date"].dt.floor("D")
        - df_forecast["issued_date"].dt.floor("D")
    ).dt.days

    df_action = df_forecast[
        df_forecast["lead_days"].between(0, 10, inclusive="both")
    ].sort_values("valid_date")
    df_readiness = df_forecast[
        df_forecast["lead_days"].between(0, 14, inclusive="both")
    ].sort_values("valid_date")

    readiness_exceeds = df_readiness.value.any() > glofas_thresh
    action_exceeds = df_action.value.any() > glofas_thresh
    activations = []
    if action_exceeds:
        activations.append("action")
    if readiness_exceeds:
        activations.append("readiness")
    return activations
