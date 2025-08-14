import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import ocha_stratus as stratus
from matplotlib.ticker import FuncFormatter

from src.constants import PROJECT_PREFIX


def combined_plots(df, glofas_thresh, save_output=True):
    fig, ax = plt.subplots(figsize=(12, 10))

    # TODO: Some duplication here from etl.check_results
    assert df.monitoring_date.nunique() == 1
    update_date = df.monitoring_date.unique()[0].strftime("%Y-%m-%d")

    df_forecast = df[df.src.str.contains("glofas_forecast")].reset_index()

    # We're taking the forecast issue date for GloFAS (not the reanalysis)
    glofas_update = df_forecast.issued_date[0].strftime("%Y-%m-%d")

    glofas_exceeds = df_forecast.value.any() > glofas_thresh
    overall_exceeds = glofas_exceeds

    forecast_subplot(
        ax,
        df_forecast,
        glofas_exceeds,
        glofas_thresh,
        "GloFAS",
        glofas_update,
    )

    if save_output:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
        buffer.seek(0)
        container_client = stratus.get_container_client(
            "projects", "dev", write=True
        )
        blob_name = (
            f"{PROJECT_PREFIX}/monitoring/{update_date}_{overall_exceeds}.png"
        )

        container_client.upload_blob(
            name=blob_name, data=buffer.getvalue(), overwrite=True
        )
        print(f"File saved on blob to {blob_name}!")
        buffer.close()


def forecast_subplot(ax, df_forecast, exceeds, thresh, dataset, date):
    ax.plot(
        df_forecast["valid_date"],
        df_forecast["value"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=4,
        label="Forecast",
        color="blue",
        alpha=0.8,
    )

    for _, row in df_forecast.iterrows():
        ax.annotate(
            f'{row["value"]:.1f}',  # noqa
            (row["valid_date"], row["value"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="blue",
        )

    # Add horizontal threshold line
    ax.axhline(
        y=thresh,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Trigger Threshold ({thresh})",
        alpha=0.8,  # noqa
    )

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %-d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    # Format y-axis with comma separators
    def format_thousands(x, pos):
        return f"{x:,.0f}"  # noqa

    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    title = f"{dataset} Monitoring: {date} | Triggers = {exceeds}"
    ax.set_ylim(0, None)
    ax.set_ylabel("Discharge, daily average (m$^3$ / s)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
