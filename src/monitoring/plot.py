import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import ocha_stratus as stratus
from matplotlib.ticker import FuncFormatter

from src.constants import FRENCH_MONTHS, PROJECT_PREFIX


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

    plt.show()
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
    thresh_str = f"{thresh:,.0f}".replace(",", " ")  # e.g., '1 000'
    ax.axhline(
        y=thresh,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Seuil ({thresh_str} m$^3$/s)",
        alpha=0.8,  # noqa
    )

    def french_date_formatter(x, pos):
        date = mdates.num2date(x)  # convert from Matplotlib's float days
        month_str = FRENCH_MONTHS[date.strftime("%b")]
        return f"{date.day} {month_str}"  # e.g., '5 août'

    # Format x-axis dates
    ax.xaxis.set_major_formatter(FuncFormatter(french_date_formatter))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    # Format y-axis with comma separators
    def format_thousands(x, pos):
        return f"{x:,.0f}"  # noqa

    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    title = f"Suivi {dataset} : {date} | Déclenche = {exceeds}"
    ax.set_ylim(0, None)
    ax.set_ylabel("Débit, moyenne journalière (m$^3$/s)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    [ax.spines[x].set_visible(False) for x in ["top", "right"]]
