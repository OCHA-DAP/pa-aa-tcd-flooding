import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import ocha_stratus as stratus
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.constants import FRENCH_MONTHS, PROJECT_PREFIX


def combined_plots(df, glofas_thresh, save_output=True):
    fig, ax = plt.subplots(figsize=(12, 6))

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

    # uncomment below to see plot for local debugging
    # plt.show()
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
    action_color = "dodgerblue"
    readiness_color = "darkorange"
    # Ensure datetime
    issue_date = pd.to_datetime(date)
    df = df_forecast.copy()
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["lead_days"] = (
        df["valid_date"].dt.floor("D") - issue_date.floor("D")
    ).dt.days

    # Split into groups
    df_action = df[
        df["lead_days"].between(0, 10, inclusive="both")
    ].sort_values("valid_date")
    df_readiness = df[
        df["lead_days"].between(0, 14, inclusive="both")
    ].sort_values("valid_date")

    # Threshold line
    thresh_str = f"{thresh:,.0f}".replace(",", " ")  # e.g., '1 000'
    ax.axhline(
        y=thresh,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Seuil ({thresh_str} m$^3$/s)",
        alpha=0.8,
        zorder=1,
    )

    # Helper: annotate a line's points
    def annotate_points(df_line, color):
        for _, row in df_line.iterrows():
            ax.annotate(
                f'{row["value"]:,.0f}'.replace(",", " "),
                (row["valid_date"], row["value"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=color,
            )

    # Plot mobilisation first (underneath), then action (on top)
    if not df_readiness.empty:
        ax.plot(
            df_readiness["valid_date"],
            df_readiness["value"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=4,
            label="Mobilisation (délai ≤ 14 jours)",
            color=readiness_color,
            alpha=0.9,
            zorder=2,
        )
        annotate_points(df_readiness, readiness_color)

    if not df_action.empty:
        ax.plot(
            df_action["valid_date"],
            df_action["value"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=4,
            label="Action (délai ≤ 10 jours)",
            color=action_color,
            alpha=0.95,
            zorder=3,
        )
        annotate_points(df_action, action_color)

    # Group-specific exceed checks
    exceeds_action = (
        (df_action["value"] >= thresh).any() if not df_action.empty else False
    )
    exceeds_readiness = (
        (df_readiness["value"] >= thresh).any()
        if not df_readiness.empty
        else False
    )

    # Build title suffix
    trig_parts = []
    if exceeds_readiness:
        trig_parts.append("mobilisation")
    if exceeds_action:
        trig_parts.append("action")
    trig_text = ", ".join(trig_parts) if trig_parts else "aucun"

    # French date formatter
    def french_date_formatter(x, pos):
        d = mdates.num2date(x)
        month_str = FRENCH_MONTHS[d.strftime("%b")]
        return f"{d.day} {month_str}"  # e.g., '5 août'

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(FuncFormatter(french_date_formatter))

    # Y-axis formatter with thin spaces
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:,.0f}".replace(",", " "))
    )

    # Titles, labels, cosmetics
    title = f"Suivi {dataset} : {issue_date.date()} | Déclenche = {trig_text}"
    ax.set_ylim(0, None)
    ax.set_ylabel("Débit, moyenne journalière (m$^3$/s)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    [ax.spines[s].set_visible(False) for s in ["top", "right"]]
