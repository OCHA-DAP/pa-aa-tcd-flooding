import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import ocha_stratus as stratus
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.constants import FRENCH_MONTHS, PROJECT_PREFIX
from src.monitoring import etl
from src.monitoring.etl import get_activations_text

# HDX design tokens (data.humdata.org design system)
HDX_PRIMARY_5 = "#1862d8"  # action trigger line
HDX_WARNING_6 = "#aa7222"  # mobilisation trigger line
HDX_ERROR_5 = "#c44536"  # activation threshold
HDX_ERROR_6 = "#9d372b"  # activated status text
HDX_NEUTRAL_1 = "#ebeff0"  # gridlines
HDX_NEUTRAL_3 = "#c4d0d1"  # axis line
HDX_NEUTRAL_7 = "#5e6a6b"  # secondary text, tick labels
HDX_NEUTRAL_8 = "#3f4748"  # legend and value labels
HDX_NEUTRAL_9 = "#1f2324"  # title

# Roboto is installed in CI (fonts-roboto); DejaVu Sans is matplotlib's
# bundled fallback everywhere else
plt.rcParams["font.sans-serif"] = ["Roboto", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"


def format_number_fr(x):
    return f"{x:,.0f}".replace(",", " ")


def format_date_fr(d):
    return f"{d.day} {FRENCH_MONTHS[d.strftime('%b')]} {d.year}"


def combined_plots(df, glofas_thresh, save_output=True):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

    # TODO: Some duplication here from etl.check_results
    assert df.monitoring_date.nunique() == 1
    update_date = df.monitoring_date.unique()[0].strftime("%Y-%m-%d")

    df_forecast = df[df.src.str.contains("glofas_forecast")].reset_index()

    # We're taking the forecast issue date for GloFAS (not the reanalysis)
    glofas_update = df_forecast.issued_date[0].strftime("%Y-%m-%d")

    activations = etl.check_results(update_date, activation=True)

    forecast_subplot(
        ax,
        df_forecast,
        glofas_thresh,
        activations,
        "GloFAS",
        glofas_update,
    )

    # uncomment below to see plot for local debugging
    # plt.show()
    if save_output:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=200)
        buffer.seek(0)
        container_client = stratus.get_container_client(
            "projects", "dev", write=True
        )
        blob_name = f"{PROJECT_PREFIX}/monitoring/{update_date}_{bool(activations)}.png"  # noqa: E501

        container_client.upload_blob(
            name=blob_name, data=buffer.getvalue(), overwrite=True
        )
        print(f"File saved on blob to {blob_name}!")
        buffer.close()
    return fig


def forecast_subplot(ax, df_forecast, thresh, activations, dataset, date):
    issue_date = pd.to_datetime(date)
    df = df_forecast.copy()
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["lead_days"] = (
        df["valid_date"].dt.floor("D") - issue_date.floor("D")
    ).dt.days
    df = df.sort_values("valid_date")

    # One continuous forecast, colored by trigger window; the mobilisation
    # segment starts at day 10 so the line connects
    df_action = df[df["lead_days"].between(0, 10, inclusive="both")]
    df_mobilisation = df[df["lead_days"].between(10, 14, inclusive="both")]

    # Recessive frame: horizontal gridlines only, single baseline axis
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=HDX_NEUTRAL_1, linewidth=1)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HDX_NEUTRAL_3)
    ax.tick_params(colors=HDX_NEUTRAL_7, labelsize=9.5, length=0)

    # Threshold line, direct-labeled instead of in the legend
    thresh_str = format_number_fr(thresh)
    ax.axhline(
        y=thresh,
        color=HDX_ERROR_5,
        linestyle=(0, (6, 4)),
        linewidth=1.5,
        zorder=2,
    )
    # Label below the line when the late forecast crowds the threshold
    crowded = not df.empty and df["value"].iloc[-5:].max() >= thresh * 0.9
    ax.annotate(
        f"Seuil d'activation : {thresh_str} m³/s",
        xy=(1, thresh),
        xycoords=("axes fraction", "data"),
        xytext=(0, -7 if crowded else 5),
        textcoords="offset points",
        ha="right",
        va="top" if crowded else "bottom",
        fontsize=9.5,
        color=HDX_ERROR_5,
    )

    line_style = dict(
        marker="o",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2,
    )
    if not df_mobilisation.empty:
        ax.plot(
            df_mobilisation["valid_date"],
            df_mobilisation["value"],
            color=HDX_WARNING_6,
            label="Mobilisation (délai ≤ 14 jours)",
            zorder=3,
            **line_style,
        )
    if not df_action.empty:
        ax.plot(
            df_action["valid_date"],
            df_action["value"],
            color=HDX_PRIMARY_5,
            label="Action (délai ≤ 10 jours)",
            zorder=4,
            **line_style,
        )

    # Label only the forecast peak, not every point; keep the text inside
    # the plot when the peak sits near either edge
    if not df.empty:
        peak = df.loc[df["value"].idxmax()]
        x_frac = (df["valid_date"] <= peak["valid_date"]).mean()
        ha = (
            "right" if x_frac > 0.85 else "left" if x_frac < 0.15 else "center"
        )
        ax.annotate(
            f"Pic : {format_number_fr(peak['value'])} m³/s",
            (peak["valid_date"], peak["value"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha=ha,
            fontsize=9.5,
            color=HDX_NEUTRAL_8,
        )

    def french_date_formatter(x, pos):
        d = mdates.num2date(x)
        return f"{d.day} {FRENCH_MONTHS[d.strftime('%b')]}"

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(FuncFormatter(french_date_formatter))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: format_number_fr(x))
    )

    # Headroom above both the forecast and the threshold for the labels
    y_max = max(df["value"].max() if not df.empty else 0, thresh)
    ax.set_ylim(0, y_max * 1.18)
    ax.set_ylabel(
        "Débit, moyenne journalière (m³/s)",
        fontsize=10.5,
        color=HDX_NEUTRAL_7,
    )

    # Left-aligned title; issue date and trigger status on a subtitle row;
    # legend in its own band above the axes so it never overlaps the data
    ax.set_title(
        f"Prévisions {dataset} — Chari à N'Djamena",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=HDX_NEUTRAL_9,
        pad=48,
    )
    ax.text(
        0,
        1.115,
        f"Émis le {format_date_fr(issue_date)}",
        transform=ax.transAxes,
        fontsize=10.5,
        color=HDX_NEUTRAL_7,
        va="bottom",
    )
    activations_text = get_activations_text(activations)
    status_color = HDX_ERROR_6 if activations else HDX_NEUTRAL_7
    ax.text(
        1,
        1.115,
        f"Statut : {activations_text}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold" if activations else "normal",
        color=status_color,
        ha="right",
        va="bottom",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "Action" not in labels[i])
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncols=2,
        frameon=False,
        fontsize=10,
        labelcolor=HDX_NEUTRAL_8,
        columnspacing=1.6,
        handlelength=1.8,
    )
