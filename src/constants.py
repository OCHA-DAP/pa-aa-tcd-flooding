import os

# from aatoolbox import CodAB, GeoBoundingBox, create_custom_country_config
#
# from src.utils import load_adm1, load_adm2
#
# iso3 = "tcd"  # noqa: F821
#
#
# # TODO: replace with country_config = create_country_config(iso3) once config
# # is finalized and created in toolbox
# # TODO: hmm this is ugly but somehow config.yaml is not working as filepath
# # Do you know why?
# def get_country_config():
#     try:
#         return create_custom_country_config(filepath="src/config.yaml")
#     except FileNotFoundError:
#         return create_custom_country_config(filepath="../src/config.yaml")
#
#
# country_config = get_country_config()
# gdf_adm0 = CodAB(country_config=country_config).load(admin_level=0)
# gdf_adm1 = load_adm1()
# gdf_adm2 = load_adm2()
# geo_bounding_box = GeoBoundingBox.from_shape(gdf_adm0)

ADM1_FLOOD_PCODES = ["TD18", "TD11"]
ADM1_FLOOD_EXTRA_PCODES = ["TD18", "TD11", "TD05", "TD03", "TD16"]

# based on GloFAS interface
NDJAMENA_LON = 15.025
NDJAMENA_LAT = 12.125

# values for RP determined by taking a screenshot of the GloFAS Reporting
# Points interface and looking at the pixel position of the RP lines
# 4000 + (4 / 337) * 2000
# NDJAMENA_2YRRP = 4023.7
NDJAMENA_2YRRP = 4030

# min and max from analysis of the values shown in the GloFAS interface,
# compare with the GloFAS forecast data from CDS
NDJAMENA_2YRRP_MAX = 4165.6796875
NDJAMENA_2YRRP_MIN = 3992.47421875
NDJAMENA_2YRRP_MEAN = 4063.9473864746096
NDJAMENA_2YRRP_MEDIAN = 4051.36568359375

# 6000 + (15 / 338) * 2000
NDJAMENA_5YRRP = 6088.8

CERF_YEARS = [2012, 2022]

NDJAMENA1 = "TD18"
NDJAMENA2 = "TD1801"

MAYOKEBBIEST1 = "TD11"

MAYOBONEYE2 = "TD1101"

PROJECT_PREFIX = "pa-aa-tcd-flooding"

LISTMONK_INFO_LIST_ID = 111
LISTMONK_TRIGGER_LIST_ID = 112
LISTMONK_TEST_LIST_ID = 5  # "Tristan only"

GLOFAS_THRESH = 4542
GLOFAS_WARNING_THRESH = 3500

FRENCH_MONTHS = {
    "Jan": "jan.",
    "Feb": "fév.",
    "Mar": "mars",
    "Apr": "avr.",
    "May": "mai",
    "Jun": "juin",
    "Jul": "juil.",
    "Aug": "août",
    "Sep": "sept.",
    "Oct": "oct.",
    "Nov": "nov.",
    "Dec": "déc.",
}


# ---------------------------------------------------------------------------
# Run-mode switches (env-driven; the GHA workflow sets them).
# ---------------------------------------------------------------------------
# STAGE selects BOTH the ocha-stratus data-plane (DB + blob) and live-vs-test
# emailing. "prod" since 2026-09-23: the dev DB lost public network access on
# 2026-09-22, so the monitoring table and GloFAS/plot blobs now live in prod.
STAGE = os.getenv("STAGE", "dev")
# "listmonk" (default) or "ses" — direct SMTP through the humdata SES account
# with explicit recipients (see src/ses_mail.py). TEMPORARY "ses" in the
# workflow while Listmonk (which runs on the dev DB) is down.
EMAIL_BACKEND = os.getenv("EMAIL_BACKEND", "listmonk").strip().lower() or "listmonk"
# TEMPORARY: send the informational email every day, not only on
# activations / warnings / Mondays — a daily heartbeat while the prod cutover
# beds in. Unset it in the workflow to restore the normal cadence.
ALWAYS_EMAIL = os.getenv("ALWAYS_EMAIL", "").strip().lower() in ("1", "true", "yes")
# Route sends to the test audience even when STAGE=prod (workflow_dispatch
# input) — lets a prod-data run be checked by one person first.
TEST_EMAIL = os.getenv("TEST_EMAIL", "").strip().lower() in ("1", "true", "yes")
SES_RECIPIENTS_LIVE = [
    "tristan.downing@un.org",
    "zachary.arno@un.org",
    "leonardo.milano@un.org",
    "hannah.ker@un.org",
]
SES_RECIPIENTS_TEST = ["tristan.downing@un.org"]
