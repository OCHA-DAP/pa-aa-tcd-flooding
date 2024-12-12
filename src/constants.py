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
NDJAMENA_2YRRP = 4023.7
# 6000 + (15 / 338) * 2000
NDJAMENA_5YRRP = 6088.8

CERF_YEARS = [2012, 2022]

NDJAMENA1 = "TD18"
NDJAMENA2 = "TD1801"
