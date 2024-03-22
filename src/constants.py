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