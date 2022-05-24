from aatoolbox import create_custom_country_config, CodAB, GeoBoundingBox
from src.utils import load_adm2, load_adm1
iso3 = "tcd"  # noqa: F821


#TODO: replace with country_config = create_country_config(iso3) once config
#is finalized and created in toolbox
#TODO: hmm this is ugly but somehow config.yaml is not working as filepath
#Do you know why? 
try: 
    country_config = create_custom_country_config(filepath="src/config.yaml")
except: 
    country_config = create_custom_country_config(filepath="../src/config.yaml")

gdf_adm0=CodAB(country_config=country_config).load(admin_level=0)
gdf_adm1=load_adm1()
gdf_adm2=load_adm2()
geo_bounding_box = GeoBoundingBox.from_shape(gdf_adm0)