from aatoolbox import CodAB, GeoBoundingBox, GlofasReanalysis
from src import constants
    
import logging
logging.basicConfig(level=logging.INFO)

codab = CodAB(country_config=constants.country_config)
codab.download()
gdf_adm0=codab.load(admin_level=0)
geo_bounding_box = GeoBoundingBox.from_shape(gdf_adm0)
glofas_reanalysis = GlofasReanalysis(country_config=constants.country_config,geo_bounding_box=geo_bounding_box)

glofas_reanalysis.download(year_min=1998)
#when you add a new reporting point, you want to set clobber to True
glofas_reanalysis.process(year_min=1998,clobber=False)
print(glofas_reanalysis.load())