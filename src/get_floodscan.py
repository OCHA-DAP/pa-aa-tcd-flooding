# TODO: remove this after making top-level
import os
import sys
from pathlib import Path
import rioxarray
import geopandas as gpd

path_mod = f"{Path(os.path.dirname(os.path.realpath(__file__))).parents[1]}/"
sys.path.append(path_mod)
from src.indicators.flooding.floodscan import floodscan
# from aatoolbox import CodAB
from src.indicators.drought.config import Config

ISO3 = "tcd"


def main(clip=True, process=False):

    

    #clip to country
    #this takes long, so only do when not saved the file yet
    #read floodscan data

    #TODO: this is copied from pa-anticipatory-action and thus broken. 
    # Probably easier to start from scratch by using the 
    # utils to read the global and country data
    if clip: 
        config=Config()
        parameters = config.parameters(ISO3)
        country_data_raw_dir = Path(config.DATA_DIR) / config.PUBLIC_DIR / config.RAW_DIR / ISO3
        adm0_bound_path=country_data_raw_dir / config.SHAPEFILE_DIR / parameters["path_admin0_shp"]
        gdf_adm0=gpd.read_file(adm0_bound_path)
        floodscan_data = floodscan.Floodscan()
        fs_raw = floodscan_data.read_raw_dataset()
        fs_clip = (fs_raw.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                .rio.write_crs("EPSG:4326")
                .rio.clip(gdf_adm0["geometry"], all_touched=True, from_disk=True))
        fs_clip.SFED_AREA.attrs.pop('grid_mapping')
        fs_clip.NDT_SFED_AREA.attrs.pop('grid_mapping')
        fs_clip.LWMASK_AREA.attrs.pop('grid_mapping')
        clipped_filepath=floodscan_data._get_clipped_filepath(iso3=ISO3)
        clipped_filepath.parent.mkdir(parents=True, exist_ok=True)

        fs_clip.to_netcdf(clipped_filepath)

class Floodscan:
    """Create an instance of a Floodscan object, from which you can process the
    raw data and read the data."""

    def read_raw_dataset(self):
        filepath = self._get_raw_filepath()
        # would be better to do with load_dataset, but since dataset is
        # huge this takes up too much memory..
        with xr.open_dataset(filepath) as ds:
            return ds

    def _get_raw_filepath(
        self,
    ):
        directory = (
            DATA_DIR
            / PRIVATE_DATA_DIR
            / RAW_DATA_DIR
            / GLOBAL_DIR
            / FLOODSCAN_DIR
        )

        return directory / Path(FLOODSCAN_FILENAME)


if __name__ == "__main__":
    main()
