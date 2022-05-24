from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from aatoolbox.config.countryconfig import CountryConfig
from aatoolbox.datasources.datasource import DataSource


# TODO: Fix this -- either make a separate ABC or make it work
#  better from the toolbox
class _DataSourceExtension(DataSource):
    def __init__(self, country_config: CountryConfig,is_global_raw=False):
        if hasattr(self, "_IS_GLOBAL_RAW"):
            is_global_raw = self._IS_GLOBAL_RAW
        super().__init__(
            country_config,
            datasource_base_dir=self._DATASOURCE_BASENAME,
            is_public=self._IS_PUBLIC,
            is_global_raw=is_global_raw,
        )
        if hasattr(self, "_RAW_FILENAME"):
            self.raw_filepath = self._raw_base_dir / self._RAW_FILENAME
        if hasattr(self, "_PROCESSED_FILENAME"):
            self.processed_filepath = (
                self._processed_base_dir / self._PROCESSED_FILENAME
            )
        #TODO: need a better method to define the exploration_dir
        if hasattr(self, "_EXPLORATION_FILENAME"):
            self._exploration_dir=Path(*Path(self._processed_base_dir).parts[:-3])/"exploration"/Path(*Path(self._processed_base_dir).parts[-2:])
            self.exploration_filepath = (self._exploration_dir / 
            self._EXPLORATION_FILENAME)

    def download(self):
        pass
    def process(self):
        pass

class FloodScan(_DataSourceExtension):
    _DATASOURCE_BASENAME = "floodscan"
    _RAW_FILENAME = "aer_sfed_area_300s_19980112_20220424_v05r01.nc"
    _PROCESSED_FILENAME = "tcd_aer_sfed_area_300s_19980112_20220424_v05r01.nc"
    _IS_PUBLIC = False
    _IS_GLOBAL_RAW = True

    def load_raw(self) -> xr.Dataset:
        with xr.open_dataset(self.raw_filepath) as ds:
            return ds

    def load(self) -> xr.Dataset:
        return xr.load_dataset(self.processed_filepath)

class FloodScanStats(_DataSourceExtension):
    def __init__(self, country_config: CountryConfig, adm_level: int, is_global_raw=False):
        
        self._PROCESSED_FILENAME = f"tcd_floodscan_stats_adm{adm_level}.csv"
        self._DATASOURCE_BASENAME = "floodscan"
        self._IS_PUBLIC = False
        super().__init__(country_config, is_global_raw)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.processed_filepath,parse_dates=['time'])

class Emdat(_DataSourceExtension):
    _DATASOURCE_BASENAME = "emdat"
    _EXPLORATION_FILENAME="emdat_public_2022_05_20_query_uid-cRaHxk.xlsx"
    _PROCESSED_FILENAME="tcd_emdat_flood_2022_05_20_processed.csv"
    _IS_PUBLIC = False

    def load_exploration(self) -> pd.DataFrame:
        return pd.read_excel(self.exploration_filepath,header=6)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.processed_filepath)

class CERF(_DataSourceExtension):
    _DATASOURCE_BASENAME = "cerf"
    _RAW_FILENAME = "CERF Allocations.csv"
    _IS_PUBLIC = True
    _IS_GLOBAL_RAW=True

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_filepath,parse_dates=["dateUSGSignature"])

class IfrcImpact(_DataSourceExtension):
    _DATASOURCE_BASENAME = "ifrc"
    _EXPLORATION_FILENAME = "flood_impact_yearly.xlsx"
    _IS_PUBLIC = False

    def load(self) -> pd.DataFrame:
        return pd.read_excel(self.exploration_filepath,header=[1,2])

class CodABExt(_DataSourceExtension):
    def __init__(self, country_config: CountryConfig, adm_level, is_global_raw=False):
        self._DATASOURCE_BASENAME = "cod_ab"
        self._IS_PUBLIC = True
        if adm_level ==1:
            self._RAW_FILENAME = "tcd_admbnda_adm1_ocha/tcd_admbnda_adm1_ocha.shp"
        elif adm_level ==2:
            self._RAW_FILENAME = ("tcd_admbnda_adm2_ocha_20170615/"
            "tcd_admbnda_adm2_ocha_20170615.shp")
        else:
            raise ValueError("no adm bounds for adm_level >2.")
        
        super().__init__(country_config, is_global_raw)

    def load(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.raw_filepath)

