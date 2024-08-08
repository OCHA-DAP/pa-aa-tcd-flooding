import os
import shutil
from pathlib import Path

import geopandas as gpd
import requests

DATA_DIR = Path(os.environ["AA_DATA_DIR_NEW"])
CODAB_RAW_DIR = DATA_DIR / "public" / "raw" / "tcd" / "codab"


def download_codab():
    url = "https://data.fieldmaps.io/cod/originals/tcd.shp.zip"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(CODAB_RAW_DIR / "tcd.shp.zip", "wb") as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    else:
        print(
            f"Failed to download file. "
            f"HTTP response code: {response.status_code}"
        )


def load_codab():
    return gpd.read_file(CODAB_RAW_DIR / "tcd.shp.zip")
