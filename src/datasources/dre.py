import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.getenv("AA_DATA_DIR_NEW"))
RAW_DRE_DIR = DATA_DIR / "private" / "raw" / "tcd" / "dre"


def open_dre_obsv(station: str = "tp"):
    filepath = RAW_DRE_DIR / f"{station}.xls"
    df = pd.read_excel(filepath, sheet_name="Sheet1", skiprows=1)
    df = df.iloc[:-1]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.rename(
        columns={"Hauteur écoulement (cm REFERENTIEL HYDROM)": "level_cm"}
    )
    df["level_cm"] = df["level_cm"].replace(-999999, np.nan)
    df = df[["Date", "level_cm"]]
    return df
