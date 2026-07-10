from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

T = TypeVar("T")
DFType = TypeVar("DFType", "pd.DataFrame", "gpd.GeoDataFrame")
BoundDFType = TypeVar("BoundDFType", bound="pd.DataFrame")
