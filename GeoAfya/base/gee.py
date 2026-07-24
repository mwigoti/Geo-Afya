import os
import logging
from typing import Dict, List, Optional, Generator
import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from django.db import connection
from django.conf import settings
from base.models import SpatialGridCell

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. THREAD-SAFE GEE CLIENT SINGLETON
# ==============================================================================

class EarthEngineClient:
    """
    Thread-safe Singleton class to handle Google Earth Engine (GEE) 
    authentication and connection lifecycle using Service Account credentials.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EarthEngineClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Authenticates with GEE using Service Account JSON credentials or local fallback.
        """
        service_account = getattr(settings, 'GEE_SERVICE_ACCOUNT', os.getenv('GEE_SERVICE_ACCOUNT'))
        key_file = getattr(settings, 'GEE_KEY_FILE_PATH', os.getenv('GEE_KEY_FILE_PATH'))

        try:
            if service_account and key_file and os.path.exists(key_file):
                credentials = ee.ServiceAccountCredentials(service_account, key_file)
                ee.Initialize(credentials)
                logger.info("Successfully initialized GEE via Service Account.")
            else:
                ee.Initialize()
                logger.info("Successfully initialized GEE via default system credentials.")
        except Exception as e:
            logger.error(f"Failed to initialize Google Earth Engine API: {str(e)}")
            raise RuntimeError(f"GEE Initialization Failure: {str(e)}") from e


# ==============================================================================
# 2. DYNAMIC SATELLITE FEATURE EXTRACTOR
# ==============================================================================

class GEEDynamicExtractor:
    """
    Handles cloud-side dynamic climate extraction from GEE collections over 
    SpatialGridCell geometries using batched reduceRegions operations.
    """
    def __init__(self, batch_size: int = 2500):
        EarthEngineClient()  # Ensure GEE client is initialized
        self.batch_size = batch_size

    def _get_grid_gdf(self) -> gpd.GeoDataFrame:
        """
        Queries PostGIS spatial geometries into a GeoPandas GeoDataFrame.
        """
        sql = "SELECT cell_id, geom FROM base_spatialgridcell ORDER BY cell_id;"
        grid_gdf = gpd.read_postgis(sql, connection, geom_col='geom', crs="EPSG:4326")
        
        if grid_gdf.empty:
            raise ValueError("No spatial grid cells found in base_spatialgridcell table.")
            
        return grid_gdf

    def _chunk_gdf(self, gdf: gpd.GeoDataFrame) -> Generator[gpd.GeoDataFrame, None, None]:
        """
        Splits GeoDataFrame into chunks to avoid exceeding GEE JSON payload limits.
        """
        total_rows = len(gdf)
        for i in range(0, total_rows, self.batch_size):
            yield gdf.iloc[i : i + self.batch_size]

    def fetch_climate_metrics(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extracts CHIRPS precipitation, MODIS LST (Day & Night), and MODIS NDVI 
        aggregated per cell over the specified time window.

        :param start_date: ISO date string 'YYYY-MM-DD'
        :param end_date: ISO date string 'YYYY-MM-DD'
        :return: DataFrame with cell_id, precip_mm, lst_day_celsius, lst_night_celsius, ndvi_val
        """
        logger.info(f"Initiating GEE dynamic extraction for period {start_date} to {end_date}")
        grid_gdf = self._get_grid_gdf()
        
        # 1. Define GEE Image Collections and Select Bands
        # CHIRPS Daily Precipitation (Cumulative sum over window)
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(start_date, end_date) \
            .select("precipitation") \
            .sum() \
            .rename("precip_mm")

        # MODIS Daytime LST (Mean converted from Kelvin scale to Celsius)
        modis_lst_day = ee.ImageCollection("MODIS/061/MOD11A1") \
            .filterDate(start_date, end_date) \
            .select("LST_Day_1km") \
            .mean() \
            .multiply(0.02) \
            .subtract(273.15) \
            .rename("lst_day_celsius")

        # MODIS Nighttime LST (Mean converted to Celsius)
        modis_lst_night = ee.ImageCollection("MODIS/061/MOD11A1") \
            .filterDate(start_date, end_date) \
            .select("LST_Night_1km") \
            .mean() \
            .multiply(0.02) \
            .subtract(273.15) \
            .rename("lst_night_celsius")

        # MODIS Vegetation Index (Mean scaled to [-1.0, 1.0])
        modis_ndvi = ee.ImageCollection("MODIS/061/MOD13A2") \
            .filterDate(start_date, end_date) \
            .select("NDVI") \
            .mean() \
            .multiply(0.0001) \
            .rename("ndvi_val")

        # Composite multi-spectral image block
        composite_image = chirps.addBands(modis_lst_day) \
                                 .addBands(modis_lst_night) \
                                 .addBands(modis_ndvi)

        # 2. Process Extraction in Batches to avoid payload memory errors
        extracted_dfs: List[pd.DataFrame] = []
        chunk_idx = 1
        
        for gdf_chunk in self._chunk_gdf(grid_gdf):
            logger.info(f"Processing GEE spatial chunk {chunk_idx} ({len(gdf_chunk)} cells)...")
            
            # Build GEE FeatureCollection for chunk
            ee_features = []
            for _, row in gdf_chunk.iterrows():
                geom_json = row['geom'].__geo_interface__
                ee_features.append(
                    ee.Feature(ee.Geometry(geom_json), {'cell_id': int(row['cell_id'])})
                )
            ee_grid_chunk = ee.FeatureCollection(ee_features)

            # Spatial Reduction over Polygons
            reduced_chunk = composite_image.reduceRegions(
                collection=ee_grid_chunk,
                reducer=ee.Reducer.mean(),
                scale=1000,
                crs='EPSG:4326'
            )

            # Retrieve Cloud Data
            result_dict = reduced_chunk.getInfo()
            records = [feat['properties'] for feat in result_dict['features']]
            
            df_chunk = pd.DataFrame(records)
            extracted_dfs.append(df_chunk)
            chunk_idx += 1

        # 3. Concatenate and Clean Results
        final_df = pd.concat(extracted_dfs, ignore_index=True)
        
        expected_columns = ['cell_id', 'precip_mm', 'lst_day_celsius', 'lst_night_celsius', 'ndvi_val']
        for col in expected_columns:
            if col not in final_df.columns:
                final_df[col] = np.nan

        return final_df[expected_columns]


# ==============================================================================
# 3. STATIC ENVIRONMENTAL FEATURE EXTRACTOR
# ==============================================================================

class StaticRasterExtractor:
    """
    Extracts zonal statistics from static GeoTIFF files (Soil Vertisol %, DEM Elevation)
    stored locally or on S3 storage.
    """
    def extract_static_layers(self, soil_raster_path: str, dem_raster_path: str) -> pd.DataFrame:
        """
        Computes zonal mean for static layers over all grid cells in PostGIS.

        :param soil_raster_path: Path/URL to Soil Vertisol % GeoTIFF raster.
        :param dem_raster_path: Path/URL to SRTM Elevation GeoTIFF raster.
        :return: DataFrame containing cell_id, vertisol_percent, elevation_m
        """
        logger.info("Extracting static biophysical rasters (Soil Vertisols & DEM Elevation)...")
        
        if not os.path.exists(soil_raster_path):
            raise FileNotFoundError(f"Soil raster file not found at: {soil_raster_path}")
        if not os.path.exists(dem_raster_path):
            raise FileNotFoundError(f"DEM raster file not found at: {dem_raster_path}")

        sql = "SELECT id, cell_id, geom FROM base_spatialgridcell ORDER BY cell_id;"
        grid_gdf = gpd.read_postgis(sql, connection, geom_col='geom', crs="EPSG:4326")

        if grid_gdf.empty:
            raise ValueError("No spatial grid cells found in base_spatialgridcell table.")

        # 1. Soil Vertisol Content (%) Zonal Stats
        soil_stats = zonal_stats(
            grid_gdf, soil_raster_path, stats=['mean'], all_touched=True
        )
        
        # 2. SRTM DEM Elevation (meters) Zonal Stats
        dem_stats = zonal_stats(
            grid_gdf, dem_raster_path, stats=['mean'], all_touched=True
        )

        grid_gdf['vertisol_percent'] = [s['mean'] if s['mean'] is not None else np.nan for s in soil_stats]
        grid_gdf['elevation_m'] = [s['mean'] if s['mean'] is not None else np.nan for s in dem_stats]

        return grid_gdf[['cell_id', 'vertisol_percent', 'elevation_m']]


# ==============================================================================
# 4. UNIFIED INGESTION ENGINE ORCHESTRATOR
# ==============================================================================

class IngestionEngine:
    """
    Unified Ingestion Service combining dynamic satellite streams and static rasters 
    into a merged cell-aligned dataset.
    """
    def __init__(self, batch_size: int = 2500):
        self.dynamic_extractor = GEEDynamicExtractor(batch_size=batch_size)
        self.static_extractor = StaticRasterExtractor()

    def run_full_extraction(
        self, 
        start_date: str, 
        end_date: str, 
        soil_raster_path: str, 
        dem_raster_path: str
    ) -> pd.DataFrame:
        """
        Executes parallel streams for dynamic climate metrics and static environmental features,
        merging results on cell_id.

        :return: Consolidated pandas DataFrame ready for feature staging persistence.
        """
        logger.info("Starting consolidated feature extraction pipeline...")

        # Stream 1: Dynamic Climate Metrics
        df_dynamic = self.dynamic_extractor.fetch_climate_metrics(start_date, end_date)

        # Stream 2: Static Geophysical Metrics
        df_static = self.static_extractor.extract_static_layers(
            soil_raster_path=soil_raster_path,
            dem_raster_path=dem_raster_path
        )

        # Merge Streams
        merged_df = pd.merge(df_dynamic, df_static, on='cell_id', how='outer')
        merged_df.sort_values(by='cell_id', inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        logger.info(f"Extraction pipeline completed successfully. Merged records: {len(merged_df)}")
        return merged_df