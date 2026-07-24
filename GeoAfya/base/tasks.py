import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from celery import shared_task
from django.db import transaction, connection
from django.utils import timezone
from django.conf import settings
import pandas as pd
import numpy as np

# Domain Engine Imports
from base.gee import IngestionEngine
from base.scalers import TRINormalizer
from base.tri import TRIEngine
from base.models import (
    SpatialGridCell,
    RawGridFeatureStaging,
    RiskAssessmentRun,
    AlertDispatchLog,
    HealthWorker
)

logger = logging.getLogger(__name__)


# ==============================================================================
# END-TO-END CELERY PIPELINE ORCHESTRATOR
# ==============================================================================

@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # 5 minute retry backoff
    name="base.tasks.run_full_tri_pipeline"
)
def run_full_tri_pipeline(
    self,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    soil_raster_path: Optional[str] = None,
    dem_raster_path: Optional[str] = None,
    batch_size: int = 2000
) -> Dict[str, Any]:
    """
    End-to-End Orchestrator Task for GeoAfya TRI Risk Modeling.

    Pipeline Stages:
    1. Lifecycle Initialization & Run State Registration
    2. Dynamic Satellite & Static Biophysical Data Ingestion
    3. Raw Feature Staging (Bulk Upsert into PostGIS)
    4. Multi-Domain Vector Normalization (AHP Weighted Scalers)
    5. TRI Vector Math Calculation & Step 4 Logistic Saturation Adjustment
    6. Bulk Persistence of Assessment Scores to Grid Cells
    7. Automated High-Risk Alert Dispatch Registration

    :param start_date: ISO string 'YYYY-MM-DD' (Defaults to 14 days prior if None)
    :param end_date: ISO string 'YYYY-MM-DD' (Defaults to today if None)
    :param soil_raster_path: Path/URL to Vertisol % GeoTIFF
    :param dem_raster_path: Path/URL to SRTM DEM GeoTIFF
    :param batch_size: PostgreSQL bulk DB transaction chunk size
    :return: Dictionary summarizing execution metrics
    """
    now = timezone.now()
    
    # --------------------------------------------------------------------------
    # 0. RESOLVE DEFAULT PARAMETERS
    # --------------------------------------------------------------------------
    if not end_date:
        end_date = now.strftime('%Y-%m-%d')
    if not start_date:
        start_date = (now - timedelta(days=14)).strftime('%Y-%m-%d')
        
    soil_path = soil_raster_path or getattr(settings, 'STATIC_SOIL_RASTER_PATH', '/data/rasters/soil_vertisol.tif')
    dem_path = dem_raster_path or getattr(settings, 'STATIC_DEM_RASTER_PATH', '/data/rasters/srtm_dem.tif')

    logger.info(f"Starting TRI Assessment Run [{self.request.id}] for window {start_date} to {end_date}")

    # --------------------------------------------------------------------------
    # 1. INITIALIZE RISK ASSESSMENT RUN TRACKER
    # --------------------------------------------------------------------------
    run_record = RiskAssessmentRun.objects.create(
        celery_task_id=self.request.id or "MANUAL_EXECUTION",
        start_date=start_date,
        end_date=end_date,
        status='RUNNING',
        started_at=now
    )

    try:
        # ----------------------------------------------------------------------
        # 2. INGESTION ENGINE (DYNAMIC + STATIC)
        # ----------------------------------------------------------------------
        logger.info("Pipeline Step 1/5: Executing feature ingestion engine...")
        ingestion_engine = IngestionEngine(batch_size=2500)
        df_raw = ingestion_engine.run_full_extraction(
            start_date=start_date,
            end_date=end_date,
            soil_raster_path=soil_path,
            dem_raster_path=dem_path
        )

        if df_raw.empty:
            raise ValueError("Ingestion engine returned an empty dataset.")

        # ----------------------------------------------------------------------
        # 3. RAW FEATURE STAGING (BULK DB WRITE)
        # ----------------------------------------------------------------------
        logger.info(f"Pipeline Step 2/5: Staging raw feature vectors for {len(df_raw)} cells...")
        
        # Load cell mapping from DB
        cells_qs = SpatialGridCell.objects.all().values('id', 'cell_id', 'poverty_rate', 'malnutrition_rate', 'health_travel_time', 'pop_density', 'settlement_dist', 'healthcare_deficit')
        cell_map = {c['cell_id']: c for c in cells_qs}

        staging_objects = []
        for _, row in df_raw.iterrows():
            cid = int(row['cell_id'])
            cell_db = cell_map.get(cid)
            if not cell_db:
                continue

            staging_objects.append(
                RawGridFeatureStaging(
                    assessment_run=run_record,
                    grid_cell_id=cell_db['id'],
                    precip_mm=row.get('precip_mm', 0.0),
                    lst_day_celsius=row.get('lst_day_celsius', 0.0),
                    lst_night_celsius=row.get('lst_night_celsius', 0.0),
                    ndvi_val=row.get('ndvi_val', 0.0),
                    vertisol_percent=row.get('vertisol_percent', 0.0),
                    elevation_m=row.get('elevation_m', 0.0)
                )
            )

        with transaction.atomic():
            RawGridFeatureStaging.objects.bulk_create(staging_objects, batch_size=batch_size)

        # ----------------------------------------------------------------------
        # 4. NORMALIZATION & COMPONENT COMPOSITION
        # ----------------------------------------------------------------------
        logger.info("Pipeline Step 3/5: Normalizing component vectors...")
        
        # Extract ordered NumPy feature arrays
        cell_ids = df_raw['cell_id'].to_numpy(dtype=int)
        lst_day = df_raw['lst_day_celsius'].to_numpy(dtype=float)
        lst_night = df_raw['lst_night_celsius'].to_numpy(dtype=float)
        precip = df_raw['precip_mm'].to_numpy(dtype=float)
        soil = df_raw['vertisol_percent'].to_numpy(dtype=float)
        ndvi = df_raw['ndvi_val'].to_numpy(dtype=float)

        # Retrieve static cell attributes matching cell order
        pop_density = np.array([cell_map[cid]['pop_density'] for cid in cell_ids], dtype=float)
        settlement_dist = np.array([cell_map[cid]['settlement_dist'] for cid in cell_ids], dtype=float)
        poverty = np.array([cell_map[cid]['poverty_rate'] for cid in cell_ids], dtype=float)
        malnutrition = np.array([cell_map[cid]['malnutrition_rate'] for cid in cell_ids], dtype=float)
        health_travel_time = np.array([cell_map[cid]['health_travel_time'] for cid in cell_ids], dtype=float)
        healthcare_deficit = np.array([cell_map[cid]['healthcare_deficit'] for cid in cell_ids], dtype=float)

        normalizer = TRINormalizer()
        hazard_arr = normalizer.build_hazard(lst_day, lst_night, precip, soil, ndvi)
        exposure_arr = normalizer.build_exposure(pop_density, settlement_dist)
        vulnerability_arr = normalizer.build_vulnerability(poverty, malnutrition, health_travel_time)

        # ----------------------------------------------------------------------
        # 5. VECTORIZED TRI COMPUTATION
        # ----------------------------------------------------------------------
        logger.info("Pipeline Step 4/5: Executing TRI engine calculation...")
        tri_engine = TRIEngine()
        df_tri = tri_engine.execute_pipeline(
            cell_ids=cell_ids,
            hazard=hazard_arr,
            exposure=exposure_arr,
            vulnerability=vulnerability_arr,
            healthcare_deficit=healthcare_deficit
        )

        # ----------------------------------------------------------------------
        # 6. BULK PERSISTENCE TO SPATIAL GRID CELLS
        # ----------------------------------------------------------------------
        logger.info("Pipeline Step 5/5: Persisting risk scores to PostGIS grid cells...")
        
        cells_to_update = []
        for _, row in df_tri.iterrows():
            cid = int(row['cell_id'])
            cell_db_info = cell_map.get(cid)
            if not cell_db_info:
                continue

            cell_obj = SpatialGridCell(
                id=cell_db_info['id'],
                hazard_score=float(row['hazard_score']),
                exposure_score=float(row['exposure_score']),
                vulnerability_score=float(row['vulnerability_score']),
                tri_raw=float(row['tri_raw']),
                tri_final=float(row['tri_final']),
                risk_level=str(row['risk_level']),
                last_assessed_at=timezone.now()
            )
            cells_to_update.append(cell_obj)

        with transaction.atomic():
            SpatialGridCell.objects.bulk_update(
                cells_to_update,
                fields=['hazard_score', 'exposure_score', 'vulnerability_score', 'tri_raw', 'tri_final', 'risk_level', 'last_assessed_at'],
                batch_size=batch_size
            )

        # ----------------------------------------------------------------------
        # 7. AUTOMATED HIGH-RISK ALERT DISPATCH REGISTRATION
        # ----------------------------------------------------------------------
        high_risk_cells = df_tri[df_tri['risk_level'].isin(['HIGH', 'CRITICAL'])]
        alerts_created = 0

        if not high_risk_cells.empty:
            logger.info(f"Triggering automated alert generation for {len(high_risk_cells)} elevated risk cells...")
            high_risk_ids = [cell_map[cid]['id'] for cid in high_risk_cells['cell_id'] if cid in cell_map]
            
            # Fetch assigned health workers in affected grid cells
            workers = HealthWorker.objects.filter(assigned_cell_id__in=high_risk_ids, is_active=True)
            
            alert_logs = []
            for worker in workers:
                cell_row = high_risk_cells[high_risk_cells['cell_id'] == worker.assigned_cell.cell_id].iloc[0]
                alert_logs.append(
                    AlertDispatchLog(
                        assessment_run=run_record,
                        recipient=worker,
                        grid_cell=worker.assigned_cell,
                        alert_level=cell_row['risk_level'],
                        tri_score=cell_row['tri_final'],
                        dispatch_status='PENDING',
                        message_body=(
                            f"GeoAfya Alert [{cell_row['risk_level']} RISK]: Cell {worker.assigned_cell.cell_id} "
                            f"TRI score is {cell_row['tri_final']:.3f}. Immediate vector intervention advised."
                        )
                    )
                )

            if alert_logs:
                with transaction.atomic():
                    AlertDispatchLog.objects.bulk_create(alert_logs, batch_size=batch_size)
                alerts_created = len(alert_logs)

        # ----------------------------------------------------------------------
        # 8. MARK RUN AS COMPLETED
        # ----------------------------------------------------------------------
        completed_at = timezone.now()
        duration_sec = (completed_at - now).total_seconds()

        run_record.status = 'COMPLETED'
        run_record.completed_at = completed_at
        run_record.total_cells_processed = len(cells_to_update)
        run_record.critical_cell_count = int(np.sum(df_tri['risk_level'] == 'CRITICAL'))
        run_record.high_cell_count = int(np.sum(df_tri['risk_level'] == 'HIGH'))
        run_record.moderate_cell_count = int(np.sum(df_tri['risk_level'] == 'MODERATE'))
        run_record.low_cell_count = int(np.sum(df_tri['risk_level'] == 'LOW'))
        run_record.save()

        summary = {
            "run_id": run_record.id,
            "status": "COMPLETED",
            "duration_seconds": round(duration_sec, 2),
            "total_cells": len(cells_to_update),
            "risk_breakdown": {
                "CRITICAL": run_record.critical_cell_count,
                "HIGH": run_record.high_cell_count,
                "MODERATE": run_record.moderate_cell_count,
                "LOW": run_record.low_cell_count
            },
            "alerts_queued": alerts_created
        }
        
        logger.info(f"TRI Pipeline successfully executed in {duration_sec:.2f}s. Summary: {summary}")
        return summary

    except Exception as exc:
        logger.error(f"TRI Pipeline failed with error: {str(exc)}\n{traceback.format_exc()}")
        
        # Mark run as failed
        run_record.status = 'FAILED'
        run_record.completed_at = timezone.now()
        run_record.error_traceback = traceback.format_exc()
        run_record.save()

        # Retry task if retries remain
        raise self.retry(exc=exc)