from django.contrib.gis.db import models


# ==============================================================================
# 1. SPATIAL GRID & ADMINISTRATIVE BOUNDARIES
# ==============================================================================

class SpatialGridCell(models.Model):
    """
    Represents one of the N=22,584 standardized spatial grid cells across the 
    surveillance region. Acts as the primary spatial anchor for all environmental 
    and epidemiological calculations.
    """
    cell_id = models.IntegerField(unique=True, db_index=True)
    geom = models.PolygonField(srid=4326)  # Boundary extent (WGS 84)
    centroid = models.PointField(srid=4326) # Cell center point for rapid proximity queries
    district_name = models.CharField(max_length=100, db_index=True)
    has_population = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Spatial Grid Cell"
        verbose_name_plural = "Spatial Grid Cells"
        indexes = [
            models.Index(fields=['cell_id']),
            models.Index(fields=['district_name']),
        ]

    def __str__(self):
        return f"Cell #{self.cell_id} ({self.district_name})"


class AdministrativeRegion(models.Model):
    """
    Represents official administrative boundaries (Counties, Sub-Counties, Wards) 
    used for spatial intersections and health worker dispatch routing.
    """
    LEVEL_CHOICES = [
        ('COUNTY', 'County'),
        ('SUB_COUNTY', 'Sub-County'),
        ('WARD', 'Ward'),
    ]

    name = models.CharField(max_length=150)
    code = models.CharField(max_length=50, unique=True, db_index=True)  # e.g. 'KE-38-TURKANA-WEST'
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default='SUB_COUNTY')
    geom = models.MultiPolygonField(srid=4326)  # Official boundary geometry

    class Meta:
        verbose_name = "Administrative Region"
        verbose_name_plural = "Administrative Regions"
        indexes = [
            models.Index(fields=['code', 'level']),
        ]

    def __str__(self):
        return f"{self.name} [{self.get_level_display()}]"


# ==============================================================================
# 2. FEATURE INGESTION & DATA STAGING
# ==============================================================================

class RawGridFeatureStaging(models.Model):
    """
    Landing zone for raw, unscaled environmental and climate values extracted 
    per cell per timestep prior to Phase 2 normalization.
    """
    cell = models.ForeignKey(
        SpatialGridCell, 
        on_delete=models.CASCADE, 
        related_name='staged_features'
    )
    time_step = models.DateField(db_index=True)  # Timestep date (e.g. 2026-07-01)

    # Dynamic Climate Metrics (Extracted via Google Earth Engine API)
    lst_day_celsius = models.FloatField(null=True, blank=True)   # MODIS Daytime LST (°C)
    lst_night_celsius = models.FloatField(null=True, blank=True) # MODIS Nighttime LST (°C)
    precip_mm = models.FloatField(null=True, blank=True)         # CHIRPS Cumulative Rainfall (mm)
    ndvi_val = models.FloatField(null=True, blank=True)          # MODIS/Sentinel Vegetation Index

    # Static Biophysical Metrics (Extracted from GeoTIFF Rasters)
    vertisol_percent = models.FloatField(null=True, blank=True)  # ISRIC Soil Vertisol/Clay Content (%)
    elevation_m = models.FloatField(null=True, blank=True)       # SRTM Elevation (meters)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Raw Grid Feature Staging"
        verbose_name_plural = "Raw Grid Feature Staging Records"
        unique_together = ('cell', 'time_step')
        indexes = [
            models.Index(fields=['time_step', 'cell']),
        ]

    def __str__(self):
        return f"Raw Staging - Cell #{self.cell.cell_id} [{self.time_step}]"


# ==============================================================================
# 3. EPIDEMIOLOGICAL RISK ASSESSMENT ENGINE
# ==============================================================================

class RiskAssessmentRun(models.Model):
    """
    Stores normalized components (Hazard, Exposure, Vulnerability), calculated raw TRI, 
    Step 4 logistic multipliers, and dual risk products for each grid cell per timestep.
    """
    cell = models.ForeignKey(
        SpatialGridCell, 
        on_delete=models.CASCADE, 
        related_name='risk_assessments'
    )
    time_step = models.DateField(db_index=True)

    # Step 3 Component Arrays (Normalized to [0.0, 1.0])
    hazard_score = models.FloatField()
    exposure_score = models.FloatField()
    vulnerability_score = models.FloatField()
    raw_tri = models.FloatField()  # TRI = Hazard × Exposure × Vulnerability

    # Step 4 Ingest & Multipliers
    healthcare_deficit_index = models.FloatField(default=0.0) # HD in [0.0, 1.0]
    surveillance_multiplier = models.FloatField(default=1.0)  # SM in [1.0, 3.0]

    # Final Products
    primary_risk_score = models.FloatField()   # Pure Environmental Risk Map
    secondary_risk_score = models.FloatField() # Operational Priority Overlay Map

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Risk Assessment Run"
        verbose_name_plural = "Risk Assessment Runs"
        unique_together = ('cell', 'time_step')
        indexes = [
            models.Index(fields=['time_step', 'secondary_risk_score']),
            models.Index(fields=['time_step', 'primary_risk_score']),
        ]

    def __str__(self):
        return f"Risk Run - Cell #{self.cell.cell_id} [{self.time_step}] - SecScore: {self.secondary_risk_score:.2f}"


# ==============================================================================
# 4. HEALTH WORKER REGISTRY & ALERT DISPATCH LOGS
# ==============================================================================

class HealthWorker(models.Model):
    """
    Registry of field personnel, community health promoters, and epidemiologists 
    mapped to administrative regions for targeted alert dispatches.
    """
    ROLE_CHOICES = [
        ('CHP', 'Community Health Promoter'),
        ('EPI', 'Sub-County Epidemiologist'),
        ('FACILITY_LEAD', 'Health Facility In-Charge'),
        ('NGO_COORD', 'Vector Control / NGO Coordinator'),
    ]

    CHANNEL_CHOICES = [
        ('SMS', 'SMS Text Message'),
        ('WHATSAPP', 'WhatsApp Message'),
        ('EMAIL', 'Email Notification'),
    ]

    full_name = models.CharField(max_length=200)
    phone_number = models.CharField(max_length=20, db_index=True)  # E.164 Format: +2547XXXXXXXX
    email = models.EmailField(blank=True, null=True)
    role = models.CharField(max_length=30, choices=ROLE_CHOICES, default='CHP')
    preferred_channel = models.CharField(max_length=10, choices=CHANNEL_CHOICES, default='SMS')
    
    # Regional Assignment
    assigned_regions = models.ManyToManyField(
        AdministrativeRegion, 
        related_name='health_workers',
        blank=True
    )
    assigned_facility = models.CharField(max_length=200, blank=True, null=True)
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Health Worker"
        verbose_name_plural = "Health Worker Registry"

    def __str__(self):
        return f"{self.full_name} - {self.get_role_display()} ({self.phone_number})"


class AlertDispatchLog(models.Model):
    """
    Audit log tracking all outgoing emergency advisories dispatched to health personnel.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('DISPATCHED', 'Dispatched'),
        ('FAILED', 'Failed'),
    ]

    worker = models.ForeignKey(
        HealthWorker, 
        on_delete=models.CASCADE, 
        related_name='alert_logs'
    )
    grid_cell = models.ForeignKey(
        SpatialGridCell, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    risk_score = models.FloatField()
    message_body = models.TextField()
    channel_used = models.CharField(max_length=10)
    dispatch_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    provider_sid = models.CharField(max_length=100, blank=True, null=True)  # Twilio/Gateway SID
    dispatched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Alert Dispatch Log"
        verbose_name_plural = "Alert Dispatch Logs"
        ordering = ['-dispatched_at']

    def __str__(self):
        return f"Alert to {self.worker.full_name} - Status: {self.dispatch_status} [{self.dispatched_at.strftime('%Y-%m-%d %H:%M')}]"