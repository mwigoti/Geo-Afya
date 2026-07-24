import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. DYNAMIC CONFIGURATION & AHP WEIGHT SPECIFICATIONS
# ==============================================================================

@dataclass(frozen=True)
class AHPWeights:
    """
    Analytical Hierarchy Process (AHP) weights and domain bounds.
    Supports dynamic overrides via Django settings (GEOAFYA_RISK_CONFIG).
    """
    # --- HAZARD (H) SUB-INDICATOR WEIGHTS ---
    W_LST_DAY: float = 0.25      # Daytime LST (°C)
    W_LST_NIGHT: float = 0.15    # Nighttime LST (°C)
    W_PRECIP: float = 0.30       # Cumulative Precipitation (mm)
    W_SOIL: float = 0.15         # Vertisol / Clay Content (%)
    W_NDVI: float = 0.15         # Vegetation Canopy Index

    # --- EXPOSURE (E) SUB-INDICATOR WEIGHTS ---
    W_POP_DENSITY: float = 0.70  # Population Density (people/km²)
    W_SETTLEMENT: float = 0.30   # Proximity to Rural Settlements (km)

    # --- VULNERABILITY (V) SUB-INDICATOR WEIGHTS ---
    W_POVERTY: float = 0.40      # Household Poverty Rate (%)
    W_MALNUTRITION: float = 0.35 # Acute Malnutrition Rate (%)
    W_HEALTH_ACCESS: float = 0.25# Travel Time to Nearest Health Facility (mins)

    # --- DOMAIN BOUNDARIES (Min, Max) FOR SCALING ---
    LST_DAY_RANGE: Tuple[float, float] = (18.0, 42.0)
    LST_NIGHT_RANGE: Tuple[float, float] = (12.0, 30.0)
    PRECIP_RANGE: Tuple[float, float] = (0.0, 350.0)
    SOIL_RANGE: Tuple[float, float] = (0.0, 100.0)
    NDVI_RANGE: Tuple[float, float] = (-0.1, 0.85)

    POP_DENSITY_RANGE: Tuple[float, float] = (0.0, 1000.0)
    SETTLEMENT_DIST_RANGE: Tuple[float, float] = (0.0, 50.0)

    POVERTY_RANGE: Tuple[float, float] = (0.0, 100.0)
    MALNUTRITION_RANGE: Tuple[float, float] = (0.0, 50.0)
    HEALTH_ACCESS_RANGE: Tuple[float, float] = (0.0, 240.0)

    def __post_init__(self):
        """
        Validates weight sum constraints within each component domain.
        """
        h_sum = self.W_LST_DAY + self.W_LST_NIGHT + self.W_PRECIP + self.W_SOIL + self.W_NDVI
        e_sum = self.W_POP_DENSITY + self.W_SETTLEMENT
        v_sum = self.W_POVERTY + self.W_MALNUTRITION + self.W_HEALTH_ACCESS

        if not np.isclose(h_sum, 1.0, atol=1e-3):
            raise ValueError(f"Hazard sub-indicator weights must sum to 1.0 (got {h_sum:.4f})")
        if not np.isclose(e_sum, 1.0, atol=1e-3):
            raise ValueError(f"Exposure sub-indicator weights must sum to 1.0 (got {e_sum:.4f})")
        if not np.isclose(v_sum, 1.0, atol=1e-3):
            raise ValueError(f"Vulnerability sub-indicator weights must sum to 1.0 (got {v_sum:.4f})")

    @classmethod
    def load_from_settings(cls) -> "AHPWeights":
        """
        Factory method to override baseline AHP weights from Django settings.GEOAFYA_RISK_CONFIG
        if present.
        """
        config = getattr(settings, 'GEOAFYA_RISK_CONFIG', {})
        if not config:
            return cls()
        
        # Override fields present in settings configuration dictionary
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)


# ==============================================================================
# 2. HARDENED FEATURE SCALER
# ==============================================================================

class RobustScaler:
    """
    Production-grade vectorized array scaler with robust outlier Winsorization,
    NaN/Inf imputation, and boundary clipping.
    """

    @staticmethod
    def clean_array(
        arr: np.ndarray, 
        feature_name: str, 
        nan_fill: float = 0.0,
        max_nan_ratio: float = 0.30
    ) -> np.ndarray:
        """
        Cleans input NumPy array: handles NaN, +Inf, -Inf, and alerts if missing data threshold is exceeded.
        """
        if arr is None or arr.size == 0:
            logger.warning(f"Feature '{feature_name}' received an empty or None array.")
            return np.array([], dtype=np.float64)

        # Convert to float64 array safely
        clean_arr = np.array(arr, dtype=np.float64, copy=True)

        # Check data quality
        invalid_mask = np.isnan(clean_arr) | np.isinf(clean_arr)
        nan_ratio = np.count_nonzero(invalid_mask) / clean_arr.size

        if nan_ratio > max_nan_ratio:
            logger.error(
                f"DATA QUALITY ALERT: Feature '{feature_name}' has {nan_ratio:.1%} "
                f"missing/corrupt pixels (exceeds threshold of {max_nan_ratio:.1%})."
            )
        elif nan_ratio > 0:
            logger.info(f"Feature '{feature_name}': Imputing {nan_ratio:.2%} missing values with {nan_fill}.")

        # Replace invalid values (NaN, +/-Inf) with fallback default
        clean_arr[invalid_mask] = nan_fill
        return clean_arr

    @classmethod
    def min_max_scale(
        cls,
        arr: np.ndarray,
        min_val: float,
        max_val: float,
        feature_name: str = "feature",
        invert: bool = False,
        nan_fill: float = 0.0,
        quantile_clip: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Normalizes a feature vector to the $[0.0, 1.0]$ interval with optional Winsorization.

        $$S_i = \text{clip}\left(\frac{x_i - x_{\min}}{x_{\max} - x_{\min}}, 0.0, 1.0\right)$$

        :param arr: Raw feature vector ($N$ cells).
        :param min_val: Domain minimum threshold.
        :param max_val: Domain maximum threshold.
        :param feature_name: Identifier for logging context.
        :param invert: If True, lower raw values yield higher normalized scores.
        :param nan_fill: Fallback value for missing values.
        :param quantile_clip: Optional tuple e.g., (1.0, 99.0) to clip extreme satellite outliers before scaling.
        :return: Scaled NumPy array bounded strictly in $[0.0, 1.0]$.
        """
        clean_arr = cls.clean_array(arr, feature_name, nan_fill=nan_fill)
        if clean_arr.size == 0:
            return clean_arr

        # Optional Winsorization: Clip extreme percentiles (e.g., 1st and 99th) to eliminate cloud artifacts
        if quantile_clip is not None:
            lower_pct, upper_pct = quantile_clip
            q_low = np.percentile(clean_arr, lower_pct)
            q_high = np.percentile(clean_arr, upper_pct)
            clean_arr = np.clip(clean_arr, q_low, q_high)

        denom = max_val - min_val
        if np.isclose(denom, 0.0):
            logger.warning(
                f"Zero dynamic range for '{feature_name}' (min_val == max_val == {min_val}). "
                "Returning zeros array."
            )
            return np.zeros_like(clean_arr, dtype=np.float64)

        # Scale array
        if invert:
            scaled = (max_val - clean_arr) / denom
        else:
            scaled = (clean_arr - min_val) / denom

        # Strict floor and ceiling enforcement
        return np.clip(scaled, 0.0, 1.0)


# ==============================================================================
# 3. TRI DOMAIN NORMALIZER ORCHESTRATOR
# ==============================================================================

class TRINormalizer:
    """
    Constructs normalized component vectors ($H, E, V$) across $N$ spatial cells
    ready for mathematical risk calculation.
    """

    def __init__(self, weights: Optional[AHPWeights] = None):
        self.w = weights if weights is not None else AHPWeights.load_from_settings()
        self.scaler = RobustScaler()

    def build_hazard(
        self,
        lst_day: np.ndarray,
        lst_night: np.ndarray,
        precip: np.ndarray,
        soil: np.ndarray,
        ndvi: np.ndarray
    ) -> np.ndarray:
        """
        Calculates normalized Hazard vector ($H$) using weighted sum of climate & environmental indicators.
        """
        s_lst_day = self.scaler.min_max_scale(
            lst_day, self.w.LST_DAY_RANGE[0], self.w.LST_DAY_RANGE[1],
            feature_name="LST_Day", nan_fill=self.w.LST_DAY_RANGE[0], quantile_clip=(0.5, 99.5)
        )
        s_lst_night = self.scaler.min_max_scale(
            lst_night, self.w.LST_NIGHT_RANGE[0], self.w.LST_NIGHT_RANGE[1],
            feature_name="LST_Night", nan_fill=self.w.LST_NIGHT_RANGE[0], quantile_clip=(0.5, 99.5)
        )
        s_precip = self.scaler.min_max_scale(
            precip, self.w.PRECIP_RANGE[0], self.w.PRECIP_RANGE[1],
            feature_name="Precipitation", nan_fill=0.0, quantile_clip=(0.0, 99.5)
        )
        s_soil = self.scaler.min_max_scale(
            soil, self.w.SOIL_RANGE[0], self.w.SOIL_RANGE[1],
            feature_name="Vertisol_Soil", nan_fill=0.0
        )
        s_ndvi = self.scaler.min_max_scale(
            ndvi, self.w.NDVI_RANGE[0], self.w.NDVI_RANGE[1],
            feature_name="NDVI", nan_fill=0.0
        )

        hazard = (
            (self.w.W_LST_DAY * s_lst_day) +
            (self.w.W_LST_NIGHT * s_lst_night) +
            (self.w.W_PRECIP * s_precip) +
            (self.w.W_SOIL * s_soil) +
            (self.w.W_NDVI * s_ndvi)
        )

        return np.clip(hazard, 0.0, 1.0)

    def build_exposure(
        self,
        pop_density: np.ndarray,
        settlement_dist: np.ndarray
    ) -> np.ndarray:
        """
        Calculates normalized Exposure vector ($E$) using population density and settlement proximity.
        """
        s_pop = self.scaler.min_max_scale(
            pop_density, self.w.POP_DENSITY_RANGE[0], self.w.POP_DENSITY_RANGE[1],
            feature_name="Pop_Density", nan_fill=0.0, quantile_clip=(0.0, 99.0)
        )
        # Inverted indicator: Closer distance to human settlement = higher exposure risk
        s_settle = self.scaler.min_max_scale(
            settlement_dist, self.w.SETTLEMENT_DIST_RANGE[0], self.w.SETTLEMENT_DIST_RANGE[1],
            feature_name="Settlement_Proximity", invert=True, nan_fill=self.w.SETTLEMENT_DIST_RANGE[1]
        )

        exposure = (self.w.W_POP_DENSITY * s_pop) + (self.w.W_SETTLEMENT * s_settle)

        return np.clip(exposure, 0.0, 1.0)

    def build_vulnerability(
        self,
        poverty: np.ndarray,
        malnutrition: np.ndarray,
        health_travel_time: np.ndarray
    ) -> np.ndarray:
        """
        Calculates normalized Vulnerability vector ($V$) using socio-economic deficit indicators.
        """
        s_poverty = self.scaler.min_max_scale(
            poverty, self.w.POVERTY_RANGE[0], self.w.POVERTY_RANGE[1],
            feature_name="Poverty_Rate", nan_fill=0.0
        )
        s_malnutrition = self.scaler.min_max_scale(
            malnutrition, self.w.MALNUTRITION_RANGE[0], self.w.MALNUTRITION_RANGE[1],
            feature_name="Malnutrition_Rate", nan_fill=0.0
        )
        s_access = self.scaler.min_max_scale(
            health_travel_time, self.w.HEALTH_ACCESS_RANGE[0], self.w.HEALTH_ACCESS_RANGE[1],
            feature_name="Health_Travel_Time", nan_fill=0.0
        )

        vulnerability = (
            (self.w.W_POVERTY * s_poverty) +
            (self.w.W_MALNUTRITION * s_malnutrition) +
            (self.w.W_HEALTH_ACCESS * s_access)
        )

        return np.clip(vulnerability, 0.0, 1.0)