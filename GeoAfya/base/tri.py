import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. CONFIGURATION & PARAMETERS
# ==============================================================================

@dataclass(frozen=True)
class TRIEngineConfig:
    """
    Configuration parameters for the TRI multiplicative model, Step 4 logistic 
    saturation adjustment, and risk classification thresholds.
    """
    # Logistic Saturation Parameters
    LOGISTIC_K: float = 10.0        # Growth steepness of the saturation curve
    LOGISTIC_D0: float = 0.50       # Inflection point threshold for Healthcare Deficit (D)
    MAX_AMPLIFICATION_L: float = 0.40 # Maximum risk boost factor (+40% upper cap)

    # Risk Classification Category Boundaries
    THRESH_LOW: float = 0.25
    THRESH_MODERATE: float = 0.50
    THRESH_HIGH: float = 0.75
    # Scores > THRESH_HIGH are classified as 'CRITICAL'

    @classmethod
    def load_from_settings(cls) -> "TRIEngineConfig":
        """
        Loads configuration overrides from Django settings.GEOAFYA_TRI_CONFIG if available.
        """
        config_dict = getattr(settings, 'GEOAFYA_TRI_CONFIG', {})
        if not config_dict:
            return cls()
        
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# 2. CORE VECTORIZED TRI ENGINE
# ==============================================================================

class TRIEngine:
    """
    High-performance NumPy vectorized engine for computing multi-domain TRI risk 
    vectors, applying non-linear healthcare saturation scaling, and categorizing alert levels.
    """

    def __init__(self, config: Optional[TRIEngineConfig] = None):
        self.cfg = config if config is not None else TRIEngineConfig.load_from_settings()

    def validate_inputs(
        self, 
        hazard: np.ndarray, 
        exposure: np.ndarray, 
        vulnerability: np.ndarray, 
        healthcare_deficit: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Enforces shape alignment, type conversion, NaN/Inf imputation, 
        and boundary clipping across input arrays ($N$ cells).
        """
        h = np.nan_to_num(np.array(hazard, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        e = np.nan_to_num(np.array(exposure, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        v = np.nan_to_num(np.array(vulnerability, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)

        n_cells = h.size
        if e.size != n_cells or v.size != n_cells:
            raise ValueError(
                f"Array dimension mismatch: Hazard ({h.size}), Exposure ({e.size}), "
                f"Vulnerability ({v.size}) must all match length N."
            )

        if healthcare_deficit is None:
            d = np.zeros(n_cells, dtype=np.float64)
        else:
            d = np.nan_to_num(np.array(healthcare_deficit, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
            if d.size != n_cells:
                raise ValueError(f"Healthcare Deficit array size ({d.size}) does not match grid size ({n_cells}).")

        # Ensure all arrays are bounded strictly within [0.0, 1.0]
        return (
            np.clip(h, 0.0, 1.0),
            np.clip(e, 0.0, 1.0),
            np.clip(v, 0.0, 1.0),
            np.clip(d, 0.0, 1.0)
        )

    def compute_raw_tri(self, h: np.ndarray, e: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Computes the base multiplicative risk index vector:
        TRI_raw = H * E * V
        """
        tri_raw = h * e * v
        return np.clip(tri_raw, 0.0, 1.0)

    def apply_logistic_saturation(self, tri_raw: np.ndarray, healthcare_deficit: np.ndarray) -> np.ndarray:
        """
        Applies Step 4 Logistic Saturation Scaling based on Healthcare Deficit (D).

        Amplification Factor = L / (1 + exp(-k * (D - D0)))
        TRI_final = clip(TRI_raw * (1 + Amplification Factor), 0.0, 1.0)
        """
        k = self.cfg.LOGISTIC_K
        d0 = self.cfg.LOGISTIC_D0
        l_max = self.cfg.MAX_AMPLIFICATION_L

        # Compute logistic sigmoid amplification shift
        exponent = -k * (healthcare_deficit - d0)
        # Numerically stable sigmoid computation
        amplification_factor = l_max / (1.0 + np.exp(exponent))

        tri_saturated = tri_raw * (1.0 + amplification_factor)
        return np.clip(tri_saturated, 0.0, 1.0)

    def classify_risk_levels(self, tri_final: np.ndarray) -> np.ndarray:
        """
        Categorizes continuous TRI risk scores into discrete alert tiers:
        - LOW      : [0.00, THRESH_LOW)
        - MODERATE : [THRESH_LOW, THRESH_MODERATE)
        - HIGH     : [THRESH_MODERATE, THRESH_HIGH)
        - CRITICAL : [THRESH_HIGH, 1.00]
        """
        conditions = [
            tri_final < self.cfg.THRESH_LOW,
            (tri_final >= self.cfg.THRESH_LOW) & (tri_final < self.cfg.THRESH_MODERATE),
            (tri_final >= self.cfg.THRESH_MODERATE) & (tri_final < self.cfg.THRESH_HIGH),
            tri_final >= self.cfg.THRESH_HIGH
        ]
        choices = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']

        return np.select(conditions, choices, default='LOW')

    def execute_pipeline(
        self,
        cell_ids: np.ndarray,
        hazard: np.ndarray,
        exposure: np.ndarray,
        vulnerability: np.ndarray,
        healthcare_deficit: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Executes end-to-end vector calculation across N grid cells and returns 
        a structured DataFrame containing all component vectors and final risk classifications.

        :param cell_ids: Array of unique cell identification numbers.
        :param hazard: Raw/Normalized Hazard scores array.
        :param exposure: Raw/Normalized Exposure scores array.
        :param vulnerability: Raw/Normalized Vulnerability scores array.
        :param healthcare_deficit: Optional Healthcare Deficit Index array (D).
        :return: Pandas DataFrame formatted for DB persistence or Celery pipeline downstream.
        """
        logger.info(f"Executing vectorized TRI risk model across {len(cell_ids)} cells...")

        # 1. Input sanitization & validation
        h, e, v, d = self.validate_inputs(hazard, exposure, vulnerability, healthcare_deficit)

        # 2. Multiplicative TRI calculation
        tri_raw = self.compute_raw_tri(h, e, v)

        # 3. Step 4 Logistic Saturation adjustment
        tri_final = self.apply_logistic_saturation(tri_raw, d)

        # 4. Discrete risk level categorization
        risk_levels = self.classify_risk_levels(tri_final)

        # 5. Build output DataFrame
        results_df = pd.DataFrame({
            'cell_id': cell_ids,
            'hazard_score': h,
            'exposure_score': e,
            'vulnerability_score': v,
            'healthcare_deficit': d,
            'tri_raw': tri_raw,
            'tri_final': tri_final,
            'risk_level': risk_levels
        })

        logger.info(
            f"TRI calculation complete. Risk Breakdown -> "
            f"CRITICAL: {np.sum(risk_levels == 'CRITICAL')}, "
            f"HIGH: {np.sum(risk_levels == 'HIGH')}, "
            f"MODERATE: {np.sum(risk_levels == 'MODERATE')}, "
            f"LOW: {np.sum(risk_levels == 'LOW')}"
        )

        return results_df