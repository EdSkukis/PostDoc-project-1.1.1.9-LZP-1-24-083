import numpy as np
from typing import Tuple, Optional


def calculate_youngs_modulus(strain: np.ndarray, stress: np.ndarray,
                             window: Tuple[float, float], mode: str = "strain"):
    """
    Calculates E and returns anchor points.
    mode: 'strain' (default) or 'stress'
    """
    if mode == "stress":
        mask = (stress >= window[0]) & (stress <= window[1])
    else:
        mask = (strain >= window[0]) & (strain <= window[1])

    x, y = strain[mask], stress[mask]
    if len(x) < 2: return None, None, None

    e_mod, intercept = np.polyfit(x, y, 1)
    e_points = {
        "start": (float(x[0]), float(y[0])),
        "end": (float(x[-1]), float(y[-1]))
    }
    return float(e_mod), float(intercept), e_points


def calculate_rp_offset(strain: np.ndarray, stress: np.ndarray, e_mod: float, offset_val: float):
    """
    Returns:
    - rp_stress (float)
    - rp_strain (float)
    """
    if not e_mod or e_mod <= 0: return None, None

    target_line = e_mod * (strain - offset_val)
    diff = stress - target_line

    for i in range(1, len(diff)):
        if np.sign(diff[i - 1]) != np.sign(diff[i]):
            ratio = abs(diff[i - 1]) / (abs(diff[i - 1]) + abs(diff[i]))

            # Находим координаты точки пересечения
            rp_stress = float(stress[i - 1] + ratio * (stress[i] - stress[i - 1]))
            rp_strain = float(strain[i - 1] + ratio * (strain[i] - strain[i - 1]))

            return rp_stress, rp_strain
    return None, None