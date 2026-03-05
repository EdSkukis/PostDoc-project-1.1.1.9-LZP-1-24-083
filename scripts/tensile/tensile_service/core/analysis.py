import numpy as np


def detect_outliers(results: list, threshold: float = 2.0):
    """
    Marks tests as outliers if their E-modulus deviates significantly from the mean.
    """
    moduli = [r["E_MPa"] for r in results if r["E_MPa"]]
    if len(moduli) < 3: return results

    avg = np.mean(moduli)
    std = np.std(moduli)

    for r in results:
        if r["E_MPa"]:
            dev = abs(r["E_MPa"] - avg)
            r["is_outlier"] = bool(dev > threshold * std)
    return results