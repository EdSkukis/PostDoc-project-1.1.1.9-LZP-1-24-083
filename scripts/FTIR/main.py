import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy import integrate

def load_ftir_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    # Названия колонок подстройте под свой формат
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    # Сортируем по x (на всякий случай)
    idx = np.argsort(x)
    return x[idx], y[idx]

def preprocess_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Сглаживание (окно и порядок подобрать под ваши данные)
    y_smooth = savgol_filter(y, window_length=15, polyorder=3)

    # Простая базовая линия: полином 3 порядка
    # (для серьёзного проекта лучше добавить AsLS/ALS)
    coeffs = np.polyfit(x, y_smooth, deg=3)
    baseline = np.polyval(coeffs, x)

    y_corr = y_smooth - baseline
    # Нормировка по максимуму
    y_norm = y_corr / np.max(np.abs(y_corr))

    return x, y_norm

def analyze_peaks(x: np.ndarray, y: np.ndarray,
                  min_prominence: float = 0.02,
                  min_distance: int = 5):
    # Поиск пиков
    peaks, props = find_peaks(y, prominence=min_prominence, distance=min_distance)

    # Вычисляем ширину на полувысоте
    widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)

    results = []
    for i, idx_peak in enumerate(peaks):
        # конвертируем позиции левой/правой границы в индексы
        left_idx = int(np.floor(left_ips[i]))
        right_idx = int(np.ceil(right_ips[i]))
        left_idx = max(left_idx, 0)
        right_idx = min(right_idx, len(x) - 1)

        x_slice = x[left_idx:right_idx + 1]
        y_slice = y[left_idx:right_idx + 1]

        area = integrate.trapz(y_slice, x_slice)

        results.append({
            "peak_index": int(idx_peak),
            "wavenumber": float(x[idx_peak]),
            "amplitude": float(y[idx_peak]),
            "prominence": float(props["prominences"][i]),
            "fwhm_points": float(widths[i]),
            "left_wavenumber": float(x[left_idx]),
            "right_wavenumber": float(x[right_idx]),
            "area": float(area),
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    x_raw, y_raw = load_ftir_csv("spectrum_example.csv")
    x_proc, y_proc = preprocess_spectrum(x_raw, y_raw)
    peaks_df = analyze_peaks(x_proc, y_proc)
    print(peaks_df.head())
