import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy import integrate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ==== Ваши функции ====

def load_ftir_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]

def preprocess_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_smooth = savgol_filter(y, window_length=15, polyorder=3)
    coeffs = np.polyfit(x, y_smooth, deg=3)
    baseline = np.polyval(coeffs, x)
    y_corr = y_smooth - baseline
    y_norm = y_corr / np.max(np.abs(y_corr))
    return x, y_norm

def analyze_peaks(x: np.ndarray, y: np.ndarray,
                  min_prominence: float = 0.02,
                  min_distance: int = 5):
    peaks, props = find_peaks(y, prominence=min_prominence, distance=min_distance)
    widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)

    results = []
    for i, idx_peak in enumerate(peaks):
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

# ==== Научный стиль графика спектра с пиками ====

def plot_ftir_with_peaks_scientific(x, y, peaks_df, max_annotated_peaks: int = 10):
    """
    Научный стиль:
    - черный спектр
    - красные маркеры пиков
    - синяя линия FWHM + подпись ширины
    - подписи частоты и амплитуды для нескольких самых сильных пиков
    """
    # Немного более крупный шрифт
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    # Основной спектр
    ax.plot(x, y, linewidth=1.3, label="Processed FTIR spectrum")

    # Инвертируем, как принято для FTIR
    ax.invert_xaxis()

    # Отсортируем пики по амплитуде, чтобы подписывать самые значимые
    peaks_sorted = peaks_df.sort_values("amplitude", ascending=False).reset_index(drop=True)
    peaks_to_annotate = peaks_sorted.head(max_annotated_peaks)

    # Все пики — маркеры
    ax.plot(peaks_df["wavenumber"], peaks_df["amplitude"], "o", markersize=4, label="Peaks")

    # FWHM + подписи
    for _, row in peaks_df.iterrows():
        wn = row["wavenumber"]
        amp = row["amplitude"]
        left = row["left_wavenumber"]
        right = row["right_wavenumber"]

        # Синяя линия FWHM
        ax.hlines(y=amp * 0.5,
                  xmin=left,
                  xmax=right,
                  linestyles="--",
                  linewidth=1,
                  color="blue")

        # Подпись ширины (в см⁻¹)
        width_val = right - left
        ax.text((left + right) / 2,
                amp * 0.5,
                f"{width_val:.1f} cm⁻¹",
                ha="center",
                va="bottom",
                fontsize=8,
                color="blue")

    # Подписи частоты и амплитуды только для нескольких пиков,
    # чтобы не перегружать график
    for _, row in peaks_to_annotate.iterrows():
        wn = row["wavenumber"]
        amp = row["amplitude"]
        ax.text(wn,
                amp + 0.03,
                f"{wn:.0f} cm⁻¹\nI={amp:.2f}",
                ha="center",
                va="bottom")

    ax.set_title("FTIR spectrum with peak parameters")
    ax.set_xlabel("Wavenumber, cm⁻¹")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()

# ==== График аппроксимации одного пика гауссианой ====

def gaussian(x, A, x0, sigma):
    return A * np.exp(- (x - x0) ** 2 / (2 * sigma ** 2))

def lorentzian(x, A, x0, gamma):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

def pseudo_voigt(x, A, x0, w, eta):
    """
    Псевдо-Войгт:
        eta * Lorentz + (1 - eta) * Gauss
    w — эффективная ширина (используем её как σ и γ одного порядка).
    """
    sigma = w / 2.355 if w > 0 else 1.0  # связь FWHM и σ для гауссианы
    gamma = w / 2.0 if w > 0 else 1.0    # порядок величины для лоренца
    return eta * lorentzian(x, A, x0, gamma) + (1 - eta) * gaussian(x, A, x0, sigma)


def plot_single_peak_fit(x, y, peak_row, window_expand: float = 1.2):
    """
    Рисует второй график:
    - точки исходных данных вокруг пика
    - гауссовскую аппроксимацию (купол)
    - заливку площади под аппроксимированным пиком
    - текстом выводит площадь (численная и аналитическая)
    """
    wn = peak_row["wavenumber"]
    amp = peak_row["amplitude"]
    left = peak_row["left_wavenumber"]
    right = peak_row["right_wavenumber"]

    # Немного расширим окно относительно FWHM для устойчивого фита
    width = right - left
    fit_left = wn - width * window_expand
    fit_right = wn + width * window_expand

    mask = (x >= fit_left) & (x <= fit_right)
    x_win = x[mask]
    y_win = y[mask]

    # Начальные приближения для фита
    sigma_guess = width / 2.355 if width > 0 else (x_win.max() - x_win.min()) / 6
    p0 = [amp, wn, sigma_guess]

    # Фит гауссианы
    try:
        popt, pcov = curve_fit(gaussian, x_win, y_win, p0=p0)
    except RuntimeError:
        print("Не удалось аппроксимировать пик (curve_fit не сошёлся).")
        return

    A_fit, x0_fit, sigma_fit = popt

    # Тонкая сетка для гладкой кривой
    x_fit = np.linspace(x_win.min(), x_win.max(), 400)
    y_fit = gaussian(x_fit, A_fit, x0_fit, sigma_fit)

    # Численная площадь под аппроксимацией
    area_num = integrate.trapz(y_fit, x_fit)
    # Аналитическая площадь гауссианы
    area_analytic = A_fit * sigma_fit * np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Точки исходных данных
    ax.plot(x_win, y_win, "o", markersize=3, label="Data")

    # Гауссовский купол
    ax.plot(x_fit, y_fit, "-", linewidth=1.5, label="Gaussian fit")

    # Заливка площади под куполом
    ax.fill_between(x_fit, 0, y_fit, alpha=0.3, label="Peak area")

    ax.invert_xaxis()
    ax.set_title(f"Peak fit around {wn:.0f} cm⁻¹")
    ax.set_xlabel("Wavenumber, cm⁻¹")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Текст с параметрами
    text_str = (
        f"Peak at {wn:.1f} cm⁻¹\n"
        f"A ≈ {A_fit:.3f}\n"
        f"σ ≈ {sigma_fit:.2f} cm⁻¹\n"
        f"Area (numeric) ≈ {area_num:.3f}\n"
        f"Area (analytic) ≈ {area_analytic:.3f}"
    )
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", alpha=0.1))

    fig.tight_layout()
    plt.show()


def fit_peak_and_area(
    x: np.ndarray,
    y: np.ndarray,
    peak_row,
    window_expand: float = 1.5,
    baseline_margin: float = 0.4,
    plot: bool = True,
):
    """
    Улучшенная аппроксимация пика:
    - выбирает окно вокруг пика по FWHM
    - строит локальную линейную базовую линию
    - вычитает baseline
    - фит псевдо-Войгта
    - площадь под пиком = интеграл аппроксимирующей кривой по окну

    Parameters
    ----------
    x, y : полный спектр (после глобальной предварительной обработки)
    peak_row : строка из peaks_df (один пик)
    window_expand : во сколько раз расширять окно относительно FWHM
    baseline_margin : доля по краям окна, используемая только для оценки baseline
    plot : рисовать ли график

    Returns
    -------
    dict с параметрами:
        {
          "A": ..., "x0": ..., "w": ..., "eta": ...,
          "area": ...,       # площадь под псевдо-Войгтом
          "area_data": ...,  # площадь по исходным данным после вычитания baseline
        }
    """

    wn = peak_row["wavenumber"]
    left = peak_row["left_wavenumber"]
    right = peak_row["right_wavenumber"]

    # ширина по FWHM
    width = max(right - left, 1e-6)

    # окно для фита
    fit_left = wn - width * window_expand
    fit_right = wn + width * window_expand
    mask = (x >= fit_left) & (x <= fit_right)
    x_win = x[mask]
    y_win = y[mask]

    if len(x_win) < 10:
        raise RuntimeError("Слишком мало точек для фита вокруг пика.")

    # --- локальная базовая линия (линейная аппроксимация по краям окна) ---
    # используем крайние baseline_margin доли по x
    x_min, x_max = x_win.min(), x_win.max()
    x_left_max = x_min + (x_max - x_min) * baseline_margin
    x_right_min = x_max - (x_max - x_min) * baseline_margin

    baseline_mask = (x_win <= x_left_max) | (x_win >= x_right_min)
    x_base = x_win[baseline_mask]
    y_base = y_win[baseline_mask]

    if len(x_base) >= 2:
        # линейный baseline
        k, b = np.polyfit(x_base, y_base, deg=1)
        baseline = k * x_win + b
    else:
        # fallback: константа
        baseline = np.full_like(y_win, np.median(y_win))

    y_corr = y_win - baseline

    # --- начальные приближения для фита псевдо-Войгта ---
    A_guess = np.max(y_corr)
    x0_guess = wn
    w_guess = width
    eta_guess = 0.5  # смесь Гаусса и Лоренца

    p0 = [A_guess, x0_guess, w_guess, eta_guess]
    # Ограничения:
    # A > 0, w > 0, 0 <= eta <= 1
    bounds = (
        [0.0, x0_guess - width, 1e-6, 0.0],
        [np.inf, x0_guess + width, np.inf, 1.0],
    )

    # Фит псевдо-Войгта
    popt, pcov = curve_fit(
        pseudo_voigt,
        x_win, y_corr,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    A_fit, x0_fit, w_fit, eta_fit = popt

    # Сетка для гладкой кривой
    x_fit = np.linspace(x_win.min(), x_win.max(), 800)
    y_fit = pseudo_voigt(x_fit, A_fit, x0_fit, w_fit, eta_fit)

    # --- площади ---
    area_model = integrate.trapz(y_fit, x_fit)           # площадь под моделью
    area_data = integrate.trapz(y_corr, x_win)           # по эксперименту (после baseline)

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))

        # исходные данные и baseline
        ax.plot(x_win, y_win, "o", markersize=3, label="Data")
        ax.plot(x_win, baseline, "--", linewidth=1, label="Local baseline")

        # скорректированные точки
        ax.plot(x_win, y_corr, ".", markersize=3, label="Data - baseline")

        # модель
        ax.plot(x_fit, y_fit, "-", linewidth=1.5, label="Pseudo-Voigt fit")

        # заливка площади под моделью
        ax.fill_between(x_fit, 0, y_fit, alpha=0.3, label="Peak area (model)")

        ax.invert_xaxis()
        ax.set_title(f"Peak around {wn:.0f} cm⁻¹: improved fit and area")
        ax.set_xlabel("Wavenumber, cm⁻¹")
        ax.set_ylabel("Normalized intensity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        text_str = (
            f"x₀ ≈ {x0_fit:.1f} cm⁻¹\n"
            f"A ≈ {A_fit:.3f}\n"
            f"w ≈ {w_fit:.2f} (eff. width)\n"
            f"η ≈ {eta_fit:.2f}\n"
            f"Area(model) ≈ {area_model:.3f}\n"
            f"Area(data)  ≈ {area_data:.3f}"
        )
        ax.text(
            0.02, 0.98, text_str,
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", alpha=0.1),
        )

        fig.tight_layout()
        plt.show()

    return {
        "A": A_fit,
        "x0": x0_fit,
        "w": w_fit,
        "eta": eta_fit,
        "area": area_model,
        "area_data": area_data,
    }

# ==== Пример использования ====

if __name__ == "__main__":
    path = r"data/akrileta_rapsu_ella.csv"
    x_raw, y_raw = load_ftir_csv(path=path)
    x_proc, y_proc = preprocess_spectrum(x_raw, y_raw)
    peaks_df = analyze_peaks(x_proc, y_proc)

    # График всего спектра с «научным» оформлением
    plot_ftir_with_peaks_scientific(x_proc, y_proc, peaks_df, max_annotated_peaks=8)

    # Второй график: аппроксимация одного выбранного пика
    # Например, самого сильного:
    main_peak = peaks_df.sort_values("amplitude", ascending=False).iloc[0]
    plot_single_peak_fit(x_proc, y_proc, main_peak)

    result = fit_peak_and_area(x_proc, y_proc, main_peak, plot=True)
    print(result)