import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def setup_scientific_style():
    """Установка научного стиля графиков."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.fontsize": 8,
        "figure.dpi": 200
    })


def plot_individual_test(df, save_path, E, intercept, e_pts, rp_stress, rp_offset_pct, test_id):
    plt.figure(figsize=(12, 8))

    # 1. Основная кривая эксперимента
    plt.plot(df['strain_pct'], df['stress_mpa'], label=f'Test {test_id}', color='royalblue', lw=2, zorder=1)

    # Максимальные значения для расчета лимитов
    idx_max = df['stress_mpa'].idxmax()
    uts_stress = df.loc[idx_max, 'stress_mpa']
    uts_strain_pct = df.loc[idx_max, 'strain_pct']

    curr_xlim = max(df['strain_pct']) * 1.05
    curr_ylim = uts_stress * 1.1

    if E:
        # 2. Продление линии модуля Юнга (E) до уровня UTS
        x_max_e = (uts_stress - intercept) / E
        x_range_e = np.array([0, x_max_e])
        y_range_e = E * x_range_e + intercept
        plt.plot(x_range_e * 100, y_range_e, color='red', linestyle='--', alpha=0.5,
                 label=f'Slope E: {round(E, 1)} MPa', lw=1)

        # 3. Границы расчета E (Красные пунктиры до графика + проекции на Y)
        for point_name in ['start', 'end']:
            px, py = e_pts[point_name]
            px_pct = px * 100

            # Вертикаль до графика
            plt.plot([px_pct, px_pct], [0, py], color='red', linestyle=':', lw=1.2)
            # Горизонталь до оси Y
            plt.plot([0, px_pct], [py, py], color='red', linestyle=':', lw=1.2)

            # Подписи значений границ на осях
            plt.text(px_pct, -curr_ylim * 0.02, f'{round(px_pct, 2)}%', color='red', fontsize=8, ha='center')
            plt.text(-curr_xlim * 0.02, py, f'{round(py, 1)}', color='red', fontsize=8, va='center', ha='right')

    # 4. Линия смещения Rp и зеленые проекции
    if rp_stress and E:
        offset_fraction = rp_offset_pct / 100.0
        rp_strain_pct = ((rp_stress - intercept) / E + offset_fraction) * 100

        # Линия смещения
        x_rp_line = np.array([offset_fraction, (uts_stress - intercept) / E + offset_fraction])
        y_rp_line = E * (x_rp_line - offset_fraction) + intercept
        plt.plot(x_rp_line * 100, y_rp_line, color='green', linestyle='-', alpha=0.6, lw=1.5,
                 label=f'Rp {rp_offset_pct}% Offset')

        # Проекции Rp
        plt.scatter(rp_strain_pct, rp_stress, color='green', s=40, zorder=5)
        plt.plot([rp_strain_pct, rp_strain_pct], [0, rp_stress], color='green', linestyle='--', lw=1, alpha=0.6)
        plt.plot([0, rp_strain_pct], [rp_stress, rp_stress], color='green', linestyle='--', lw=1, alpha=0.6)

        # Подписи Rp
        plt.text(rp_strain_pct, -curr_ylim * 0.04, f'{round(rp_strain_pct, 2)}%', color='green', fontsize=9,
                 ha='center', fontweight='bold')
        plt.text(-curr_xlim * 0.02, rp_stress, f'{round(rp_stress, 1)}', color='green', fontsize=9, va='center',
                 ha='right', fontweight='bold')

    # 5. Проекции UTS (Черные пунктиры)
    plt.scatter(uts_strain_pct, uts_stress, color='black', s=60, zorder=6)
    plt.plot([uts_strain_pct, uts_strain_pct], [0, uts_stress], color='black', linestyle='--', lw=1.2)
    plt.plot([0, uts_strain_pct], [uts_stress, uts_stress], color='black', linestyle='--', lw=1.2)

    # Подписи UTS
    plt.text(uts_strain_pct, uts_stress + (curr_ylim * 0.02), f'UTS: {round(uts_stress, 1)} MPa',
             fontsize=10, fontweight='bold', ha='center')
    plt.text(uts_strain_pct, -curr_ylim * 0.06, f'{round(uts_strain_pct, 2)}%', color='black', fontsize=9, ha='center')

    # Настройка осей и сетки
    plt.title(f"Tensile Test Analysis (ID: {test_id})", fontsize=14)
    plt.xlabel("Strain (%)", fontsize=11)
    plt.ylabel("Stress (MPa)", fontsize=11)

    plt.xlim(0, curr_xlim)
    plt.ylim(0, curr_ylim)

    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.legend(loc='lower right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def plot_combined_tests(bundle, save_path, rp_offset_pct):
    """
    Сравнительный график всех тестов.
    Пояснение 'Rp' выводится только для первого теста, чтобы избежать путаницы.
    """
    plt.figure(figsize=(12, 8))

    # Цветовая карта
    colors = plt.cm.tab10(np.linspace(0, 1, len(bundle)))

    max_x = 0
    max_y = 0

    for i, data in enumerate(bundle):
        df = data['df']
        rp_stress = data['rp']
        E = data['E']
        intercept = data['intercept']
        color = colors[i]

        # 1. Отрисовка основной кривой
        plt.plot(df['strain_pct'], df['stress_mpa'],
                 label=f'Test {i}', color=color, lw=1.5, alpha=0.7)

        # 2. Отрисовка Rp и проекций
        if rp_stress and E:
            offset_fraction = rp_offset_pct / 100.0
            rp_x_pct = ((rp_stress - intercept) / E + offset_fraction) * 100

            # Тонкие пунктирные линии (проекции)
            plt.plot([rp_x_pct, rp_x_pct], [0, rp_stress], color=color, linestyle='--', lw=0.8, alpha=0.3)
            plt.plot([0, rp_x_pct], [rp_stress, rp_stress], color=color, linestyle='--', lw=0.8, alpha=0.3)

            # Жирная точка Rp
            plt.scatter(rp_x_pct, rp_stress, color=color, s=80, edgecolor='black', linewidth=0.7, zorder=5)

            # ПОДПИСЬ ТОЛЬКО ДЛЯ ПЕРВОГО ТЕСТА
            if i == 0:
                plt.annotate(f'Rp',
                             xy=(rp_x_pct, rp_stress),
                             xytext=(rp_offset_pct/2 - rp_offset_pct/10, rp_stress),
                             fontsize=12, fontweight='bold', color=color)

        # Обновление лимитов
        max_x = max(max_x, df['strain_pct'].max())
        max_y = max(max_y, df['stress_mpa'].max())

    # Настройка оформления
    plt.title(f"Combined Analysis: Yield Strength Comparison (Offset {rp_offset_pct}%)", fontsize=14)
    plt.xlabel("Strain (%)", fontsize=12)
    plt.ylabel("Stress (MPa)", fontsize=12)

    plt.xlim(0, max_x * 1.1)
    plt.ylim(0, max_y * 1.1)
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Experiments")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()