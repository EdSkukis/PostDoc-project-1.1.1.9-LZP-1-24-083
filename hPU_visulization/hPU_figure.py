import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline  # Для сглаживания синтетической кривой
import io
import os

# Импорты для химии
from rdkit import Chem
from rdkit.Chem import Draw



class ArticlePlotter:
    """
    Класс для генерации графиков и схем научной статьи в едином стиле.
    """

    def __init__(self, font_family='Times New Roman', font_size=14, save_format='pdf'):
        self.font_family = font_family
        self.font_size = font_size
        self.save_format = f'visualisation/{save_format.lower()}'
        self.data_store = {}

        # Единая палитра цветов для ВСЕХ графиков в статье
        self.colors = {
            'T22': '#2C3E50',
            'T40': '#8E44AD',
            'T50': '#E74C3C',
            'RH75': '#7F8C8D',
            'RH98': '#2980B9',
            'H2O': '#E67E22',

            '0.66': '#AED6F1',
            '6.6': '#5DADE2',
            '66': '#2874A6',
            '660': '#154360',
        }
        self._setup_global_style()

    def _setup_global_style(self):
        """Устанавливает глобальные параметры matplotlib для соответствия стандартам публикаций."""
        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.linewidth': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.top': True,
            'ytick.right': True,
            'legend.frameon': False
        })

    def _save_plot(self, fig, filename_base):
        """Универсальный метод для сохранения графиков."""
        output_filename = f"{filename_base}.{self.save_format}"
        plt.tight_layout()
        plt.savefig(output_filename, format=self.save_format, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"График сохранен: {output_filename}")

    # =========================================================================
    # 1. ХИМИЧЕСКИЕ СТРУКТУРЫ
    # =========================================================================
    def draw_chemical_structures(self, components=None):
        """
        Генерирует изображения молекул на основе SMILES и сохраняет их
        через универсальный метод в заданном self.save_format.
        """
        if components is None:
            components = {
                "BDO": "OCCCCO",
                "MDI": "O=C=Nc1ccc(Cc2ccc(N=C=O)cc2)cc1",
                "PTMEG": "OCCCCOCCCCOCCCCO"
            }

        output_paths = {}

        for name, smiles in components.items():
            mol = Chem.MolFromSmiles(smiles)
            Chem.rdDepictor.Compute2DCoords(mol)

            # 1. Настройка опций отрисовки
            options = Draw.MolDrawOptions()
            options.useBWAtomPalette()  # Черно-белая палитра
            options.bondLineWidth = 2.0  # Связи делаем толще для четкости при печати
            options.minFontSize = 14  # Размер шрифта для атомов
            options.clearBackground = True  # Белый фон

            # 2. Рендерим молекулу в высоком разрешении (PIL Image)
            # Размеры увеличены для компенсации того, что это больше не SVG
            img = Draw.MolToImage(mol, size=(800, 400), options=options)

            # 3. Интеграция с matplotlib
            # Создаем фигуру нужной пропорции
            fig, ax = plt.subplots(figsize=(4, 2))

            # Вставляем отрендеренную молекулу
            ax.imshow(img)

            # Отключаем оси, рамки и сетку (нам нужна только сама молекула)
            ax.axis('off')

            # 4. Сохранение через централизованный метод
            filename_base = f"{name.lower()}_structure"
            self._save_plot(fig, filename_base)

            output_paths[name] = f"{filename_base}.{self.save_format}"

        print("Химические структуры успешно сохранены:", output_paths)
        return output_paths

    # =========================================================================
    # 2. МЕТОДОЛОГИЧЕСКИЕ СХЕМЫ (Синтетические графики)
    # =========================================================================
    def plot_tensile_schematic(self, filename_base='tensile_testing_schematic'):
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        # 1. Данные
        eps_points = np.array([0.000, 0.002, 0.004, 0.006, 0.008, 0.012, 0.020, 0.025, 0.030, 0.050, 0.070, 0.090, 0.120])
        sig_points = np.array([0.000, 6.00, 12.00, 18.00, 24.00, 36.00, 52.00, 58.00, 62.00, 70.00, 74.00, 78.00, 82.00])
        spline = make_interp_spline(eps_points, sig_points, k=3)
        eps = np.linspace(0, 0.10, 500)
        sig = spline(eps)

        # 2. Расчеты
        eps_start, eps_end = 0.001, 0.015
        mask = (eps >= eps_start) & (eps <= eps_end)
        eps_window = eps[mask]
        sig_window = sig[mask]

        # Линейная регрессия (y = kx) для расчета модуля
        # E_modulus = slope
        E_modulus, _, _, _ = np.linalg.lstsq(eps_window[:, np.newaxis], sig_window, rcond=None)
        E_modulus = E_modulus[0]

        offset = 0.002
        sig_offset = E_modulus * (eps - offset)

        # Поиск ключевых точек
        eps_y = eps[np.where((sig - sig_offset) < 0)[0][0]]
        sig_y = spline(eps_y)
        idx_max = np.argmax(sig)
        eps_max, sig_max = eps[idx_max], sig[idx_max]
        eps_break, sig_break = eps[-1], sig[-1]

        # 3. Отрисовка
        with plt.rc_context({'xtick.top': False, 'ytick.right': False}):
            fig, ax = plt.subplots(figsize=(9, 6))

            # --- Основные элементы на главном графике ---
            ax.plot(eps, sig, color=self.colors['T22'], linewidth=2.5, label='Stress-Strain', zorder=3)

            # Линия модуля Юнга
            ax.plot([0, eps_y + 0.0077], [0, (eps_y + 0.0077) * E_modulus], '-', color='#E74C3C',
                    linewidth=2.5, label=f'E = {E_modulus:.0f} MPa', zorder=4)

            # Линия смещения (красная)
            mask_offset = (eps >= offset) & (eps <= eps_y + 0.01)
            ax.plot(eps[mask_offset], sig_offset[mask_offset], '--', color='#E74C3C',
                    linewidth=1.5, label='0.2% Offset', zorder=2)

            # xy = координаты точки, xytext = куда сдвинуть текст, arrowprops = стиль линии
            annotations = [
                (eps_y, sig_y, r'$R_{p0.2}$', '#E74C3C', (0.005, -5)),
                (eps_max, sig_max, r'$\sigma_{max}$', '#2980B9', (0.0035, 1)),
                (eps_break, sig_break, r'$\sigma_{break}$', 'black', (-0.015, 2))
            ]

            # Точки
            for ex, ey, label, color, offset_xy in annotations:
                # Рисуем точку
                ax.scatter(ex, ey, color=color, s=80, zorder=5, edgecolor='white', linewidth=1)

                # Линии проекции (тонкие пунктиры до осей)
                ax.plot([0, ex], [ey, ey], ':', color='gray', lw=1, zorder=1)
                ax.plot([ex, ex], [0, ey], ':', color='gray', lw=1, zorder=1)

                # Аннотация с выноской
                ax.annotate(label, xy=(ex, ey), xytext=(ex + offset_xy[0], ey + offset_xy[1]),
                            fontsize=12, fontweight='bold',
                            arrowprops=dict(arrowstyle="->", color="black", lw=1))

            # 4. Врезка (ЗУМ на зону модуля)
            ax_inset = fig.add_axes([0.40, 0.265, 0.35, 0.3])
            ax_inset.plot(eps, sig, color=self.colors['T22'], lw=2)
            ax_inset.plot(eps, E_modulus * eps, '-', color='#E74C3C')
            ax_inset.axvspan(eps_start, eps_end, color='#D5DBDB', alpha=0.5, hatch='///')
            ax_inset.set_xlim(0, 0.02)
            ax_inset.set_ylim(0, 50)
            ax_inset.set_title("E-Modulus Window", fontsize=9)
            mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

            ax.text(offset + 0.0015, 2, "0.2% offset", color=self.colors.get('T50', '#E74C3C'),
                    fontsize=self.font_size - 3, rotation=72)

            # Финализация
            ax.set_xlim(0, 0.11)
            ax.set_ylim(0, 85)
            ax.set_xlabel(r'Strain, $\epsilon$ [-]')
            ax.set_ylabel(r'Stress, $\sigma$ [MPa]')
            ax.legend(
                loc='lower right',
                fontsize=10,
                frameon=True,  # Включаем рамку
                facecolor='white',  # Белый фон
                edgecolor='black',  # Цвет рамки (можно 'none', чтобы убрать)
                framealpha=1.0  # Полная непрозрачность фона
            )

            # Сохранение
            output_filename = f"{filename_base}.{self.save_format}"
            fig.savefig(output_filename, format=self.save_format, dpi=600, bbox_inches='tight')
            plt.close(fig)
            print(f"График сохранен: {output_filename}")


    # =========================================================================
    # 3. АБСОРБЦИЯ ВОДЫ И ДИФФУЗИЯ
    # =========================================================================
    @staticmethod
    def fickian_approx(t, W_inf, k):
        sum_terms = 0
        for n in range(15):
            sum_terms += (1 / (2 * n + 1) ** 2) * np.exp(-k * (2 * n + 1) ** 2 * t)
        return W_inf * (1 - (8 / np.pi ** 2) * sum_terms)

    def load_absorption_data(self, data_source, is_string=False):
        if is_string:
            df = pd.read_csv(io.StringIO(data_source))
        else:
            df = pd.read_csv(data_source)

        conditions = ['H2O', 'RH97', 'RH75']
        for cond in conditions:
            cols = [f'{cond}_1', f'{cond}_2', f'{cond}_3']
            if all(col in df.columns for col in cols):
                df[f'{cond}_mean'] = df[cols].mean(axis=1)
                df[f'{cond}_std'] = df[cols].std(axis=1)
            else:
                print(f"Внимание: Колонки для {cond} не найдены.")

        self.data_store['absorption'] = df
        print("Данные по абсорбции успешно загружены и обработаны.")

    def plot_absorption_kinetics(self, filename_base='water_absorption_kinetics'):
        if 'absorption' not in self.data_store:
            raise ValueError("Сначала загрузите данные методом load_absorption_data()")

        df = self.data_store['absorption']
        fig, ax = plt.subplots(figsize=(8, 5))

        styles = {
            'H2O': {'marker': 'o', 'color': self.colors['H2O'], 'label': '-RH H2O'},
            'RH97': {'marker': 'D', 'color': self.colors['RH98'], 'label': '-RH 97'},
            'RH75': {'marker': '^', 'color': self.colors['RH75'], 'label': '-RH 75'},
        }

        t_smooth = np.linspace(0, df['t'].max(), 500)
        sqrt_t_smooth = np.sqrt(t_smooth)

        for cond, style in styles.items():
            if f'{cond}_mean' not in df.columns:
                continue

            ax.errorbar(df['sqrt_t'], df[f'{cond}_mean'], yerr=df[f'{cond}_std'],
                        fmt=style['marker'], color=style['color'], label=style['label'],
                        capsize=3, elinewidth=1.2, markersize=7,
                        markerfacecolor=style['color'], markeredgecolor='grey', markeredgewidth=0.5)

            try:
                p0 = [df[f'{cond}_mean'].max(), 0.001]
                popt, _ = curve_fit(self.fickian_approx, df['t'], df[f'{cond}_mean'], p0=p0)
                w_fit = self.fickian_approx(t_smooth, *popt)
                ax.plot(sqrt_t_smooth, w_fit, '--', color=style['color'],
                        linewidth=2, zorder=1)
            except RuntimeError:
                print(f"Не удалось подобрать аппроксимацию для {cond}")

        ax.set_xlabel(r'$\sqrt{t}$, [$\sqrt{s}$]', fontsize=14)
        ax.set_ylabel('w, [%]', fontsize=14)
        ax.set_xlim(-5, 200)
        ax.set_ylim(0, 3.0)

        handles, labels = ax.get_legend_handles_labels()
        w_fick_line = mlines.Line2D([], [], color='black', linestyle='--',
                                    linewidth=1.5, label='Fick fit')
        handles.append(w_fick_line)
        labels.append('Fick fit')
        ax.legend(handles=handles, labels=labels, loc='center left',
                  bbox_to_anchor=(1, 0.5), fontsize=11, frameon=False)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        self._save_plot(fig, filename_base)

    # =========================================================================
    # 4. МЕХАНИКА (Растяжение)
    # =========================================================================
    def load_mechanical_data(self, data_source):
        df = pd.read_csv(data_source)
        df.rename(columns={df.columns[0]: 'Condition_Raw', df.columns[1]: 'Speed'}, inplace=True)

        cond_map = {
            'Temp Series (~22°C)': 'T22',
            'Temp Series (~40°C)': 'T40',
            'Temp Series (~50°C)': 'T50',
            'Aging RH 75%': 'RH75',
            'Aging RH 98%': 'RH98',
            'Aging H20': 'H2O',
            'Aging H2O': 'H2O'
        }
        df['Condition'] = df['Condition_Raw'].map(cond_map)
        df = df.dropna(subset=['Condition', 'Speed'])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Speed' in numeric_cols:
            numeric_cols.remove('Speed')

        agg_df = df.groupby(['Condition', 'Speed'])[numeric_cols].agg(['mean', 'std']).reset_index()
        agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]

        self.data_store['mechanics'] = agg_df
        print("Механические данные успешно загружены. Доступные параметры для Y:")
        print([col.replace('_mean', '') for col in agg_df.columns if '_mean' in col])

    def plot_mechanics(self, plot_by='grouped_by_condition', fixed_value='T22', y_column='_Sigma_max_MPa',
                       y_label=None, items_to_plot=None, filename_base='mech_plot'):
        df = self.data_store['mechanics']
        col_mean = f"{y_column}_mean"
        col_std = f"{y_column}_std"

        if col_mean not in df.columns:
            raise ValueError(f"Колонка {y_column} не найдена! Проверьте название.")

        format_lbl = lambda x: str(x).replace('H2O', r'H$_2$O')

        default_labels = {
            '_Sigma_max_MPa': r'$\sigma_{max}$, [MPa]',
            '_E_MPa': r'$E$, [MPa]',
            '_Strain_at_Sigmax': r'$\epsilon$ at $\sigma_{max}$, [-]',
            '_Rp0.2_MPa': r'$R_{p0.2}$, [MPa]'
        }
        if y_label is None:
            y_label = default_labels.get(y_column, y_column.replace('_', ' ').strip())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

        legend_handles = []

        if plot_by == 'grouped_by_condition':
            if items_to_plot is None:
                items_to_plot = ['T22', 'T40', 'T50', 'RH75', 'RH98', 'H2O']

            valid_conditions = [c for c in items_to_plot if c in df['Condition'].values]
            all_speeds = sorted(df['Speed'].dropna().unique())

            x_indices = np.arange(len(valid_conditions))
            bar_width = 0.8 / len(all_speeds)

            for i, speed in enumerate(all_speeds):
                speed_data = df[df['Speed'] == speed]
                means, stds = [], []
                for cond in valid_conditions:
                    row = speed_data[speed_data['Condition'] == cond]
                    if not row.empty:
                        means.append(row[col_mean].values[0])
                        stds.append(row[col_std].values[0])
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                offset = (i - len(all_speeds) / 2) * bar_width + bar_width / 2
                speed_key = str(int(speed)) if speed.is_integer() else str(speed)
                color = self.colors.get(speed_key, '#27AE60')

                ax.bar(x_indices + offset, means, yerr=stds, width=bar_width * 0.95,
                       color=color, edgecolor='black', linewidth=1.2, zorder=3, capsize=3,
                       error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})

                legend_handles.append(mpatches.Patch(color=color, label=f"{speed} mm/min"))

            ax.set_xticks(x_indices)
            ax.set_xticklabels([format_lbl(c) for c in valid_conditions])
            x_title = 'Condition'
            legend_title = 'Test Speed'

        elif plot_by == 'grouped_by_speed':
            if items_to_plot is None:
                valid_speeds = sorted(df['Speed'].dropna().unique())
            else:
                valid_speeds = sorted([s for s in items_to_plot if s in df['Speed'].values])

            all_conditions = ['T22', 'T40', 'T50', 'RH75', 'RH98', 'H2O']
            valid_conditions = [c for c in all_conditions if c in df['Condition'].values]

            x_indices = np.arange(len(valid_speeds))
            bar_width = 0.8 / len(valid_conditions)

            for i, cond in enumerate(valid_conditions):
                cond_data = df[df['Condition'] == cond]
                means, stds = [], []
                for speed in valid_speeds:
                    row = cond_data[cond_data['Speed'] == speed]
                    if not row.empty:
                        means.append(row[col_mean].values[0])
                        stds.append(row[col_std].values[0])
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                offset = (i - len(valid_conditions) / 2) * bar_width + bar_width / 2
                color = self.colors.get(cond, 'gray')

                ax.bar(x_indices + offset, means, yerr=stds, width=bar_width * 0.95,
                       color=color, edgecolor='black', linewidth=1.2, zorder=3, capsize=3,
                       error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})

                legend_handles.append(mpatches.Patch(color=color, label=format_lbl(cond)))

            x_tick_labels = [str(int(s)) if s.is_integer() else str(s) for s in valid_speeds]
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_tick_labels)
            x_title = r'Speed, [mm/min]'
            legend_title = 'Condition'

        elif plot_by == 'speed':
            plot_data = df[df['Condition'] == fixed_value].sort_values('Speed')
            x_labels = plot_data['Speed'].astype(str)
            bar_colors = [self.colors.get(str(int(s)) if s.is_integer() else str(s), 'gray') for s in
                          plot_data['Speed']]
            ax.bar(x_labels, plot_data[col_mean], yerr=plot_data[col_std],
                   width=0.6, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3, capsize=4,
                   error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
            x_title = r'Speed, [mm/min]'
            legend_handles = [
                mpatches.Patch(color=self.colors.get(fixed_value, 'black'), label=format_lbl(fixed_value))]
            legend_title = None

        elif plot_by == 'condition':
            plot_data = df[df['Speed'] == fixed_value]
            if items_to_plot:
                plot_data = plot_data[plot_data['Condition'].isin(items_to_plot)]
                plot_data['Condition'] = pd.Categorical(plot_data['Condition'], categories=items_to_plot, ordered=True)
                plot_data = plot_data.sort_values('Condition')

            x_labels = [format_lbl(c) for c in plot_data['Condition']]
            bar_colors = [self.colors.get(c, 'gray') for c in plot_data['Condition']]
            ax.bar(x_labels, plot_data[col_mean], yerr=plot_data[col_std],
                   width=0.6, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3, capsize=4,
                   error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
            x_title = 'Condition'
            legend_handles = [mpatches.Patch(color=self.colors.get(c, 'gray'), label=format_lbl(c)) for c in
                              plot_data['Condition']]
            legend_title = f'Speed: {fixed_value}'

        else:
            raise ValueError("plot_by должен быть: 'grouped_by_condition', 'grouped_by_speed', 'speed', 'condition'")

        if plot_by in ['grouped_by_condition', 'grouped_by_speed']:
            ax.legend(handles=legend_handles, title=legend_title, loc='upper left',
                      bbox_to_anchor=(1.02, 1), frameon=False, fontsize=11)
        else:
            ax.legend(handles=legend_handles, title=legend_title, loc='upper right', frameon=False, fontsize=11)

        ax.set_xlabel(x_title, fontsize=16, labelpad=10)
        ax.set_ylabel(y_label, fontsize=16, labelpad=10)

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5)

        self._save_plot(fig, filename_base)

    # =========================================================================
    # 4. ДИНАМИЧЕСКИЕ КРИВЫЕ РАСТЯЖЕНИЯ (Многолистовой Excel)
    # =========================================================================
    def load_excel_stress_strain(self, file_path):
        """
        Загружает данные испытаний на растяжение из Excel-файла со всеми закладками.
        Распознает 4-строчную структуру иерархических заголовков.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл Excel не найден: {file_path}")

        xls = pd.ExcelFile(file_path)
        self.data_store['raw_curves'] = {}

        for sheet_name in xls.sheet_names:
            # Считываем 4 строки иерархического заголовка
            df = pd.read_excel(xls, sheet_name=sheet_name, header=[0, 1, 2, 3])

            # Итерируемся по колонкам группами по 3 (Time, Strain, Force)
            for col in df.columns:
                speed_raw = str(col[0]).strip()
                sample_rep = str(col[1]).strip()
                dtype_raw = str(col[2]).strip().lower()

                # Парсинг численного значения скорости (например, "6.6 mm/min" -> "6.6")
                speed_key = speed_raw.split()[0] if ' ' in speed_raw else speed_raw
                # Выделение чистого условия (например, "T22-1" -> "T22", "RH75-3" -> "RH75")
                condition_key = sample_rep.split('-')[0] if '-' in sample_rep else sample_rep

                # Уникальный ключ для идентификации кривой конкретного образца
                curve_id = (speed_key, condition_key, sample_rep)

                if curve_id not in self.data_store['raw_curves']:
                    self.data_store['raw_curves'][curve_id] = {}

                # Запись векторов данных, удаляя хвостовые пустые ячейки (NaN)
                series_data = df[col].dropna().reset_index(drop=True)

                if 'strain' in dtype_raw:
                    self.data_store['raw_curves'][curve_id]['strain'] = series_data
                elif 'force' in dtype_raw or 'stress' in dtype_raw:
                    self.data_store['raw_curves'][curve_id]['stress'] = series_data

        print(
            f"Успешно импортировано {len(self.data_store['raw_curves'])} кривых деформации изо всех закладок Excel.")

    def plot_stress_strain_curves(self, group_by='condition', target_value='T22', filename_base=None):
        """
        Строит высококачественные графики деформации для научных публикаций.

        :param group_by: 'condition' -> График для одного материала (например, T22), кривые раскрашены по скоростям.
                         'speed'     -> График для одной скорости (например, 6.6), кривые раскрашены по материалам.
        :param target_value: Выбранное значение для фильтрации группы (например, 'T22' или '6.6').
        """
        if 'raw_curves' not in self.data_store or not self.data_store['raw_curves']:
            raise ValueError("Сначала загрузите экспериментальные кривые методом load_excel_stress_strain()")

        fig, ax = plt.subplots(figsize=(7.5, 6))
        legend_map = {}  # Избегаем дублирования одинаковых серий в легенде

        curves_found = False

        for (speed, condition, sample_rep), data in self.data_store['raw_curves'].items():
            if 'strain' not in data or 'stress' not in data:
                continue

            # Фильтрация данных по критерию группировки
            if group_by == 'condition':
                if condition != target_value:
                    continue
                # Группировка по условию: цвет кодирует скорость испытания
                color = self.colors.get(speed, '#34495E')
                label_text = f"{speed} mm/min"
            elif group_by == 'speed':
                if speed != str(target_value):
                    continue
                # Группировка по скорости: цвет кодирует состояние/старение полимера
                color = self.colors.get(condition, '#34495E')
                label_text = condition.replace('H2O', r'H$_2$O')
            else:
                raise ValueError("Параметр group_by должен принимать значение 'condition' или 'speed'")

            curves_found = True
            x = data['strain'].values
            y = data['stress'].values

            # Отрисовка кривой (реплики отображаются тонкими линиями в одном стиле для чистоты восприятия)
            line, = ax.plot(x, y, color=color, linewidth=1.6, alpha=0.8)

            if label_text not in legend_map:
                legend_map[label_text] = line

        if not curves_found:
            print(f"Кривые для {group_by} = '{target_value}' не найдены в базе данных.")
            plt.close(fig)
            return

        # Академическое оформление координатных осей
        ax.set_xlabel(r'Strain, $\epsilon$ [%]', fontsize=self.font_size, fontname=self.font_family, labelpad=8)
        ax.set_ylabel(r'Stress, $\sigma$ [MPa]', fontsize=self.font_size, fontname=self.font_family, labelpad=8)

        # Текстовая аннотация группы внутри графика (взамен загромождающих title)
        info_title = f"Condition: {target_value.replace('H2O', r'H$_2$O')}" if group_by == 'condition' else f"Test Speed: {target_value} mm/min"
        ax.text(0.04, 0.95, info_title, transform=ax.transAxes, fontsize=self.font_size,
                fontname=self.font_family, fontweight='bold', va='top', ha='left')

        # Формирование чистой журнальной легенды без рамок
        ax.legend(legend_map.values(), legend_map.keys(), loc='lower right',
                  fontsize=self.font_size - 2, frameon=False)

        # Обеспечение верхней и правой ограничительных линий осей (стандарт Wiley)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        if filename_base is None:
            filename_base = f"fig_stress_strain_{group_by}_{target_value.replace('.', '_')}"

        self._save_plot(fig, filename_base)


# === Пример использования ===
if __name__ == "__main__":
    raw_csv_sorption = r'data/sorption.csv'
    raw_excel_sigma_strain = r'data/sigma_strain.csv'

    # Инициализация с заданным форматом
    plotter = ArticlePlotter(save_format='jpg', font_family='Times New Roman', font_size=14)

    # 1. Генерация химических формул
    plotter.draw_chemical_structures()

    # 2. Генерация методологической схемы (сохраняется в выбранном self.save_format, например PDF)
    plotter.plot_tensile_schematic('fig_tensile_testing_schematic')

    plotter.load_absorption_data(raw_csv_sorption, is_string=True)
    plotter.plot_absorption_kinetics('fig_water_absorption')

    if os.path.exists(raw_excel_sigma_strain):
        plotter.load_excel_stress_strain(raw_excel_sigma_strain)

        # ГРУППИРОВКА ВАРИАНТ №1: По условиям старения/эксперимента (Condition)
        # На одном графике будут собраны все кривые деформации для материала T22 при разных скоростях.
        plotter.plot_stress_strain_curves(group_by='condition', target_value='T22')
        plotter.plot_stress_strain_curves(group_by='condition', target_value='H2O')

        # ГРУППИРОВКА ВАРИАНТ №2: По скорости деформирования (Speed)
        # На одном графике будут представлены кривые для ВСЕХ типов старения, испытанных строго на скорости 6.6 мм/мин.
        plotter.plot_stress_strain_curves(group_by='speed', target_value='6.6')
        plotter.plot_stress_strain_curves(group_by='speed', target_value='660')
    else:
        print(f"Для демонстрации работы поместите ваш файл Excel под именем '{raw_excel_sigma_strain}' рядом со скриптом.")

    # 3. Построение основных механических графиков из файла
    # plotter.load_mechanical_data('calculait_experimental_data.csv')

    # plotter.plot_mechanics(
    #     plot_by='grouped_by_condition',
    #     y_column='_Sigma_max_MPa',
    #     filename_base='fig_grouped_by_condition_sigma'
    # )

    # plotter.plot_mechanics(
    #     plot_by='grouped_by_speed',
    #     y_column='_Sigma_max_MPa',
    #     filename_base='fig_grouped_by_speed_sigma'
    # )

