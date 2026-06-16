import pandas as pd
import numpy as np
from utils.logger import logger
from utils.config import PIPELINE_CONFIG


class PhysicsInformedAugmenter:
    def __init__(self, expansion_factor: int = 5, intrinsic_noise: float = 0.15, measurement_noise: float = 3.0):
        self.expansion_factor = expansion_factor
        self.y1_col = PIPELINE_CONFIG["y1"]["col_name"]
        self.y2_col = PIPELINE_CONFIG["y2"]["col_name"]

        # Настройки шума
        self.intrinsic_noise = intrinsic_noise  # 15% вариативность скорости деградации из-за микродефектов
        self.measurement_noise = measurement_noise  # +-3.0% ошибка лабораторных измерений

    def enrich_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Начало обогащения (Физический режим: {self.y1_col}, Expansion: {self.expansion_factor}x)...")
        logger.info(
            f"Настройки шума: Внутренний={self.intrinsic_noise * 100}%, Инструментальный=±{self.measurement_noise}%")

        if self.y1_col not in df.columns:
            raise ValueError(f"Для физического обогащения в датасете должна быть колонка '{self.y1_col}'")

        df = df.reset_index(drop=True)

        expanded_dfs = []
        for i in range(self.expansion_factor):
            temp_df = df.copy()

            # Генерируем случайные условия эксплуатации среды
            temp_df['T_env'] = np.random.uniform(250, 400, len(temp_df))
            temp_df['humidity'] = np.random.uniform(0, 100, len(temp_df))
            temp_df['exposure_hours'] = np.random.uniform(500, 15000, len(temp_df))

            if self.y1_col == 'Tc':
                temp_df['heat_load'] = np.random.uniform(10, 50, len(temp_df))

            expanded_dfs.append(temp_df)

        enriched_df = pd.concat(expanded_dfs, ignore_index=True)

        # Базовая константа скорости старения
        k_base = 0.00005

        # 1. ФИЗИКА: Расчет тепловых и влажностных ускорений
        if self.y1_col == 'Tg':
            delta_T = enriched_df['T_env'] - enriched_df[self.y1_col]
            thermal_acceleration = np.exp(delta_T / 45.0)
        elif self.y1_col == 'Tc':
            geom_constant = 100.0
            T_internal = enriched_df['T_env'] + (enriched_df['heat_load'] / (enriched_df[self.y1_col] * geom_constant))
            thermal_acceleration = np.exp((T_internal - 293) / 30.0)
            enriched_df = enriched_df.drop(columns=['heat_load'])
        else:
            thermal_acceleration = 1.0

        humidity_factor = 1.0 + (enriched_df['humidity'] / 100.0)

        # ==========================================
        # 2. ВНЕДРЕНИЕ ФИЗИЧЕСКОГО ШУМА
        # ==========================================

        # ШУМ 1 (Гетерогенность): Добавляем случайность в саму скорость старения для каждого образца
        # np.random.normal(1.0, 0.15) означает, что скорость старения будет случайно плавать на ±15%
        heterogeneity_factor = np.random.normal(1.0, self.intrinsic_noise, len(enriched_df))
        heterogeneity_factor = np.clip(heterogeneity_factor, 0.5, 1.5)  # Защита от экстремальных выбросов

        k_rate = k_base * thermal_acceleration * humidity_factor * heterogeneity_factor

        # Идеальная формула удержания свойств (Долговечность)
        ideal_retention = 100.0 - (k_rate * enriched_df['exposure_hours'])

        # ШУМ 2 (Инструментальный): Смазываем итоговые лабораторные показания
        sensor_error = np.random.normal(0, self.measurement_noise, len(enriched_df))
        final_retention = ideal_retention + sensor_error

        # Ограничиваем физически возможными рамками от 10% до 100%
        enriched_df[self.y2_col] = np.clip(final_retention, 10.0, 100.0)

        logger.info(f"Обогащение и зашумление завершены. Размер датасета: {len(enriched_df)} строк.")
        return enriched_df