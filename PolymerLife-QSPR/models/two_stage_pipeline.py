import pandas as pd
import joblib
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils.logger import logger


class TwoStagePolymerModel:
    def __init__(self, config: dict, r2_threshold: float = 0.65):
        self.y1_col = config["y1"]["col_name"]
        self.y2_col = config["y2"]["col_name"]
        self.env_features = config["env_features"]
        self.r2_threshold = r2_threshold

        self.stage1_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
        self.durability_model = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7)

    def train_pipeline(self, df: pd.DataFrame, all_chem_cols: list, selected_chem_cols: list = None):
        os.makedirs("models/saved", exist_ok=True)

        if selected_chem_cols is not None and len(selected_chem_cols) > 0:
            active_chem_cols = selected_chem_cols
            logger.info(f"Training on a REDUCED set of chemical properties: {len(active_chem_cols)}")
        else:
            active_chem_cols = all_chem_cols
            logger.info(f"Training on the FULL set of chemical properties: {len(active_chem_cols)}")

        logger.info(f"--- STAGE 1: Training {self.y1_col} ---")
        X1 = df[active_chem_cols]
        y1 = df[self.y1_col]
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

        self.stage1_model.fit(X1_train, y1_train)
        y1_pred = self.stage1_model.predict(X1_test)
        r2_stage1 = r2_score(y1_test, y1_pred)
        mae_s1 = mean_absolute_error(y1_test, y1_pred)
        rmse_s1 = np.sqrt(mean_squared_error(y1_test, y1_pred))
        logger.info(f"Stage 1 R2: {r2_stage1:.3f}")
        logger.info(f"MAE: {mae_s1:.4f}")
        logger.info(f"RMSE: {rmse_s1:.4f}")

        if r2_stage1 < self.r2_threshold:
            logger.critical("Модель y1 не прошла порог качества. Остановка.")
            return False, None, None

        logger.info(f"--- STAGE 2: Обучение {self.y2_col} ---")
        X2 = pd.concat([X1, df[[self.y1_col] + self.env_features]], axis=1)
        y2 = df[self.y2_col]

        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
        self.durability_model.fit(X2_train, y2_train)
        y2_pred = self.durability_model.predict(X2_test)
        r2_s2 = r2_score(y2_test, y2_pred)
        mae_s2 = mean_absolute_error(y2_test, y2_pred)
        rmse_s2 = np.sqrt(mean_squared_error(y2_test, y2_pred))

        logger.info(f"--- Stage 2 Metrics ({self.y2_col}) ---")
        logger.info(f"R2 Score: {r2_s2:.4f}")
        logger.info(f"MAE: {mae_s2:.4f}")
        logger.info(f"RMSE: {rmse_s2:.4f}")

        joblib.dump(self.stage1_model, "models/saved/model_y1_stage1.pkl")
        joblib.dump(self.durability_model, "models/saved/model_y2_stage2.pkl")
        return True, (y1_test, y1_pred), (y2_test, y2_pred, X2_train)