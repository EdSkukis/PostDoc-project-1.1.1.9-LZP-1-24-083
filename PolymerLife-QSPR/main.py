import pandas as pd
from data.download import DatasetDownloader
from data.augmenter import PhysicsInformedAugmenter
from smiles.descriptor_builder import SafeDescriptorBuilder
from models.two_stage_pipeline import TwoStagePolymerModel
from visualization.plot_results import ModelEvaluator
from visualization.shap_analysis import SHAPAnalyzer
from utils.logger import logger
from utils.config import PIPELINE_CONFIG

N_BEST_FEATURES = 8

def run_framework():
    logger.info("=== Launch PolymerLife-QSPR ===")

    downloader = DatasetDownloader()
    df_raw = downloader.download_and_verify()

    # Берем названия из конфига
    x_col = PIPELINE_CONFIG["x"]["col_name"]
    y1_col = PIPELINE_CONFIG["y1"]["col_name"]
    y1_name = PIPELINE_CONFIG["y1"]["display_name"]
    y2_name = PIPELINE_CONFIG["y2"]["display_name"]

    if x_col in df_raw.columns and y1_col in df_raw.columns:
        df_clean = df_raw.dropna(subset=[x_col, y1_col]).copy()

        # Обязательно приводим название колонки структуры к 'smiles', чтобы RDKit её увидел
        df_clean = df_clean.rename(columns={x_col: 'smiles'})
        logger.info(f"Отобрано строк с {y1_name} ({y1_col}): {df_clean.shape[0]}")
    else:
        logger.critical(f"В датасете не найдены обязательные колонки: {x_col} или {y1_col}!")
        return

    # Для быстрой проверки скрипта берем 300 случайных молекул.
    df_sample = df_clean

    logger.info("Data Augmentation...")
    augmenter = PhysicsInformedAugmenter(expansion_factor=5)
    df_enriched = augmenter.enrich_dataset(df_sample)

    logger.info("Vectorization of chemical structures...")
    builder = SafeDescriptorBuilder()
    df_features = builder.featurize_dataframe(df_enriched)
    chem_cols = builder.get_feature_names()

    logger.info("=== Launching Machine Learning ===")
    model = TwoStagePolymerModel(config=PIPELINE_CONFIG, r2_threshold=0.65)
    success, y1_results, y2_results = model.train_pipeline(df_features, chem_cols)

    if not success:
        logger.error("Training stopped. The Stage 1 model failed the Quality Gate.")
        return

    y_test_y1, y_pred_y1 = y1_results
    y_test_y2, y_pred_y2, X_train_f = y2_results

    logger.info("=== Generating graphs for an article ===")
    evaluator = ModelEvaluator()

    evaluator.plot_parity(y_test_y1, y_pred_y1, PIPELINE_CONFIG["y1"]["display_name"], PIPELINE_CONFIG["y1"]["units"])
    evaluator.plot_parity(y_test_y2, y_pred_y2, PIPELINE_CONFIG["y2"]["display_name"], PIPELINE_CONFIG["y2"]["units"])

    logger.info("=== Running SHAP analysis ===")
    # 1. Отсекаем физику/среду, оставляем только химические дескрипторы
    X_train_chem = X_train_f[chem_cols]

    # 2. Передаем в анализатор первую модель (stage1_model) и химические данные
    shap_analyzer_tc = SHAPAnalyzer(model.stage1_model, X_train_chem)

    # 3. Строим график
    y1_display = PIPELINE_CONFIG["y1"]["display_name"]
    shap_analyzer_tc.plot_summary(target_name=y1_display)

    best_features = shap_analyzer_tc.log_top_features(top_n=N_BEST_FEATURES, target_name=y1_display)

    best_features.to_csv(f"results/top_n/{y1_display}_top{N_BEST_FEATURES}_best_features.csv", index=False)

    logger.info("=== LAUNCHING THE EXPERIMENT: Training only on the TOP-xx features ===")

    top_x_chem_cols = best_features['Feature'].tolist()

    reduced_model = TwoStagePolymerModel(config=PIPELINE_CONFIG, r2_threshold=0.65)


    success_red, y1_red, y2_red = reduced_model.train_pipeline(
        df_features,
        chem_cols,
        top_x_chem_cols)

    if success_red:
        y1_test_red, y1_pred_red = y1_red
        evaluator.plot_parity(
            y1_test_red, y1_pred_red,
            f"{y1_display}_TOP{N_BEST_FEATURES}_Features",
            PIPELINE_CONFIG["y1"]["units"]
        )
        logger.info("=== The experiment is complete! Compare the R2 values in the terminal. ===")


if __name__ == "__main__":
    run_framework()