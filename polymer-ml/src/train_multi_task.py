import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.models.analysis_feature import explain_top_fingerprint_bits
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

from src.config import (
    MODEL_DIR,
    MODEL_FILENAME,
    CONFUSION_MATRIX_FIG,
    DO_CV,
    N_SPLITS_CV,
    RANDOM_STATE,
    FP_N_BITS,
)
from src.data_loader import load_polymers_dataset, process_and_save_smiles
from src.models.multi_task import build_multi_task_pipeline
from src.evaluation.metrics import (
    evaluate_regression,
    evaluate_classification,
    save_confusion_matrix,
)
from src.visualization.smiles_quality import plot_smiles_quality_stats

import joblib


def run_cv(pipeline, X, y, cv):
    """
    K-fold CV for the entire multi-task pipeline.
    Evaluates the F1 score for the classification task.
    """
    f1_scores = []
    for train_idx, val_idx in cv.split(X, y['PolymerClass']):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        f1 = f1_score(y_val['PolymerClass'], y_pred['PolymerClass_pred'], average='weighted')
        f1_scores.append(f1)

    return np.mean(f1_scores), np.std(f1_scores)


def show_feature_importances(pipeline, top_n=10):
    """
    Показывает наиболее важные фичи для регрессора Tg.
    """
    print("\n--- Top-10 Feature Importances (Tg regressor) ---")
    regressor = pipeline.named_steps["multi"].regressor
    importances = regressor.feature_importances_

    # Отделяем фингерпринты от дескрипторов
    fp_importances = importances[:FP_N_BITS]
    
    indices = np.argsort(fp_importances)[::-1]

    print(f"Top {top_n} most important fingerprint bits for Tg prediction:")
    for i in range(top_n):
        bit_idx = indices[i]
        importance = fp_importances[bit_idx]
        print(f"  - Bit {bit_idx}: {importance:.4f}")


def train_and_evaluate(min_samples_per_class: int = 10):
    df = load_polymers_dataset(debug=False)
    if df.empty:
        print("Dataset could not be loaded. Exiting.")
        return
        
    _, df_valid_final, _, _ = process_and_save_smiles(df)

    # Apply min_samples_per_class filtering after SMILES processing
    if min_samples_per_class > 1:
        counts = df_valid_final["PolymerClass"].value_counts()
        valid_classes = counts[counts >= min_samples_per_class].index
        df_valid_final = df_valid_final[df_valid_final["PolymerClass"].isin(valid_classes)]
        print(f"Shape after filtering by min_samples_per_class={min_samples_per_class}:", df_valid_final.shape)

    df = df_valid_final
    # Логика: X = SMILES, y = [Tg, PolymerClass]
    X = df[["SMILES"]]  # DataFrame с одной колонкой
    y = df[["Tg", "PolymerClass"]]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y["PolymerClass"],
    )

    pipeline = build_multi_task_pipeline()

    if DO_CV:
        cv = StratifiedKFold(
            n_splits=N_SPLITS_CV,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        cv_mean, cv_std = run_cv(pipeline, X_train, y_train, cv)
        print(f"CV (F1_weighted, PolymerClass): {cv_mean:.3f} ± {cv_std:.3f}")

    print("Fitting multi-task model (Tg + PolymerClass)...")
    pipeline.fit(X_train, y_train)

    # Инференс на тесте
    y_pred = pipeline.predict(X_test)
    Tg_pred = y_pred["Tg_pred"]
    class_pred = y_pred["PolymerClass_pred"]

    # Оценка Tg
    reg_metrics = evaluate_regression(y_test["Tg"], Tg_pred)
    print("\nRegression (Tg):")
    for k, v in reg_metrics.items():
        print(f"{k}: {v:.3f}")

    # Оценка классификации
    clf_metrics = evaluate_classification(y_test["PolymerClass"], class_pred)
    print("\nClassification (PolymerClass):")
    print(clf_metrics["report"])

    # Матрица ошибок
    os.makedirs(MODEL_DIR, exist_ok=True)
    cm_path = os.path.join(MODEL_DIR, CONFUSION_MATRIX_FIG)
    multi = pipeline.named_steps["multi"]
    save_confusion_matrix(clf_metrics["cm"], labels=multi.classifier.classes_, out_path=cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # Сохранение модели
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # Показываем важность фич
    show_feature_importances(pipeline)

    df_bits = explain_top_fingerprint_bits(
        pipeline=pipeline,
        df=df_valid_final,
        smiles_col="SMILES_clean",
        top_n=10,
        out_dir="fp_bit_explanations",
        max_examples_per_bit=5,
    )