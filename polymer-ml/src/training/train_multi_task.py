import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_DIR,
    MODEL_FILENAME,
    CONFUSION_MATRIX_FIG,
    DO_CV,
    N_SPLITS_CV,
    RANDOM_STATE,
)
from src.data_loader import load_polymers_dataset
from src.models.multi_task import build_multi_task_pipeline, build_preprocessor
from src.training.evaluation import (
    evaluate_regression,
    evaluate_classification,
    save_confusion_matrix,
)

import joblib


def run_cv(X, y_class):
    """
    K-fold CV только для классификации PolymerClass
    с тем же препроцессором, но отдельным классификатором.
    """
    preprocessor = build_preprocessor()

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        cv_pipeline,
        X,
        y_class,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    return scores.mean(), scores.std()


def train_and_evaluate():
    df = load_polymers_dataset(min_samples_per_class=10, debug=False)

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
        cv_mean, cv_std = run_cv(X_train, y_train["PolymerClass"])
        print(f"CV (F1_weighted, PolymerClass): {cv_mean:.3f} ± {cv_std:.3f}")

    print("Fitting multi-task model (Tg + PolymerClass)...")
    pipeline.fit(X_train, y_train)

    # Инференс на тесте
    # Удобнее отдельно получить предсказания из "multi" шага
    preprocess = pipeline.named_steps["preprocess"]
    multi = pipeline.named_steps["multi"]
    X_test_trans = preprocess.transform(X_test)

    Tg_pred = multi.predict_regression(X_test_trans)
    class_pred = multi.predict_class(X_test_trans)

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
    save_confusion_matrix(clf_metrics["cm"], labels=multi.classifier.classes_, out_path=cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # Сохранение модели
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_evaluate()
