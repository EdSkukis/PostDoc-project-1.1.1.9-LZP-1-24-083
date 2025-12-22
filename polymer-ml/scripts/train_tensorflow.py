import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RANDOM_STATE,
    FP_N_BITS,
    EPOCHS,
    BATCH_SIZE,
    TF_MODEL_DIR,
    TF_MODEL_FILENAME,
)
from src.data_loader import load_polymers_dataset, process_and_save_smiles
from src.models.multi_task import CombinedSmilesFeaturizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.featurizers.rdkit_descriptors import NUM_DESCRIPTORS


def build_tf_model(n_features, n_classes):
    """
    Создает модель TensorFlow с двумя выходами.
    """
    inputs = tf.keras.Input(shape=(n_features,))
    
    # Общая часть
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    
    # Выход для регрессии (Tg)
    out_reg = tf.keras.layers.Dense(1, name="tg")(x)
    
    # Выход для классификации (PolymerClass)
    out_class = tf.keras.layers.Dense(n_classes, activation="softmax", name="polymer_class")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[out_reg, out_class])
    return model


def main():
    # Загрузка и подготовка данных
    df = load_polymers_dataset(debug=False)
    if df.empty:
        print("Dataset could not be loaded. Exiting.")
        return
        
    _, df_valid_final, _, _ = process_and_save_smiles(df)

    df = df_valid_final
    X = df[["SMILES"]]
    y = df[["Tg", "PolymerClass"]]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Предобработка SMILES
    preprocessor = CombinedSmilesFeaturizer(n_bits=FP_N_BITS)
    X_features = preprocessor.fit_transform(X)

    # Предобработка меток
    le = LabelEncoder()
    y_class = le.fit_transform(y["PolymerClass"])
    y_reg = y["Tg"].values

    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X_features, y_reg, y_class, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print("Data prepared for TensorFlow model.")

    # Создание и компиляция модели
    n_features = X_features.shape[1]
    n_classes = len(le.classes_)
    
    model = build_tf_model(n_features, n_classes)
    
    model.compile(
        optimizer="adam",
        loss={
            "tg": "mse",
            "polymer_class": "sparse_categorical_crossentropy",
        },
        loss_weights={"tg": 1.0, "polymer_class": 0.5},
        metrics={
            "tg": "mae",
            "polymer_class": "accuracy",
        },
    )
    
    model.summary()

    # Обучение модели
    print("\nTraining TensorFlow model...")
    history = model.fit(
        X_train,
        {"tg": y_reg_train, "polymer_class": y_class_train},
        validation_data=(X_test, {"tg": y_reg_test, "polymer_class": y_class_test}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # Оценка модели
    print("\nEvaluating TensorFlow model...")
    results = model.evaluate(X_test, {"tg": y_reg_test, "polymer_class": y_class_test}, verbose=0)
    
    print("Test loss:", results[0])
    print("Test Tg loss (mse):", results[1])
    print("Test PolymerClass loss (crossentropy):", results[2])
    print("Test Tg mae:", results[3])
    print("Test PolymerClass accuracy:", results[4])

    # Дополнительные метрики по Tg и классам (в тех же единицах, что и в df["Tg"])
    print("\nComputing metrics from raw predictions...")

    # Предсказания модели
    y_pred_reg, y_pred_class = model.predict(X_test, verbose=0)

    # Tg: MAE и RMSE
    y_true_tg = y_reg_test.astype(float).ravel()
    y_pred_tg = y_pred_reg.astype(float).ravel()

    mae_tg = mean_absolute_error(y_true_tg, y_pred_tg)
    mse_tg = mean_squared_error(y_true_tg, y_pred_tg)
    rmse_tg = np.sqrt(mse_tg)

    print("Tg MAE  (from predict):", mae_tg)
    print("Tg RMSE (from predict):", rmse_tg)

    # Классы: accuracy из предсказанных вероятностей
    y_true_class = y_class_test.ravel()
    y_pred_class_labels = np.argmax(y_pred_class, axis=1)

    acc_class = (y_pred_class_labels == y_true_class).mean() * 100.0
    print("PolymerClass accuracy (from predict):", acc_class)

    # Сохранение модели
    os.makedirs(TF_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(TF_MODEL_DIR, TF_MODEL_FILENAME)
    model.save(model_path)
    print(f"\nTensorFlow model saved to: {model_path}")



if __name__ == "__main__":
    main()
