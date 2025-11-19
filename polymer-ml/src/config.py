# src/config.py

MODEL_DIR = "models_artifacts"
MODEL_FILENAME = "multi_output_polymer_model.joblib"
CONFUSION_MATRIX_FIG = "confusion_matrix_polymer_class.png"

# Параметры фингерпринта
FP_N_BITS = 2048
FP_RADIUS = 2

# Флаг: делать ли k-fold CV (можно отключить для ускорения)
DO_CV = True
N_SPLITS_CV = 5
RANDOM_STATE = 42
