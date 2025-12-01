# src/config.py

MODEL_DIR = "models_artifacts"
MODEL_FILENAME = "multi_output_polymer_model.joblib"
CONFUSION_MATRIX_FIG = "confusion_matrix_polymer_class.png"
SMILES_CHECK_DIR = "smiles_check"

# Параметры фингерпринта
FP_N_BITS = 2048
FP_RADIUS = 2

# Флаг: делать ли k-fold CV (можно отключить для ускорения)
DO_CV = False
N_SPLITS_CV = 5
RANDOM_STATE = 42

# Параметры моделей
N_ESTIMATORS = 300
MODEL_RANDOM_STATE = 42

