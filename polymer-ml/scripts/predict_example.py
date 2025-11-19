import joblib
import pandas as pd
import os

from src.config import MODEL_DIR, MODEL_FILENAME


def main():
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    pipeline = joblib.load(model_path)

    # Пример новых полимеров:
    data = pd.DataFrame(
        [
            {"SMILES": "C=C(C)C*"},       # условный пример
            {"SMILES": "C1=CC=CC=C1*"},   # условный пример
        ]
    )

    preds = pipeline.predict(data)
    print(preds)


if __name__ == "__main__":
    main()
