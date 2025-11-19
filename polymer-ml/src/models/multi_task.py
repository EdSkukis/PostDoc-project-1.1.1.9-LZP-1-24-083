import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FP_N_BITS, FP_RADIUS
from src.featurizers.smiles_fp import SmilesFeaturizer
from src.featurizers.smiles_descriptors import DescriptorFeaturizer


class MultiOutputModel(BaseEstimator):
    """
    Обёртка над двумя моделями:
      - регрессия: Tg
      - классификация: PolymerClass
    """

    def __init__(self, regressor=None, classifier=None):
        self.regressor = regressor or RandomForestRegressor(
            n_estimators=300, random_state=42
        )
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=300, random_state=42
        )

    def fit(self, X, y):
        self.regressor.fit(X, y["Tg"])
        self.classifier.fit(X, y["PolymerClass"])
        return self

    def predict(self, X):
        Tg_pred = self.regressor.predict(X)
        class_pred = self.classifier.predict(X)
        return pd.DataFrame(
            {"Tg_pred": Tg_pred, "PolymerClass_pred": class_pred}
        )

    def predict_regression(self, X):
        return self.regressor.predict(X)

    def predict_class(self, X):
        return self.classifier.predict(X)


def build_preprocessor():
    """
    Строит ColumnTransformer, который принимает DataFrame с колонкой 'SMILES'.
    """
    # Важно: мы ожидаем DataFrame с колонкой 'SMILES'
    fp = SmilesFeaturizer(n_bits=FP_N_BITS, radius=FP_RADIUS)
    desc = Pipeline(
        steps=[
            ("desc", DescriptorFeaturizer()),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("fp", fp, ["SMILES"]),
            ("desc", desc, ["SMILES"]),
        ],
        remainder="drop",
    )
    return preprocessor


def build_multi_task_pipeline(regressor=None, classifier=None):
    """
    Полный sklearn Pipeline: препроцессор + MultiOutputModel.
    """
    preprocessor = build_preprocessor()
    multi = MultiOutputModel(regressor=regressor, classifier=classifier)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("multi", multi),
        ]
    )
    return pipe
