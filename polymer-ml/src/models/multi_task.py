import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.config import FP_N_BITS, FP_RADIUS, N_ESTIMATORS, MODEL_RANDOM_STATE
from src.featurizers.rdkit_descriptors import calc_descriptor_vector


class CombinedSmilesFeaturizer(BaseEstimator, TransformerMixin):
    """
    Один трансформер:
      SMILES -> [Morgan FP | scaled RDKit descriptors]
    """

    def __init__(self, n_bits=2048, radius=2, remove_asterisk=True):
        self.n_bits = n_bits
        self.radius = radius
        self.remove_asterisk = remove_asterisk
        self.scaler_ = StandardScaler()

    # --- вспомогательные методы ---

    def _clean_smiles(self, s):
        if s is None:
            return ""
        s = str(s)
        if self.remove_asterisk:
            s = s.replace("*", "")
        return s

    def _smiles_to_mol(self, s):
        s = self._clean_smiles(s)
        mol = Chem.MolFromSmiles(s)
        return mol

    def _mol_to_fp(self, mol):
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


    def _mol_to_descriptors(self, mol):
        """
        Обёртка над calc_descriptor_vector, чтобы код был единообразным.
        """
        return calc_descriptor_vector(mol)


    def _extract_smiles_array(self, X):
        """
        Приводим X к массиву строк длины n_samples.
        X может быть DataFrame, Series или np.array.
        """
        if isinstance(X, pd.DataFrame):
            if "SMILES" in X.columns:
                arr = X["SMILES"].values
            else:
                # 1 колонка, берём её
                arr = X.iloc[:, 0].values
        elif isinstance(X, pd.Series):
            arr = X.values
        else:
            arr = np.asarray(X)
        return np.asarray(arr).ravel().astype(str)

    # --- API sklearn ---

    def fit(self, X, y=None):
        smiles = self._extract_smiles_array(X)
        mols = [self._smiles_to_mol(s) for s in smiles]

        # дескрипторы как np.array [n_samples, n_desc]
        descs = np.vstack([self._mol_to_descriptors(m) for m in mols])
        # обучаем scaler только на дескрипторах
        self.scaler_.fit(descs)
        return self

    def transform(self, X):
        smiles = self._extract_smiles_array(X)
        mols = [self._smiles_to_mol(s) for s in smiles]

        fps = np.vstack([self._mol_to_fp(m) for m in mols]).astype(np.float32)
        descs = np.vstack([self._mol_to_descriptors(m) for m in mols]).astype(np.float32)
        descs_scaled = self.scaler_.transform(descs)

        features = np.hstack([fps, descs_scaled])
        return features


class MultiOutputModel(BaseEstimator):
    """
    Обёртка над двумя моделями:
      - регрессия: Tg
      - классификация: PolymerClass
    """

    def __init__(self, regressor=None, classifier=None):
        if regressor is None:
            regressor = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                random_state=MODEL_RANDOM_STATE,
            )
        if classifier is None:
            classifier = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=MODEL_RANDOM_STATE,
            )

        self.regressor = regressor
        self.classifier = classifier

    def fit(self, X, y):
        # y — DataFrame с колонками "Tg" и "PolymerClass"
        self.regressor.fit(X, y["Tg"])
        self.classifier.fit(X, y["PolymerClass"])
        return self

    def predict(self, X):
        Tg_pred = self.regressor.predict(X)
        class_pred = self.classifier.predict(X)
        return pd.DataFrame(
            {"Tg_pred": Tg_pred, "PolymerClass_pred": class_pred}
        )


def build_preprocessor():
    """
    Вместо ColumnTransformer используем один комбинированный трансформер.
    """
    return CombinedSmilesFeaturizer(
        n_bits=FP_N_BITS,
        radius=FP_RADIUS,
        remove_asterisk=True,
    )


def build_multi_task_pipeline(regressor=None, classifier=None):
    preprocessor = build_preprocessor()
    multi = MultiOutputModel(regressor=regressor, classifier=classifier)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("multi", multi),
        ]
    )
    return pipe
