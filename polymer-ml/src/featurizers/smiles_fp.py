import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, TransformerMixin


class SmilesFeaturizer(BaseEstimator, TransformerMixin):
    """
    SMILES -> Morgan fingerprint (битовый вектор).
    """

    def __init__(self, n_bits=2048, radius=2, remove_asterisk=True):
        self.n_bits = n_bits
        self.radius = radius
        self.remove_asterisk = remove_asterisk

    def fit(self, X, y=None):
        return self

    def _smiles_to_fp(self, smiles):
        if smiles is None:
            return np.zeros(self.n_bits, dtype=np.float32)

        s = str(smiles)
        if self.remove_asterisk:
            s = s.replace("*", "")

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def transform(self, X):
        # X — pandas Series или 1D массив со SMILES
        smiles_list = list(X)
        fps = [self._smiles_to_fp(s) for s in smiles_list]
        return np.vstack(fps)
