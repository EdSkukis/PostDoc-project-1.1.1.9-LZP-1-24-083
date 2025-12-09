import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from sklearn.base import BaseEstimator, TransformerMixin


class DescriptorFeaturizer(BaseEstimator, TransformerMixin):
    """
    SMILES -> RDKit дескрипторы:
      - MolWt
      - LogP
      - NumHDonors (HBD)
      - NumHAcceptors (HBA)
    """

    def __init__(self, remove_asterisk=True):
        self.remove_asterisk = remove_asterisk

    def fit(self, X, y=None):
        return self

    def _calc_descriptors(self, smiles):
        if smiles is None:
            return [0.0, 0.0, 0.0, 0.0]

        s = str(smiles)
        if self.remove_asterisk:
            s = s.replace("*", "")

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return [0.0, 0.0, 0.0, 0.0]

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        return [mw, logp, hbd, hba]

    def transform(self, X):
        arr = np.asarray(X).ravel()
        descs = [self._calc_descriptors(s) for s in arr]
        return np.array(descs, dtype=np.float32)
