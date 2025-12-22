import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from src.models.multi_task import CombinedSmilesFeaturizer
from rdkit import Chem


class TestCombinedSmilesFeaturizer(unittest.TestCase):

    def setUp(self):
        self.featurizer = CombinedSmilesFeaturizer(n_bits=128, radius=2)

    def test_clean_smiles(self):
        self.assertEqual(self.featurizer._clean_smiles("C*C"), "CC")
        self.assertEqual(self.featurizer._clean_smiles(" C C "), "C C")
        self.assertEqual(self.featurizer._clean_smiles(None), "")

    def test_smiles_to_mol(self):
        mol = self.featurizer._smiles_to_mol("CCO")
        self.assertIsNotNone(mol)
        self.assertIsInstance(mol, Chem.Mol)

        mol_none = self.featurizer._smiles_to_mol(None)
        self.assertIsNone(mol_none)

    def test_mol_to_fp(self):
        mol = Chem.MolFromSmiles("CCO")
        fp = self.featurizer._mol_to_fp(mol)
        self.assertEqual(len(fp), self.featurizer.n_bits)
        self.assertIsInstance(fp, (list, tuple, Chem.DataStructs.cDataStructs.BitVectWrapper, Chem.DataStructs.ExplicitBitVect, Chem.DataStructs.SparseBitVect)) # Check for various possible types

        fp_none = self.featurizer._mol_to_fp(None)
        self.assertTrue(all(val == 0 for val in fp_none))

    def test_extract_smiles_array(self):
        df = pd.DataFrame({"SMILES": ["CCO", "CCC"]})
        arr = self.featurizer._extract_smiles_array(df)
        self.assertListEqual(list(arr), ["CCO", "CCC"])

        series = pd.Series(["CCO", "CCC"])
        arr = self.featurizer._extract_smiles_array(series)
        self.assertListEqual(list(arr), ["CCO", "CCC"])

        np_array = ["CCO", "CCC"]
        arr = self.featurizer._extract_smiles_array(np_array)
        self.assertListEqual(list(arr), ["CCO", "CCC"])

    def test_fit_transform(self):
        X = pd.DataFrame({"SMILES": ["CCO", "CCC", "COC"]})
        features = self.featurizer.fit_transform(X)
        self.assertEqual(features.shape[0], 3)
        # 128 for FP + number of RDKit descriptors
        # We need to get the number of RDKit descriptors for a precise assertion
        # Let's assume it's 200 for now, as a placeholder
        self.assertTrue(features.shape[1] > 128) # At least fingerprints are there

    def test_pipeline_integration(self):
        X = pd.DataFrame({"SMILES": ["CCO", "CCC", "COC"]})
        y = pd.DataFrame({"Tg": [100, 110, 120], "PolymerClass": ["A", "B", "A"]})

        pipeline = Pipeline([
            ('featurizer', CombinedSmilesFeaturizer(n_bits=128, radius=2))
        ])
        features = pipeline.fit_transform(X, y)
        self.assertEqual(features.shape[0], 3)
        self.assertTrue(features.shape[1] > 128)


if __name__ == '__main__':
    unittest.main()
