import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_loader import classify_smiles_with_reason, fix_smiles, process_and_save_smiles, load_polymers_dataset

class TestSmilesClassifier(unittest.TestCase):

    def test_valid_smiles(self):
        """Test that valid SMILES are classified correctly."""
        self.assertEqual(classify_smiles_with_reason("CCO"), ("valid", "parsed_and_sanitized"))

    def test_invalid_smiles(self):
        """Test that invalid SMILES are classified correctly."""
        self.assertEqual(classify_smiles_with_reason("C(C)O"), ("fixable", "sanitization_error"))
        self.assertEqual(classify_smiles_with_reason(""), ("invalid", "empty_or_none"))
        self.assertEqual(classify_smiles_with_reason(None), ("invalid", "empty_or_none"))
        self.assertEqual(classify_smiles_with_reason("C%12=C=C%12"), ("invalid", "unparsable"))


    def test_fixable_smiles(self):
        """Test that fixable SMILES are classified correctly."""
        self.assertEqual(classify_smiles_with_reason("C()O"), ("fixable", "empty_branch"))
        self.assertEqual(classify_smiles_with_reason("CC "), ("fixable", "contains_spaces"))
        self.assertEqual(classify_smiles_with_reason("C*C"), ("fixable", "contains_asterisk"))
        self.assertEqual(classify_smiles_with_reason("CC.O"), ("fixable", "multi_fragment_maybe_salt"))
        self.assertEqual(classify_smiles_with_reason("CN(=O)=O"), ("fixable", "raw_nitro_group"))

class TestSmilesFixer(unittest.TestCase):

    def test_fix_spaces(self):
        self.assertEqual(fix_smiles("CC O"), ("CCO", "applied:remove_spaces"))

    def test_fix_nitro_group(self):
        self.assertEqual(fix_smiles("CN(=O)=O"), ("C[N+](=O)[O-]", "applied:normalize_nitro"))

    def test_fix_empty_branches(self):
        self.assertEqual(fix_smiles("C()C()O"), ("CCO", "applied:remove_empty_branches"))

    def test_fix_asterisks(self):
        self.assertEqual(fix_smiles("C*CO*"), ("CCO", "applied:remove_asterisks"))

    def test_fix_fragments(self):
        self.assertEqual(fix_smiles("CCO.Cl"), ("CCO", "applied:keep_largest_fragment"))

    def test_no_fix_needed(self):
        self.assertEqual(fix_smiles("CCO"), ("CCO", "no_fix_needed_but_was_called"))

    def test_cannot_fix(self):
        self.assertIsNone(fix_smiles("C%12=C=C%12")[0])

class TestDataProcessing(unittest.TestCase):

    @patch('src.data_loader.os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_process_and_save_smiles(self, mock_to_csv, mock_makedirs):
        """Test the overall SMILES processing and saving function."""
        data = {
            "SMILES": ["CCO", "CC O", "C%12=C=C%12"],
            "Tg": [100, 110, 120],
            "PolymerClass": ["A", "B", "A"]
        }
        df = pd.DataFrame(data)
        df_full, df_valid, df_invalid, df_recovered = process_and_save_smiles(df, output_dir="/fake/dir")

        self.assertEqual(len(df_full), 3)
        self.assertEqual(len(df_valid), 2)
        self.assertEqual(len(df_invalid), 1)
        self.assertEqual(len(df_recovered), 1)
        self.assertEqual(df_valid.iloc[1]["SMILES_clean"], "CCO")
        self.assertEqual(mock_makedirs.call_count, 1)
        self.assertEqual(mock_to_csv.call_count, 6)

    @patch('src.data_loader.kagglehub.dataset_download')
    @patch('src.data_loader.os.listdir')
    @patch('pandas.read_csv')
    def test_load_polymers_dataset(self, mock_read_csv, mock_listdir, mock_kaggle_download):
        """Test the dataset loading function with mocks."""
        mock_kaggle_download.return_value = "/fake/path"
        mock_listdir.return_value = ["fake_data.csv"]
        mock_df = pd.DataFrame({
            "smiles": ["CCO", "COC"],
            "tg": ["100", "110"],
            "polymer_class": ["A", "B"]
        })
        mock_read_csv.return_value = mock_df

        df = load_polymers_dataset(debug=False)

        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["SMILES", "Tg", "PolymerClass"])
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Tg']))

if __name__ == '__main__':
    unittest.main()
