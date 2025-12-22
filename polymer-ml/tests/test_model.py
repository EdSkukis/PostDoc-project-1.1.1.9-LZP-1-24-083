import unittest
import pandas as pd
import numpy as np
from src.models.multi_task import MultiOutputModel

class TestMultiOutputModel(unittest.TestCase):

    def setUp(self):
        self.model = MultiOutputModel()
        self.X = np.random.rand(10, 5)
        self.y = pd.DataFrame({
            "Tg": np.random.rand(10) * 100,
            "PolymerClass": np.random.choice(["A", "B"], 10)
        })

    def test_fit(self):
        """Test if the model can be fitted without errors."""
        try:
            self.model.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"fit() raised an exception: {e}")

    def test_predict(self):
        """Test if the model can make predictions and the output is correct."""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), 10)
        self.assertIn("Tg_pred", predictions.columns)
        self.assertIn("PolymerClass_pred", predictions.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(predictions["Tg_pred"]))
        self.assertTrue(pd.api.types.is_object_dtype(predictions["PolymerClass_pred"]))

if __name__ == '__main__':
    unittest.main()
