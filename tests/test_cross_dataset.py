import unittest
import numpy as np
import pandas as pd
import os
from scripts.run_experiments import (
    evaluate_cross_dataset_automata,
    evaluate_cross_dataset_dl
)

class TestCrossDataset(unittest.TestCase):
    def test_evaluate_cross_dataset_automata_skab_to_batadal(self):
        """
        Verify that evaluate_cross_dataset_automata runs cleanly from SKAB to BATADAL.
        """
        try:
            results = evaluate_cross_dataset_automata(train_dataset="SKAB", test_dataset="BATADAL", seed=42)
            self.assertIn("f1", results)
            self.assertIn("accuracy", results)
            self.assertTrue(0.0 <= results["f1"] <= 1.0)
            self.assertTrue(0.0 <= results["accuracy"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_cross_dataset_automata (SKAB -> BATADAL) failed with error: {e}")

    def test_evaluate_cross_dataset_automata_batadal_to_skab(self):
        """
        Verify that evaluate_cross_dataset_automata runs cleanly from BATADAL to SKAB.
        """
        try:
            results = evaluate_cross_dataset_automata(train_dataset="BATADAL", test_dataset="SKAB", seed=42)
            self.assertIn("f1", results)
            self.assertIn("accuracy", results)
            self.assertTrue(0.0 <= results["f1"] <= 1.0)
            self.assertTrue(0.0 <= results["accuracy"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_cross_dataset_automata (BATADAL -> SKAB) failed with error: {e}")

    def test_evaluate_cross_dataset_dl_flow(self):
        """
        Test that evaluate_cross_dataset_dl executes cleanly under baseline checkpoints.
        """
        try:
            results = evaluate_cross_dataset_dl(
                model_type="LSTM", 
                train_dataset="SKAB", 
                test_dataset="BATADAL", 
                seed=42
            )
            self.assertIn("f1", results)
            self.assertIn("accuracy", results)
            self.assertTrue(0.0 <= results["f1"] <= 1.0)
            self.assertTrue(0.0 <= results["accuracy"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_cross_dataset_dl failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
