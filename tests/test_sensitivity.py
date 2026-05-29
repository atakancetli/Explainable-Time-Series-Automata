import unittest
import numpy as np
import pandas as pd
import os
from scripts.run_experiments import (
    evaluate_automata_window_sensitivity,
    evaluate_automata_alphabet_sensitivity,
    evaluate_automata_sensitivity
)

class TestParameterSensitivity(unittest.TestCase):
    def test_evaluate_automata_window_sensitivity(self):
        """
        Verify that evaluate_automata_window_sensitivity runs cleanly and returns valid metrics.
        """
        try:
            # We run on a small window size (e.g., 3) on BATADAL with a single seed
            results = evaluate_automata_window_sensitivity(window_size=3, seed=42, dataset_name="BATADAL")
            self.assertIn("f1", results)
            self.assertIn("accuracy", results)
            self.assertTrue(0.0 <= results["f1"] <= 1.0)
            self.assertTrue(0.0 <= results["accuracy"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_automata_window_sensitivity failed with error: {e}")

    def test_evaluate_automata_alphabet_sensitivity(self):
        """
        Verify that evaluate_automata_alphabet_sensitivity runs cleanly and returns valid metrics.
        """
        try:
            # We run on a small alphabet size (e.g., 3) on BATADAL with a single seed
            results = evaluate_automata_alphabet_sensitivity(alphabet_size=3, seed=42, dataset_name="BATADAL")
            self.assertIn("f1", results)
            self.assertIn("accuracy", results)
            self.assertTrue(0.0 <= results["f1"] <= 1.0)
            self.assertTrue(0.0 <= results["accuracy"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_automata_alphabet_sensitivity failed with error: {e}")

    def test_evaluate_automata_sensitivity_skab(self):
        """
        Verify that evaluate_automata_sensitivity runs cleanly on SKAB dataset.
        """
        try:
            results = evaluate_automata_sensitivity(
                param_name="window_size", 
                param_value=4, 
                seed=42, 
                dataset_name="SKAB"
            )
            self.assertIn("f1", results)
            self.assertTrue(0.0 <= results["f1"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_automata_sensitivity on SKAB failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
