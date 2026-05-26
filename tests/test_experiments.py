import unittest
import numpy as np
import pandas as pd
import os
from utils.robustness import inject_gaussian_noise
from scripts.run_experiments import (
    evaluate_automata_skab_kfold_robustness,
    evaluate_automata_batadal_robustness
)

class TestExperiments(unittest.TestCase):
    def test_inject_gaussian_noise_shapes(self):
        """
        Verify that noise injection maintains dataset shapes and formats.
        """
        np.random.seed(42)
        arr = np.random.randn(50, 4)
        
        # Test numpy array noise
        noisy_arr = inject_gaussian_noise(arr, scale=0.1)
        self.assertEqual(arr.shape, noisy_arr.shape)
        self.assertFalse(np.array_equal(arr, noisy_arr))
        
        # Test pandas DataFrame noise
        df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])
        noisy_df = inject_gaussian_noise(df, scale=0.1)
        self.assertEqual(df.shape, noisy_df.shape)
        self.assertTrue(isinstance(noisy_df, pd.DataFrame))
        
        # Test scale zero returning identical values
        clean_arr = inject_gaussian_noise(arr, scale=0.0)
        np.testing.assert_array_equal(arr, clean_arr)

    def test_evaluate_automata_skab_robustness_flow(self):
        """
        Test that evaluate_automata_skab_kfold_robustness executes cleanly under quick parameters.
        """
        # Run a quick check with seed 42 and noise scale 0.1
        try:
            results = evaluate_automata_skab_kfold_robustness(seed=42, noise_scale=0.1)
            self.assertIn("orig_f1", results)
            self.assertIn("noisy_f1", results)
            self.assertTrue(0.0 <= results["orig_f1"] <= 1.0)
            self.assertTrue(0.0 <= results["noisy_f1"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_automata_skab_kfold_robustness failed with error: {e}")

    def test_evaluate_automata_batadal_robustness_flow(self):
        """
        Test that evaluate_automata_batadal_robustness executes cleanly.
        """
        try:
            results = evaluate_automata_batadal_robustness(seed=42, noise_scale=0.1)
            self.assertIn("orig_f1", results)
            self.assertIn("noisy_f1", results)
            self.assertTrue(0.0 <= results["orig_f1"] <= 1.0)
            self.assertTrue(0.0 <= results["noisy_f1"] <= 1.0)
        except Exception as e:
            self.fail(f"evaluate_automata_batadal_robustness failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
