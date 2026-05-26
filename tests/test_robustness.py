import unittest
import numpy as np
import pandas as pd
from models.automata import TimeSeriesAutomata
from utils.robustness import inject_gaussian_noise, calculate_unseen_metrics

class TestRobustnessAndUnseenAnalytics(unittest.TestCase):
    def setUp(self):
        # Seed generator for reproducibility
        np.random.seed(42)

    def test_inject_gaussian_noise_numpy(self):
        # 1. Test numpy array noise injection
        data = np.zeros((100, 2))
        
        # Scale = 0 should return the same array
        perturbed_zero = inject_gaussian_noise(data, scale=0.0)
        np.testing.assert_array_equal(data, perturbed_zero)
        
        # Scale > 0 should introduce variance
        perturbed_noisy = inject_gaussian_noise(data, scale=0.5)
        self.assertNotEqual(np.sum(perturbed_noisy), 0.0)
        self.assertAlmostEqual(np.mean(perturbed_noisy), 0.0, places=1)
        self.assertAlmostEqual(np.std(perturbed_noisy), 0.5, places=1)

    def test_inject_gaussian_noise_dataframe(self):
        # 2. Test pandas DataFrame noise injection
        df = pd.DataFrame(np.zeros((100, 3)), columns=["s1", "s2", "s3"])
        
        perturbed_df = inject_gaussian_noise(df, scale=0.2)
        self.assertEqual(perturbed_df.shape, (100, 3))
        self.assertNotEqual(perturbed_df["s1"].sum(), 0.0)
        self.assertAlmostEqual(perturbed_df["s2"].mean(), 0.0, places=1)
        self.assertAlmostEqual(perturbed_df["s3"].std(), 0.2, places=1)

    def test_calculate_unseen_metrics_toy(self):
        # 3. Test unseen metrics calculation on toy sequences
        automata = TimeSeriesAutomata(alphabet_size=3, word_size=4)
        
        # Train data: flat constant at 0.0 (vocabulary will only have 'bbbb')
        train_data = np.zeros((50, 1))
        train_labels = np.zeros(50)
        train_labels[20:30] = 1
        
        automata.fit(train_data, window_size=10)
        
        # Test data: flat constant at 0.0 (all seen)
        test_data = np.zeros((50, 1))
        test_labels = np.zeros(50)
        test_labels[20:30] = 1
        
        # Initially, all test states should be in vocabulary since test_data is identical
        unseen_res_clean = calculate_unseen_metrics(
            automata, train_data, test_data, train_labels, test_labels,
            window_size=10, threshold=0.01
        )
        self.assertEqual(unseen_res_clean["unseen_count"], 0)
        self.assertEqual(unseen_res_clean["detection_rate"], 1.0)
        self.assertEqual(unseen_res_clean["mapping_accuracy"], 1.0)
        
        # Now introduce a completely unseen segment in the test data (e.g. flat at 10.0)
        # This will generate 'cccc' (unseen state not in training vocabulary 'bbbb')
        test_data_noisy = test_data.copy()
        test_data_noisy[35:45] = 10.0
        
        unseen_res_noisy = calculate_unseen_metrics(
            automata, train_data, test_data_noisy, train_labels, test_labels,
            window_size=10, threshold=0.01
        )
        # We expect out-of-vocabulary unseen states to exist now
        self.assertTrue(unseen_res_noisy["unseen_count"] > 0)
        self.assertTrue(0.0 <= unseen_res_noisy["detection_rate"] <= 1.0)
        self.assertTrue(0.0 <= unseen_res_noisy["mapping_accuracy"] <= 1.0)

if __name__ == '__main__':
    unittest.main()
