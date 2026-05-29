import unittest
import numpy as np
from utils.statistics import calculate_mcnemar_test, calculate_wilcoxon_test

class TestStatisticalSignificance(unittest.TestCase):
    def test_mcnemar_exact_binomial(self):
        """
        Verify McNemar test exact binomial calculations for small discordant counts (b + c < 25).
        """
        # y_true: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        y_true = np.ones(10)
        # preds_A: all correct
        preds_A = np.ones(10)
        # preds_B: 2 incorrect (b = 2, c = 0, discordant = 2)
        preds_B = np.ones(10)
        preds_B[0:2] = 0
        
        res = calculate_mcnemar_test(preds_A, preds_B, y_true)
        
        self.assertIn("contingency_table", res)
        self.assertIn("p_value", res)
        self.assertIn("statistic", res)
        
        # Table A Right (b) vs B Right (c)
        # correct_A = [T, T, T, T, T, T, T, T, T, T]
        # correct_B = [F, F, T, T, T, T, T, T, T, T]
        # a (both T) = 8
        # b (A right, B wrong) = 2
        # c (A wrong, B right) = 0
        # d (both F) = 0
        self.assertEqual(res["contingency_table"], [[8, 2], [0, 0]])
        self.assertEqual(res["statistic"], 0.0) # float(min(b, c)) = min(2, 0) = 0
        # p_value for n=2, successes=0 under p=0.5:
        # binomial exact p = 2 * (1/4) = 0.5
        self.assertAlmostEqual(res["p_value"], 0.5)
        self.assertFalse(res["significant"])

    def test_mcnemar_corrected_chi2(self):
        """
        Verify McNemar test chi-squared calculations with Yates's correction for larger discordant counts.
        """
        y_true = np.ones(100)
        preds_A = np.ones(100)
        preds_B = np.ones(100)
        # 30 discordant pairs (b = 25, c = 5)
        preds_B[0:25] = 0 # A correct, B incorrect (b = 25)
        preds_A[25:30] = 0 # A incorrect, B correct (c = 5)
        
        res = calculate_mcnemar_test(preds_A, preds_B, y_true)
        
        self.assertEqual(res["contingency_table"][0][1], 25)
        self.assertEqual(res["contingency_table"][1][0], 5)
        # Yates corrected stat = (|25 - 5| - 1)^2 / 30 = 19^2 / 30 = 361 / 30 = 12.0333
        self.assertAlmostEqual(res["statistic"], 12.033333333333333, places=4)
        self.assertTrue(res["significant"])
        self.assertLess(res["p_value"], 0.05)

    def test_mcnemar_input_validation(self):
        """
        Verify ValueError on mismatched lengths.
        """
        with self.assertRaises(ValueError):
            calculate_mcnemar_test([1, 1], [1], [1, 1])

    def test_wilcoxon_standard(self):
        """
        Verify Wilcoxon signed-rank significance calculations.
        """
        scores_A = [0.8, 0.82, 0.79, 0.85, 0.81, 0.83, 0.84, 0.82, 0.86, 0.80]
        scores_B = [0.72, 0.74, 0.71, 0.78, 0.70, 0.75, 0.73, 0.72, 0.76, 0.71] # A is consistently better
        
        res = calculate_wilcoxon_test(scores_A, scores_B)
        self.assertIn("p_value", res)
        self.assertIn("statistic", res)
        self.assertLess(res["p_value"], 0.05)
        self.assertTrue(res["significant"])

    def test_wilcoxon_zero_diff(self):
        """
        Verify Wilcoxon handling when differences are completely zero.
        """
        scores_A = [0.8, 0.8, 0.8]
        scores_B = [0.8, 0.8, 0.8]
        
        res = calculate_wilcoxon_test(scores_A, scores_B)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["statistic"], 0.0)
        self.assertFalse(res["significant"])

if __name__ == "__main__":
    unittest.main()
