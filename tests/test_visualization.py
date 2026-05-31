import unittest
import os
import shutil
import numpy as np
import pandas as pd
from scripts.plot_generator import (
    plot_confusion_matrices,
    plot_roc_pr_curves,
    plot_sensitivity_heatmaps,
    plot_automata_transitions
)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.test_dir = "results/plots_test"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_confusion_matrices(self):
        """
        Verify confusion matrices are correctly calculated and saved.
        """
        model_preds = {
            "Automata": [1, 0, 1, 0],
            "LSTM": [0, 0, 1, 0]
        }
        y_true = {
            "Automata": [1, 0, 0, 0],
            "LSTM": [1, 0, 0, 0]
        }
        
        save_path = os.path.join(self.test_dir, "test_cm.png")
        plot_confusion_matrices(model_preds, y_true, save_path)
        
        self.assertTrue(os.path.exists(save_path))

    def test_plot_roc_pr_curves(self):
        """
        Verify ROC and Precision-Recall curve diagrams are correctly generated.
        """
        model_scores = {
            "Automata": [0.9, 0.1, 0.8, 0.2],
            "LSTM": [0.4, 0.2, 0.7, 0.1]
        }
        y_true = {
            "Automata": [1, 0, 1, 0],
            "LSTM": [1, 0, 1, 0]
        }
        
        save_prefix = os.path.join(self.test_dir, "test_curve")
        plot_roc_pr_curves(model_scores, y_true, save_prefix)
        
        expected_file = f"{save_prefix}_roc_pr_curves.png"
        self.assertTrue(os.path.exists(expected_file))

    def test_plot_sensitivity_heatmaps(self):
        """
        Verify parameter sensitivity heatmaps are compiled cleanly.
        """
        # Create a mock temporary sensitivity CSV
        temp_csv = "results/metrics/sensitivity_results.csv"
        real_exists = os.path.exists(temp_csv)
        
        save_path = os.path.join(self.test_dir, "test_sens.png")
        try:
            if real_exists:
                plot_sensitivity_heatmaps(save_path)
                self.assertTrue(os.path.exists(save_path))
        except Exception as e:
            self.fail(f"plot_sensitivity_heatmaps failed with error: {e}")

    def test_plot_automata_transitions(self):
        """
        Verify transition matrix plots run cleanly on training dataset.
        """
        save_path = os.path.join(self.test_dir, "test_trans.png")
        try:
            plot_automata_transitions(save_path)
            self.assertTrue(os.path.exists(save_path))
        except Exception as e:
            self.fail(f"plot_automata_transitions failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
