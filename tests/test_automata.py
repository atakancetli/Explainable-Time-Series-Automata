import unittest
import numpy as np
from models.automata import TimeSeriesAutomata

class TestTimeSeriesAutomata(unittest.TestCase):
    def setUp(self):
        self.automata = TimeSeriesAutomata(alphabet_size=5, word_size=4)

    def test_initialization(self):
        self.assertEqual(self.automata.alphabet_size, 5)
        self.assertEqual(self.automata.word_size, 4)
        
        with self.assertRaises(ValueError):
            TimeSeriesAutomata(alphabet_size=2)
        with self.assertRaises(ValueError):
            TimeSeriesAutomata(alphabet_size=27)

    def test_get_breakpoints(self):
        breakpoints = self.automata.get_breakpoints()
        self.assertEqual(len(breakpoints), 4)
        np.testing.assert_array_almost_equal(breakpoints, [-0.841621, -0.253347, 0.253347, 0.841621], decimal=5)

    def test_paa_transform_1d(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        paa = self.automata.paa_transform(series)
        self.assertEqual(paa.shape, (4,))
        np.testing.assert_array_almost_equal(paa, [1.5, 3.5, 5.5, 7.5])

    def test_paa_transform_fractional(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        paa = self.automata.paa_transform(series)
        
        expected_paa = np.array([
            (1.0*1.0 + 0.5*2.0)/1.5,
            (0.5*2.0 + 1.0*3.0)/1.5,
            (1.0*4.0 + 0.5*5.0)/1.5,
            (0.5*5.0 + 1.0*6.0)/1.5
        ])
        np.testing.assert_array_almost_equal(paa, expected_paa)

    def test_paa_transform_2d(self):
        series = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0]
        ])
        paa = self.automata.paa_transform(series)
        self.assertEqual(paa.shape, (4, 2))
        np.testing.assert_array_almost_equal(paa[:, 0], [1.5, 3.5, 5.5, 7.5])
        np.testing.assert_array_almost_equal(paa[:, 1], [15.0, 35.0, 55.0, 75.0])

    def test_sax_transform_1d(self):
        paa_coeffs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.automata.word_size = 5
        sax = self.automata.sax_transform(paa_coeffs)
        self.assertEqual(sax, "abcde")

    def test_sax_transform_2d(self):
        paa_coeffs = np.array([
            [-1.0, 1.0],
            [1.0, -1.0]
        ])
        self.automata.word_size = 2
        sax = self.automata.sax_transform(paa_coeffs)
        self.assertEqual(len(sax), 2)
        self.assertEqual(sax[0], "ae")
        self.assertEqual(sax[1], "ea")

    def test_generate_states_1d(self):
        # Time series of 15 elements, window_size=10, word_size=4
        # We expect 15 - 10 + 1 = 6 states
        self.automata.word_size = 4
        series = np.arange(15, dtype=float)
        states = self.automata.generate_states(series, window_size=10)
        self.assertEqual(len(states), 6)
        self.assertTrue(all(isinstance(s, str) for s in states))
        self.assertTrue(all(len(s) == 4 for s in states))

    def test_generate_states_2d(self):
        # 2D series: 12 elements, 2 features, window_size=8
        # We expect 12 - 8 + 1 = 5 states, each state should be a tuple of length 2
        self.automata.word_size = 4
        series = np.column_stack([np.arange(12), np.arange(12)*10])
        states = self.automata.generate_states(series, window_size=8)
        self.assertEqual(len(states), 5)
        self.assertTrue(all(isinstance(s, tuple) for s in states))
        self.assertTrue(all(len(s) == 2 for s in states))
        self.assertTrue(all(len(dim_s) == 4 for s in states for dim_s in s))

    def test_generate_states_value_error(self):
        series = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.automata.generate_states(series, window_size=5)

    def test_build_transition_matrix(self):
        state_sequence = ['a', 'b', 'a', 'b', 'c', 'a']
        self.automata.build_transition_matrix(state_sequence)
        
        # Check unique states
        self.assertEqual(self.automata.unique_states, {'a', 'b', 'c'})
        
        # Check state counts
        expected_counts = {'a': 3, 'b': 2, 'c': 1}
        self.assertEqual(self.automata.state_counts, expected_counts)
        
        # Check transition dictionary structure and counts
        # Transitions are:
        # a -> b (2 times)
        # b -> a (1 time)
        # b -> c (1 time)
        # c -> a (1 time)
        expected_transitions = {
            'a': {'b': 2},
            'b': {'a': 1, 'c': 1},
            'c': {'a': 1}
        }
        self.assertEqual(self.automata.transitions, expected_transitions)

    def test_build_transition_matrix_short(self):
        # Testing transition construction with less than 2 elements
        self.automata.build_transition_matrix(['a'])
        self.assertEqual(len(self.automata.transitions), 0)

    def test_get_transition_distribution(self):
        state_sequence = ['a', 'b', 'a', 'b', 'c', 'a']
        self.automata.build_transition_matrix(state_sequence)
        
        # Distribution for 'b' should have 'a' and 'c' with prob 0.5 each
        dist_b = self.automata.get_transition_distribution('b')
        self.assertEqual(len(dist_b), 2)
        # Order should be sorted by count/probability, but since counts are equal (1),
        # any order is fine. Let's check contents.
        states_in_dist = [item[0] for item in dist_b]
        self.assertIn('a', states_in_dist)
        self.assertIn('c', states_in_dist)
        for state, count, prob in dist_b:
            self.assertEqual(count, 1)
            self.assertAlmostEqual(prob, 0.5)

        # Distribution for non-existent or leaf/last state 'c'
        # 'c' transitions to 'a' 1 time
        dist_c = self.automata.get_transition_distribution('c')
        self.assertEqual(dist_c, [('a', 1, 1.0)])

        # Distribution for state not in transitions
        dist_none = self.automata.get_transition_distribution('d')
        self.assertEqual(dist_none, [])

    def test_fit_pipeline(self):
        series = np.sin(np.linspace(0, 10, 50))
        self.automata.word_size = 5
        self.automata.fit(series, window_size=15)
        
        # Verify transition dictionary was created
        self.assertTrue(hasattr(self.automata, 'transitions'))
        self.assertTrue(hasattr(self.automata, 'state_counts'))
        self.assertTrue(len(self.automata.unique_states) > 0)

    def test_levenshtein_distance(self):
        self.assertEqual(TimeSeriesAutomata.levenshtein_distance("abc", "abc"), 0)
        self.assertEqual(TimeSeriesAutomata.levenshtein_distance("abc", "abd"), 1)
        self.assertEqual(TimeSeriesAutomata.levenshtein_distance("abc", "abcd"), 1)
        self.assertEqual(TimeSeriesAutomata.levenshtein_distance("abc", "ac"), 1)
        self.assertEqual(TimeSeriesAutomata.levenshtein_distance("kitten", "sitting"), 3)

    def test_state_distance(self):
        # Univariate
        self.assertEqual(self.automata.state_distance("abc", "abd"), 1.0)
        # Multivariate
        self.assertEqual(self.automata.state_distance(("abc", "xyz"), ("abd", "xyw")), 2.0)
        # Type mismatch
        self.assertEqual(self.automata.state_distance("abc", ("abc",)), float('inf'))
        self.assertEqual(self.automata.state_distance(("abc",), "abc"), float('inf'))
        # Length mismatch
        self.assertEqual(self.automata.state_distance(("abc", "def"), ("abc",)), float('inf'))

    def test_unseen_state_mapping(self):
        # Set up a known set of states
        self.automata.unique_states = {"abc", "bcd", "cde"}
        self.automata.state_counts = {"abc": 5, "bcd": 2, "cde": 5}
        
        # Test exact match
        self.assertEqual(self.automata._map_unseen_state("abc"), "abc")
        
        # Test mapping to nearest
        self.assertEqual(self.automata._map_unseen_state("abb"), "abc") # dist("abb", "abc")=1, others=3
        self.assertEqual(self.automata._map_unseen_state("bce"), "bcd") # dist("bce", "bcd")=1, others=3
        
        # Test tie-breaking by count
        # "abd" has dist=1 to both "abc" and "bcd". "abc" count is 5, "bcd" count is 2.
        self.assertEqual(self.automata._map_unseen_state("abd"), "abc")
        
        # Test tie-breaking lexicographically
        # "b" is dist 3 to both "abc" and "cde". Both have count 5.
        # Lexicographically, "abc" < "cde".
        self.assertEqual(self.automata._map_unseen_state("b"), "abc")

    def test_calculate_path_probability(self):
        # Set up a simple transition structure
        # a -> b (prob 0.8), a -> c (prob 0.2)
        # b -> c (prob 1.0)
        self.automata.unique_states = {"a", "b", "c"}
        self.automata.state_counts = {"a": 10, "b": 8, "c": 3}
        self.automata.transitions = {
            "a": {"b": 8, "c": 2},
            "b": {"c": 8}
        }
        
        # Path a -> b -> c probability: P(a -> b) * P(b -> c) = 0.8 * 1.0 = 0.8
        self.assertAlmostEqual(self.automata.calculate_path_probability(["a", "b", "c"]), 0.8)
        
        # Unseen state mapping in path probability
        # "x" is unseen, maps to nearest seen state. Let's make "x" map to "a".
        # dist("x", "a") = 1, dist("x", "b") = 1, dist("x", "c") = 1.
        # Counts are "a": 10, "b": 8, "c": 3. "a" has the highest count, so "x" maps to "a".
        # Path x -> b -> c should map to a -> b -> c, probability 0.8
        self.assertAlmostEqual(self.automata.calculate_path_probability(["x", "b", "c"]), 0.8)
        
        # Zero transition smoothed
        # c -> a has transition count 0.
        self.assertAlmostEqual(self.automata.calculate_path_probability(["c", "a"], min_prob=0.05), 0.05)

    def test_predict_anomaly_and_explain(self):
        # Fit the automata on a simple sin wave
        series = np.sin(np.linspace(0, 20, 100))
        self.automata.word_size = 4
        self.automata.fit(series, window_size=10)
        
        # Test on the same series - should have normal path probabilities
        predictions_normal = self.automata.predict_anomaly(series, window_size=10, threshold=0.001)
        # Should be mostly normal (few or zero anomalies)
        self.assertTrue(np.mean(predictions_normal) < 0.2)
        
        # Now introduce a sudden flat line at the end of the test series (anomaly transition)
        test_series = series.copy()
        test_series[-20:] = 0.0
        
        predictions_anom = self.automata.predict_anomaly(test_series, window_size=10, threshold=0.001)
        explanations = self.automata.explain_decision(test_series, window_size=10, threshold=0.001)
        
        # Check that we detected the anomaly transition
        self.assertTrue(np.sum(predictions_anom) >= 1)
        
        # Verify explain_decision format matches Section X.F and X.A perfectly
        self.assertEqual(len(explanations), len(predictions_anom) - 1)
        
        # Check first explanation structure
        first_diag = explanations[0]
        self.assertEqual(first_diag["time_step"], 1)
        self.assertIn("state", first_diag)
        self.assertIn("pattern", first_diag)
        self.assertIn("status", first_diag)
        self.assertIn("mapped_to", first_diag)
        self.assertIn("distance", first_diag)
        self.assertIn("transitions", first_diag)
        self.assertIn("probability", first_diag)
        self.assertIn("decision", first_diag)
        self.assertIn("confidence_score", first_diag)
        self.assertIn("reason", first_diag)
        
        # Ensure transitions has correct schema
        self.assertTrue(len(first_diag["transitions"]) > 0)
        first_trans = first_diag["transitions"][0]
        self.assertIn("from", first_trans)
        self.assertIn("to", first_trans)
        self.assertIn("probability", first_trans)

    def test_find_best_threshold_automata(self):
        """
        Verifies that find_best_threshold_automata returns a valid threshold and runs cleanly.
        """
        from scripts.evaluate_automata import find_best_threshold_automata
        
        series = np.sin(np.linspace(0, 20, 50))
        self.automata.fit(series, window_size=10)
        
        val_labels = np.zeros(50, dtype=int)
        val_labels[20:25] = 1 # Introduce mock anomalies
        
        thresh, f1 = find_best_threshold_automata(self.automata, series, val_labels, window_size=10)
        self.assertTrue(0.0 <= thresh <= 1.0)
        self.assertTrue(0.0 <= f1 <= 1.0)

    def test_run_automata_skab_kfold(self):
        """
        Verifies that run_automata_skab_kfold runs cleanly for seed 42.
        """
        from scripts.evaluate_automata import run_automata_skab_kfold
        
        results = run_automata_skab_kfold(42)
        self.assertEqual(results["dataset"], "SKAB")
        self.assertEqual(results["model"], "Automata")
        self.assertEqual(results["seed"], 42)
        self.assertIn("f1", results)
        self.assertIn("accuracy", results)

if __name__ == '__main__':
    unittest.main()

