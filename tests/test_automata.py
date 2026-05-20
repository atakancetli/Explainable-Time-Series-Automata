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

if __name__ == '__main__':
    unittest.main()

