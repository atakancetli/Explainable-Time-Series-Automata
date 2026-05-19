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

if __name__ == '__main__':
    unittest.main()
