import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger("TimeSeriesAutomata")

class TimeSeriesAutomata:
    """
    Symbolic Time Series Automata classifier core.
    Implements Piecewise Aggregate Approximation (PAA) and Symbolic Aggregate Approximation (SAX)
    with dynamic Gaussian breakpoints for time-series discretization and sequence analysis.
    
    Mathematical details:
      1. PAA: Reduces dimensionality from N to w using fractional overlap bin averages.
      2. SAX: Discretizes PAA coefficients into symbolic sequences ('a', 'b', 'c', ...)
         based on equiprobable regions separated by standard normal (Gaussian) breakpoints.
    """
    def __init__(self, alphabet_size=5, word_size=10):
        """
        Initializes the TimeSeriesAutomata model.

        Parameters:
        -----------
        alphabet_size : int
            Size of the symbolic alphabet (a, b, c, ...). Must be between 3 and 26.
        word_size : int
            Target dimensionality reduction size (number of PAA segments per window).
        """
        if not (3 <= alphabet_size <= 26):
            raise ValueError("Alphabet size must be between 3 and 26.")
        
        self.alphabet_size = alphabet_size
        self.word_size = word_size
        
        logger.info(f"Initialized TimeSeriesAutomata with alphabet_size={alphabet_size}, word_size={word_size}")

    def paa_transform(self, x):
        """
        Applies Piecewise Aggregate Approximation (PAA) to a time series.
        Reduces dimensionality from N to word_size (w) using fractional overlap.

        Parameters:
        -----------
        x : array-like of shape (N,) or (N, D)
            The input time series signal.

        Returns:
        --------
        paa_coeffs : np.ndarray of shape (w,) or (w, D)
            The PAA coefficients.
        """
        x_arr = np.asarray(x, dtype=float)
        n = x_arr.shape[0]
        w = self.word_size

        if n < w:
            raise ValueError(f"Time series length N={n} is less than word_size w={w}. Cannot reduce.")

        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr[:, np.newaxis]

        n, d = x_arr.shape
        paa_coeffs = np.zeros((w, d))

        for i in range(w):
            start = i * n / w
            end = (i + 1) * n / w
            
            j_indices = np.arange(n)
            starts = np.maximum(j_indices, start)
            ends = np.minimum(j_indices + 1, end)
            overlaps = np.maximum(0.0, ends - starts)
            
            paa_coeffs[i] = np.sum(x_arr * overlaps[:, np.newaxis], axis=0) / (n / w)

        if is_1d:
            return paa_coeffs.squeeze(axis=1)
        return paa_coeffs

    def get_breakpoints(self):
        """
        Dynamically computes equiprobable Gaussian breakpoints based on the alphabet size.

        Returns:
        --------
        breakpoints : np.ndarray of shape (alphabet_size - 1,)
            The Gaussian breakpoints separating the probability regions.
        """
        probabilities = np.linspace(1 / self.alphabet_size, 1 - 1 / self.alphabet_size, self.alphabet_size - 1)
        breakpoints = norm.ppf(probabilities)
        return breakpoints

    def sax_transform(self, paa_coeffs):
        """
        Converts PAA coefficients into SAX symbolic representations.

        Parameters:
        -----------
        paa_coeffs : array-like of shape (w,) or (w, D)
            The Piecewise Aggregate Approximation coefficients.

        Returns:
        --------
        sax_sequence : str or list of str
            A single symbolic string if input is 1D, or a list of strings (one per dimension) if 2D.
        """
        paa_arr = np.asarray(paa_coeffs, dtype=float)
        is_1d = (paa_arr.ndim == 1)
        if is_1d:
            paa_arr = paa_arr[:, np.newaxis]

        w, d = paa_arr.shape
        breakpoints = self.get_breakpoints()
        
        sax_sequences = []
        for col in range(d):
            indices = np.digitize(paa_arr[:, col], breakpoints)
            symbols = [chr(97 + idx) for idx in indices]
            sax_sequences.append("".join(symbols))

        if is_1d:
            return sax_sequences[0]
        return sax_sequences
