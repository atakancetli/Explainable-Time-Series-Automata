import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger("TimeSeriesAutomata")

class TimeSeriesAutomata:
    """
    Symbolic Time Series Automata classifier core.
    Implements Piecewise Aggregate Approximation (PAA) and Symbolic Aggregate Approximation (SAX)
    with dynamic Gaussian breakpoints for time-series discretization and sequence analysis.
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
