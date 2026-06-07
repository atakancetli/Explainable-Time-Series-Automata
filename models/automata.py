import numpy as np
from scipy.stats import norm
import logging
from typing import Union, List, Dict, Tuple, Set

logger = logging.getLogger("TimeSeriesAutomata")

class TimeSeriesAutomata:
    """
    Symbolic Time Series Automata classifier core.
    Implements Piecewise Aggregate Approximation (PAA) and Symbolic Aggregate Approximation (SAX)
    with dynamic Gaussian breakpoints for time-series discretization and sequence analysis.
    
    Mathematical details & Features:
      1. PAA: Reduces dimensionality from N to w using fractional overlap bin averages.
      2. SAX: Discretizes PAA coefficients into symbolic sequences ('a', 'b', 'c', ...)
         based on equiprobable regions separated by standard normal (Gaussian) breakpoints.
      3. Levenshtein Distance & Unseen Pattern Mapping: Handles out-of-vocabulary test patterns.
      4. Path Probability & Anomaly Scoring: Computes rolling sequence path probabilities for explaining predictions.

    """
    def __init__(self, alphabet_size=3, word_size=4):
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

    def generate_states(self, time_series: np.ndarray, window_size: int = 10) -> List[Union[str, Tuple[str, ...]]]:
        """
        Slices a normalized time series using a sliding window and converts each slice
        into a symbolic state representation via PAA and SAX.

        Parameters:
        -----------
        time_series : np.ndarray
            The input normalized time series data.
        window_size : int
            The raw sliding window size L.

        Returns:
        --------
        states : List[Union[str, Tuple[str, ...]]]
            A chronological list of states. Each state is a string (for 1D series) 
            or a tuple of strings (for 2D series).
        """
        arr = np.asarray(time_series)
        n = arr.shape[0]

        if n < window_size:
            raise ValueError(f"Time series length N={n} is less than window_size L={window_size}.")

        states = []
        for i in range(n - window_size + 1):
            window_slice = arr[i : i + window_size]
            paa_coeffs = self.paa_transform(window_slice)
            sax_word = self.sax_transform(paa_coeffs)
            
            if isinstance(sax_word, list):
                states.append(tuple(sax_word))
            else:
                states.append(sax_word)

        return states

    def build_transition_matrix(self, state_sequence: List[Union[str, Tuple[str, ...]]]) -> None:
        """
        Builds the transition frequency matrix and tracks individual state counts
        from a chronological sequence of symbolic states.

        Parameters:
        -----------
        state_sequence : List[Union[str, Tuple[str, ...]]]
            A chronological list of states.
        """
        self.transitions = {}
        self.state_counts = {}
        self.unique_states = set()

        if len(state_sequence) < 2:
            logger.warning("State sequence is too short to construct transitions.")
            return

        for i in range(len(state_sequence) - 1):
            s_curr = state_sequence[i]
            s_next = state_sequence[i + 1]

            self.unique_states.add(s_curr)
            self.unique_states.add(s_next)

            self.state_counts[s_curr] = self.state_counts.get(s_curr, 0) + 1

            if s_curr not in self.transitions:
                self.transitions[s_curr] = {}
            self.transitions[s_curr][s_next] = self.transitions[s_curr].get(s_next, 0) + 1

        s_last = state_sequence[-1]
        self.state_counts[s_last] = self.state_counts.get(s_last, 0) + 1
        
        logger.info(f"Transition frequency matrix built successfully. Unique states: {len(self.unique_states)}")

    def fit(self, time_series: np.ndarray, window_size: int = 10) -> "TimeSeriesAutomata":
        """
        Fits the TimeSeriesAutomata model by slicing the input time series,
        converting it to a SAX state sequence, and constructing the transition matrix.

        Parameters:
        -----------
        time_series : np.ndarray
            The training normalized time series.
        window_size : int
            The sliding window size.
        """
        logger.info(f"Fitting TimeSeriesAutomata with window_size={window_size}...")
        states = self.generate_states(time_series, window_size=window_size)
        self.build_transition_matrix(states)
        logger.info("Fitting TimeSeriesAutomata completed successfully.")
        return self

    def get_transition_distribution(self, state: Union[str, Tuple[str, ...]]) -> List[Tuple[Union[str, Tuple[str, ...]], int, float]]:
        """
        Retrieves the transition count and frequency distribution for a given state.

        Parameters:
        -----------
        state : Union[str, Tuple[str, ...]]
            The source state.

        Returns:
        --------
        distribution : List[Tuple[Union[str, Tuple[str, ...]], int, float]]
            A sorted list of tuples (next_state, count, probability) in descending order of frequency.
        """
        if not hasattr(self, 'transitions') or state not in self.transitions:
            return []

        next_states = self.transitions[state]
        total_transitions = sum(next_states.values())

        dist = []
        for n_state, count in next_states.items():
            prob = count / total_transitions
            dist.append((n_state, count, prob))

        dist.sort(key=lambda item: item[1], reverse=True)
        return dist

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Computes the standard Levenshtein (edit) distance between two strings s1 and s2.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        return dp[m][n]

    def state_distance(self, s1: Union[str, Tuple[str, ...]], s2: Union[str, Tuple[str, ...]]) -> float:
        """
        Computes the distance between two states.
        For univariate states (strings), returns Levenshtein distance.
        For multivariate states (tuples of strings), returns the sum of component-wise Levenshtein distances.
        If types do not match, returns infinity.
        """
        if isinstance(s1, tuple) and isinstance(s2, tuple):
            if len(s1) != len(s2):
                return float('inf')
            return float(sum(self.levenshtein_distance(x1, x2) for x1, x2 in zip(s1, s2)))
        elif isinstance(s1, str) and isinstance(s2, str):
            return float(self.levenshtein_distance(s1, s2))
        else:
            return float('inf')

    def _map_unseen_state(self, state: Union[str, Tuple[str, ...]]) -> Union[str, Tuple[str, ...]]:
        """
        Maps an unseen state (not in self.unique_states) to the closest known state using state_distance.
        If the state is already in self.unique_states, returns the state itself.
        If self.unique_states is empty, returns the state itself.
        If multiple states have the same minimal distance, returns the one with the highest state count.
        If counts are also equal, uses lexicographical sorting of their string representations for deterministic tie-breaking.
        """
        if not hasattr(self, 'unique_states') or not self.unique_states:
            return state
            
        if state in self.unique_states:
            return state

        best_state = None
        min_dist = float('inf')
        max_count = -1

        for known_state in self.unique_states:
            dist = self.state_distance(state, known_state)
            if dist < min_dist:
                min_dist = dist
                best_state = known_state
                max_count = self.state_counts.get(known_state, 0)
            elif dist == min_dist:
                count = self.state_counts.get(known_state, 0)
                if count > max_count:
                    max_count = count
                    best_state = known_state
                elif count == max_count:
                    # Deterministic secondary tie-breaker
                    if str(known_state) < str(best_state):
                        best_state = known_state
                        
        return best_state

    def calculate_path_probability(self, state_sequence: List[Union[str, Tuple[str, ...]]], min_prob: float = 1e-6) -> float:
        """
        Calculates the path probability of a sequence of states using transition probabilities.
        First maps any unseen states to their nearest seen states in training data.
        
        Parameters:
        -----------
        state_sequence : List[Union[str, Tuple[str, ...]]]
            The sequence of symbolic states (words) to analyze.
        min_prob : float
            Minimal default probability to prevent zero-likelihood multiplication.
            
        Returns:
        --------
        path_prob : float
            The cumulative path probability (product of successive transition probabilities).
        """
        if len(state_sequence) < 2:
            return 1.0

        mapped_sequence = [self._map_unseen_state(s) for s in state_sequence]
        
        path_prob = 1.0
        for i in range(len(mapped_sequence) - 1):
            s_curr = mapped_sequence[i]
            s_next = mapped_sequence[i + 1]
            
            # Calculate transition probability P(s_curr -> s_next)
            if hasattr(self, 'transitions') and s_curr in self.transitions:
                outgoing = self.transitions[s_curr]
                total_outgoing = sum(outgoing.values())
                if total_outgoing > 0 and s_next in outgoing:
                    prob = outgoing[s_next] / total_outgoing
                else:
                    prob = min_prob
            else:
                prob = min_prob
                
            path_prob *= prob
            
        return path_prob

    def predict_anomaly(self, time_series: np.ndarray, window_size: int = 10, threshold: float = 0.01, path_window: int = 2, min_prob: float = 1e-6) -> np.ndarray:
        """
        Predicts anomalies in a test time series.
        For each sliding window, maps its state, evaluates rolling path probability,
        and marks it as an anomaly if the path probability is below threshold.
        
        Parameters:
        -----------
        time_series : np.ndarray
            The test time series.
        window_size : int
            The sliding window size L.
        threshold : float
            The anomaly detection path probability threshold.
        path_window : int
            The length of the rolling transitions window to evaluate (e.g. 2).
        min_prob : float
            Minimal probability smoothing.
            
        Returns:
        --------
        predictions : np.ndarray of shape (N - window_size + 1,)
            Binary array where 1 indicates anomaly and 0 indicates normal.
        """
        states = self.generate_states(time_series, window_size=window_size)
        num_windows = len(states)
        predictions = np.zeros(num_windows, dtype=int)
        
        # S0 has no previous state for transition, so by default it's normal (0).
        # We start predicting from window 1 onwards.
        for t in range(1, num_windows):
            # Sequence of states for rolling path ending at t or forward-looking from t-1
            # We use forward-looking of path_window starting at t-1 (matching PDF transitions: S_{t-1} -> S_t -> S_{t+1})
            end_idx = min(t - 1 + path_window + 1, num_windows)
            active_sequence = states[t - 1 : end_idx]
            
            prob = self.calculate_path_probability(active_sequence, min_prob=min_prob)
            if prob < threshold:
                predictions[t] = 1
                
        return predictions

    def explain_decision(self, time_series: np.ndarray, window_size: int = 10, threshold: float = 0.01, path_window: int = 2, min_prob: float = 1e-6) -> List[Dict]:
        """
        Generates structured, JSON-compliant explainability diagnostics for each step
        in the test time series anomaly detection process.
        
        Parameters:
        -----------
        time_series : np.ndarray
            The test time series.
        window_size : int
            The sliding window size L.
        threshold : float
            The anomaly detection path probability threshold.
        path_window : int
            The length of the rolling transitions window.
        min_prob : float
            Minimal probability smoothing.
            
        Returns:
        --------
        explanations : List[Dict]
            A list of diagnostic dictionaries, one for each prediction step (from step 1 onwards).
        """
        states = self.generate_states(time_series, window_size=window_size)
        num_windows = len(states)
        explanations = []
        
        for t in range(1, num_windows):
            s_curr_raw = states[t]
            s_prev_raw = states[t - 1]
            
            s_curr_mapped = self._map_unseen_state(s_curr_raw)
            s_prev_mapped = self._map_unseen_state(s_prev_raw)
            
            status = "seen" if s_curr_raw in self.unique_states else "unseen"
            
            # Identify active transitions for the rolling path starting at t-1
            end_idx = min(t - 1 + path_window + 1, num_windows)
            active_sequence = states[t - 1 : end_idx]
            mapped_active = [self._map_unseen_state(s) for s in active_sequence]
            
            # List individual transitions
            transition_list = []
            for i in range(len(mapped_active) - 1):
                s_from = mapped_active[i]
                s_to = mapped_active[i + 1]
                prob = min_prob
                if hasattr(self, 'transitions') and s_from in self.transitions:
                    outgoing = self.transitions[s_from]
                    total = sum(outgoing.values())
                    if total > 0 and s_to in outgoing:
                        prob = outgoing[s_to] / total
                transition_list.append({
                    "from": s_from,
                    "to": s_to,
                    "probability": prob
                })
                
            path_prob = self.calculate_path_probability(active_sequence, min_prob=min_prob)
            decision = "anomaly" if path_prob < threshold else "normal"
            
            # JSON-compliant structure matching Sections X.A & X.F exactly
            diag = {
                "time_step": t,
                "state": s_prev_mapped,
                "pattern": s_curr_raw,
                "status": status,
                "mapped_to": s_curr_mapped,
                "distance": self.state_distance(s_curr_raw, s_curr_mapped) if status == "unseen" else 0.0,
                "transitions": transition_list,
                "probability": path_prob,
                "decision": decision,
                "confidence_score": path_prob,
                "reason": "Low probability path detected" if decision == "anomaly" else "Normal path transition probability"
            }
            explanations.append(diag)
            
        return explanations




