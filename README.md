# Explainable Time Series Automata

An interpretable, symbolic anomaly detection model based on sliding windows, **Piecewise Aggregate Approximation (PAA)**, and **Symbolic Aggregate Approximation (SAX)**, rigorously compared against deep learning baselines (**LSTM**, **GRU**, **1D-CNN**) on complex multivariate industrial datasets (**SKAB** and **BATADAL**).

This repository represents the official implementation and academic research findings for the *Explainable Time Series Automata* project, exploring the balance between model interpretability and diagnostic accuracy in critical industrial control systems (ICS).

---

## 1. Mathematical Design of TimeSeriesAutomata

The core symbolic model discretizes continuous multivariate sensor streams into distinct symbolic states, mapping transition probabilities to construct a deterministic finite automaton (DFA) representing normal system boundaries.

### A. Piecewise Aggregate Approximation (PAA)
To reduce temporal dimensionality and filter high-frequency sensor noise, each sliding window of length $L$ is divided into $w$ equal-sized segments. The mean value of each segment is calculated as:

$$\bar{x}_{i} = \frac{w}{L} \sum_{j=\frac{L}{w}(i-1)+1}^{\frac{L}{w}i} x_{j}$$

This maps a continuous window $X = \{x_1, x_2, \dots, x_L\}$ to a low-dimensional vector $\bar{X} = \{\bar{x}_1, \bar{x}_2, \dots, \bar{x}_w\}$.

### B. Symbolic Aggregate Approximation (SAX)
The continuous PAA coefficients are discretized into symbolic words using Gaussian breakpoints. Given an alphabet size $a$, the real line is divided into $a$ equiprobable regions determined by Gaussian breakpoints $B = \{\beta_1, \beta_2, \dots, \beta_{a-1}\}$:

$$\text{SAX}(\bar{x}_{i}) = \alpha_{j} \quad \text{if} \quad \beta_{j-1} \le \bar{x}_{i} < \beta_{j}$$

This maps the continuous vector $\bar{X}$ to a symbolic word of length $w$ (e.g., `'aabc'`).

### C. Transition Probability and Path Anomaly Scores
During training, the Automata maps all SAX state transitions observed in normal operating data. The transition probability from state $S_A$ to state $S_B$ is computed as:

$$P(S_A \to S_B) = \frac{\text{Count}(S_A \to S_B)}{\sum_{S' \in \mathcal{S}} \text{Count}(S_A \to S')}$$

For an inference sequence of states $\{S_1, S_2, \dots, S_N\}$, the rolling path probability over a sliding path window of length $k$ is calculated as the cumulative product of successive transition probabilities:

$$\text{PathProb}(S_{t-k}, \dots, S_t) = \prod_{i=t-k+1}^{t} P(S_{i-1} \to S_i)$$

A state sequence is marked as an anomaly if its path probability falls below a dynamically optimized decision threshold $\theta$:

$$Y_t = \begin{cases} 1 & \text{if } \text{PathProb}(S_{t-k}, \dots, S_t) < \theta \\ 0 & \text{otherwise} \end{cases}$$

### D. Unseen Pattern Management using Levenshtein Distance
If a state sequence $\{S_1, S_2, \dots, S_k\}$ contains a state $S_u$ that was never visited during training, standard probabilistic models collapse to zero probability. To prevent false positives, we compute the Levenshtein distance between $S_u$ and all known training states:

$$\text{Lev}(S_u, S_v) = \text{edit\_distance}(S_u, S_v)$$

We map $S_u$ to the closest known state $S_v^*$ that minimizes the edit distance, transferring its transition probabilities with a minor smoothing penalty $\sigma$:

$$P(S_{t-1} \to S_u) \approx P(S_{t-1} \to S_v^*) \cdot \sigma$$
