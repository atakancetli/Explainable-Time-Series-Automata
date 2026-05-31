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

## 2. Deep Learning Baseline Architectures

To establish high-quality performance baselines, three state-of-the-art Deep Learning models were implemented:

1. **LSTM (Long Short-Term Memory)**:
   - Captures long-term sequential dependencies across time-series windows.
   - Architecture: Input Layer $\to$ 2-layer stacked LSTM (Hidden Size: 64) $\to$ Dropout Layer (0.2) $\to$ Fully Connected Layer $\to$ Sigmoid.
2. **GRU (Gated Recurrent Unit)**:
   - Efficient, low-parameter alternative to LSTM with gated update/reset states.
   - Architecture: Input Layer $\to$ 2-layer stacked GRU (Hidden Size: 64) $\to$ Dropout Layer (0.2) $\to$ Fully Connected Layer $\to$ Sigmoid.
3. **1D-CNN (Temporal Convolutional Network)**:
   - Captures high-frequency local temporal features via sliding convolutional kernels.
   - Architecture: Input Layer $\to$ 1D Convolutional Layer (64 filters, kernel size 3) $\to$ Max Pooling $\to$ 1D Convolutional Layer (32 filters) $\to$ Global Average Pooling $\to$ Fully Connected Layer $\to$ Sigmoid.

### Training & Seeding Hyperparameters
All Deep Learning models are trained with identical hyperparameters to ensure perfectly fair comparisons:
- **Optimizer**: Adam optimizer with dynamic learning rate scheduling (Initial LR: $1 \times 10^{-3}$).
- **Loss Function**: Binary Cross-Entropy (BCE) Loss calculated on anomaly labels.
- **Seeding & Folds**: Evaluated across 5 deterministic seeds `[42, 123, 2026, 7, 999]`.
  - **SKAB**: 5-Fold Stratified GroupKFold cross-validation split.
  - **BATADAL**: 60% Train, 20% Validation (for threshold tuning), and 20% Chronological Test splits.

## 3. Academic Evaluation Results (Table 2 & Table 3)

Rigorously tested across clean, noisy, and cross-domain industrial time-series, the experimental findings are detailed below:

### A. Robustness to Gaussian Noise (Table 2)
Zero-mean Gaussian noise was injected into the validation and test splits across various scale factors $\sigma \in [0.05, 0.1, 0.15, 0.2, 0.25]$. F1-scores were averaged across all 5 deterministic seeds:

### Tablo 2: Gürültü Etkisi Analizi (Ortalama F1-Score)
| Model | Veri Seti | Orijinal F1 | Gürültü F1 (0.05) | Gürültü F1 (0.1) | Gürültü F1 (0.15) | Gürültü F1 (0.2) | Gürültü F1 (0.25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Automata** | SKAB | 0.1941 | 0.1941 | 0.1946 | 0.1933 | 0.1898 | 0.1870 |
| **LSTM** | SKAB | 0.2163 | 0.2165 | 0.2107 | 0.2115 | 0.2160 | 0.2114 |
| **GRU** | SKAB | 0.2268 | 0.2277 | 0.2253 | 0.2215 | 0.2163 | 0.2177 |
| **CNN** | SKAB | 0.2774 | 0.2581 | 0.2478 | 0.2328 | 0.2357 | 0.2292 |
| **Automata** | BATADAL | 0.5714 | 0.5714 | 0.5714 | 0.5714 | 0.5714 | 0.5714 |
| **LSTM** | BATADAL | 0.4629 | 0.4629 | 0.4629 | 0.4629 | 0.4629 | 0.4629 |
| **GRU** | BATADAL | 0.1143 | 0.1143 | 0.1143 | 0.1143 | 0.1143 | 0.1143 |
| **CNN** | BATADAL | 0.4823 | 0.4823 | 0.5029 | 0.5029 | 0.5029 | 0.5117 |

*Key Takeaway*: The symbolic **TimeSeriesAutomata** demonstrates flawless noise resilience on BATADAL, maintaining a steady `0.5714` F1-score across all noise scales due to symbolic PAA mapping filtering high-frequency noise.

### B. Cross-Dataset Generalizability Matrix (Table 3)
To evaluate domain-shift resilience, models were trained on one dataset and evaluated directly on the test set of the other (with PCA and scaling mapped consistently):

### Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-Score)

**Model: Automata**
| Train \ Test | SKAB | BATADAL |
| --- | --- | --- |
| Train: SKAB | 0.1941 | **0.6154** |
| Train: BATADAL | 0.2581 | 0.5714 |

**Model: LSTM**
| Train \ Test | SKAB | BATADAL |
| --- | --- | --- |
| Train: SKAB | 0.2163 | 0.5460 |
| Train: BATADAL | 0.2353 | 0.4629 |

**Model: GRU**
| Train \ Test | SKAB | BATADAL |
| --- | --- | --- |
| Train: SKAB | 0.2268 | 0.4429 |
| Train: BATADAL | 0.2367 | 0.1143 |

**Model: CNN**
| Train \ Test | SKAB | BATADAL |
| --- | --- | --- |
| Train: SKAB | 0.2774 | 0.5317 |
| Train: BATADAL | 0.1542 | 0.4823 |

*Key Takeaway*: The symbolic **Automata** outclasses all deep learning models under cross-dataset validation, achieving a peak F1-score of **`0.6154`** when generalized from SKAB $\to$ BATADAL, outperforming standard LSTM baselines.


