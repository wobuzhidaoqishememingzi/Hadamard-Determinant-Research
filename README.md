# PatternBoost for the Hadamard Maximal Determinant Problem

This repository contains the implementation, experimental logs, and empirical analysis for investigating the **Hadamard Maximal Determinant Problem** on $\{0, 1\}$-matrices using the **PatternBoost** framework (combining fast local search in Julia with Transformer-guided global search on GPUs).

* **Author:** Yuebi Li (University of California, Davis)
* **Faculty Mentor:** Dr. Luze Xu
* **Base Framework:** [PatternBoost (`transformers_math_experiments`)](https://github.com/zawagner22/transformers_math_experiments)
* **Sequence Reference:** [OEIS A003432](https://oeis.org/A003432)

---

## 1. Problem Formulation & Theoretical Background

The **Hadamard Maximal Determinant Problem** seeks the largest possible determinant $D(n) = \max |\det(A)|$ over all $n \times n$ binary matrices $A \in \{0, 1\}^{n \times n}$.

By appending a border of ones and applying elementary row operations, maximizing $|\det(A)|$ for $A \in \{0, 1\}^{n \times n}$ is algebraically equivalent to maximizing $|\det(H)|$ for a bipolar matrix $H \in \{-1, 1\}^{N \times N}$ of order $N = n + 1$:

$$|\det(H)| = 2^n |\det(A)|$$

Analytical upper bounds are derived via continuous spectral relaxation on the positive semi-definite Gram matrix $G = H^T H$, partitioned by $N \pmod 4$:

| Residue Class ($N = n+1$) | Upper Bound on $|\det(H)|$ | Necessary Attainability Condition |
| :--- | :--- | :--- |
| $N \equiv 0 \pmod 4$ | **Hadamard (1893):** $N^{N/2}$ | Existence of an order-$N$ Hadamard matrix |
| $N \equiv 1 \pmod 4$ | **Barba (1933):** $\sqrt{2N-1}(N-1)^{(N-1)/2}$ | $2N - 1$ must be a perfect square |
| $N \equiv 2 \pmod 4$ | **Ehlich–Wojtas (1964):** $(2N-2)(N-2)^{N/2 - 1}$ | $2N - 2$ must be a sum of two integer squares (Hasse–Minkowski) |
| $N \equiv 3 \pmod 4$ | **Ehlich (1964):** Block-diagonal Gram matrix bound | Unattainable for small $N$ (smallest theoretical candidate is $N=511$) |

---

## 2. Methodology

We adapt the iterative **PatternBoost** loop to search for optimal $\{0, 1\}$-matrices across orders $n = 6$ to $22$:

1. **Local Search (Julia):** A low-overhead local search routine climbs from candidate matrices to local maxima of $|\det(A)|$.
2. **Global Search (Transformer on NVIDIA GPUs):** A Transformer model is trained on top-scoring constructions from local search to learn structural patterns and sample new seeds (`sample-only` up to $50{,}000$).
3. **Isomorphism & Uniqueness Verification:** Discovered maximal matrices are deduplicated under row/column permutations and verified against known literature constructions using graph isomorphism (`equivalent_via_graph(A, B)`).
4. **High-Dimensional Probing ($n \ge 16$):** Because standard PatternBoost fails to discover upper-bound matrices from scratch within 20 epochs for $n \ge 16$, we evaluate two guided strategies:
   * **Manual Upper-Bound Injection ($n = 16\text{–}19$):** Injecting a known optimal matrix into the initial pool to test whether the Transformer can generalize and discover *other* non-equivalent maximizers.
   * **Single-Flip vs. Double-Flip Perturbation ($n = 16, 20\text{–}22$):** Perturbing a known optimal matrix to create near-optimal training seeds. Single-flip perturbations were trivially recovered in the first local search step; **double-flip** perturbations were introduced to increase learning difficulty and test structural recovery.

---

## 3. Experimental Results & Theoretical Benchmarks ($n = 8$ to $22$)

| $n$ ($N = n+1$) | Max Determinant $D(n)$ | Known # Maximizers | Theoretical Bound & Attainability | PatternBoost Result |
| :--- | :--- | :--- | :--- | :--- |
| 8 ($N=9$) | 56 | 1 | Barba ($2N-1=17$ not a square; unachievable) | Yes (Local Search) |
| 9 ($N=10$) | 144 | 1 | Ehlich–Wojtas ($2N-2=18=3^2+3^2$; achieved) | Yes (Local Search) |
| 10 ($N=11$) | 320 | 3 | Ehlich ($N \equiv 3 \pmod 4$; unachievable) | Yes (Local Search) |
| 11 ($N=12$) | 1,458 | 1 | Hadamard ($N \equiv 0 \pmod 4$; achieved) | Yes (Local Search) |
| 12 ($N=13$) | 3,645 | 1 | Barba ($2N-1=25=5^2$; achieved) | Yes (Local Search) |
| 13 ($N=14$) | 9,477 | 1 | Ehlich–Wojtas ($2N-2=26=5^2+1^2$; achieved) | Yes (Local Search) |
| 14 ($N=15$) | 25,515 | 1 | Ehlich ($N \equiv 3 \pmod 4$; unachievable) | Yes |
| 15 ($N=16$) | 131,072 | 5 | Hadamard ($N \equiv 0 \pmod 4$; achieved) | Partial (3/5) |
| 16 ($N=17$) | 327,680 | 3 | Barba ($2N-1=33$ not a square; unachievable) | No |
| 17 ($N=18$) | 1,114,112 | 3 | Ehlich–Wojtas ($2N-2=34=5^2+3^2$; achieved) | No |
| 18 ($N=19$) | 3,411,968 | 3 | Ehlich ($N \equiv 3 \pmod 4$; unachievable) | No |
| 19 ($N=20$) | 19,531,250 | 3 | Hadamard ($N \equiv 0 \pmod 4$; achieved) | No |
| 20 ($N=21$) | 56,640,625 | 7 | Barba ($2N-1=41$ not a square; unachievable) | No |
| 21 ($N=22$) | 195,312,500 | 2 | Ehlich–Wojtas ($2N-2=42 \neq x^2+y^2$; unachievable) | No |
| 22 ($N=23$) | 662,671,875 | 1 | Ehlich ($N \equiv 3 \pmod 4$; unachievable) | No |

---

## 4. Key Findings & Limitations

* **Low Dimensions ($n \le 13$):** Standalone local search reliably finds $\{0, 1\}$-matrices attaining the known maximal determinant $D(n)$.
* **Medium Dimensions ($n = 14, 15$):** Full PatternBoost (local + Transformer global search, `nb_local_searches = 800`, `sample-only = 50000`) successfully reaches $D(14) = 25{,}515$ and $D(15) = 131{,}072$ (hitting the $n=15$ bound in Epoch 1).
* **High Dimensions & Mode Collapse ($n \ge 16$):**
  * **Manual Injection ($n = 16\text{–}19$):** Even with a known optimal matrix injected into training, the model failed to discover additional non-equivalent maximizers within 20 epochs (unique maximizer count stayed at $1$, whereas $3$ non-equivalent solutions are known).
  * **Double-Flip Recovery ($n = 16$ for 35 epochs; $n = 20\text{–}22$):** PatternBoost proved highly effective at *error correction*—consistently recovering the exact unperturbed upper-bound matrix within the first epoch. However, structural diversity remained limited to a single equivalence class (`unique count = 1`), indicating that the Transformer tends to remain trapped in the local basin of attraction of the seeded construction rather than exploring disjoint optimal basins.

---

## 5. Repository Structure

* `dim6/`, `8x8_run/`, `dim8_run/` – `dim11_run/`: Experimental configurations, local search outputs, and training logs for low-dimensional runs.
* `cycle_run/`: Iterative PatternBoost local-global search cycles.
* `gpu_run_output/`: Logs and outputs from NVIDIA GPU-accelerated Transformer training runs.
* `results/`: Decoded Transformer outputs (`transformer-output-decoded`), saved candidate matrices, and high-dimensional runs (e.g., `dim18_run_1`).
* `paper/`: Research documentation and LaTeX source files:
  * [Experimental Research Report](paper/Research_Report.pdf) — Empirical results, training visualizations, and double-flip experiments.
  * [Survey of Maximal Determinants](paper/Hadamard_Maximal_Determinants.pdf) — Analytical proofs of the Hadamard, Barba, Ehlich–Wojtas, and Ehlich bounds, plus the $n=8\text{–}22$ benchmark table.

---

## 6. References

1. P. Browne, R. Egan, F. Hegarty, and P. Ó Catháin, *A Survey of the Hadamard Maximal Determinant Problem*, The Electronic Journal of Combinatorics, 28(4): #P4.41, 2021.
2. J. Hadamard, *Résolution d'une question relative aux déterminants*, Bull. des Sciences Math., 17: 240–246, 1893.
3. H. Ehlich, *Determinantenabschätzungen für binäre Matrizen*, Mathematische Zeitschrift, 83: 123–132, 1964.
4. V. Chasiotis, S. Kounias, and N. Farmakis, *The D-optimal saturated designs of order 22*, Discrete Mathematics, 341: 380–387, 2018.
5. F. Charton, J. S. Ellenberg, A. Z. Wagner, and G. Williamson, *PatternBoost: Constructions in Mathematics with a Little Help from AI* (`zawagner22/transformers_math_experiments`).
