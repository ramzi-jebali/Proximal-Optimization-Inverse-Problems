# Convex Optimization for High-Dimensional Inverse Problems: Signal Recovery & Deblurring

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)
![Optimization](https://img.shields.io/badge/Math-Convex_Optimization-success)
![Academic](https://img.shields.io/badge/Course-ENSTA_Paris_4OPT2-purple)

## 📌 Overview
This repository contains the numerical implementation of scalable first-order optimization algorithms to solve ill-posed inverse problems, developed as part of the *Continuous Optimisation (4OPT2)* course at ENSTA Paris / Institut Polytechnique de Paris. While visually applied to image deblurring, the mathematical core—recovering a sparse signal from noisy, convoluted observations using Total Variation (TV) regularization—is a fundamental framework in quantitative finance. These exact proximal methods are heavily utilized in systematic trading for microstructure noise filtering, order book denoising, and covariance matrix regularization.

## 📐 Mathematical Framework
We address the inverse problem of reconstructing an unknown, high-dimensional true signal $x \in \mathbb{R}^n$ from a degraded and noisy observation $y \in \mathbb{R}^m$:

$$y = A x + \epsilon$$

Where $A$ represents a linear degradation operator (e.g., convolution/blur) and $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian noise. To recover $x$, we solve the following non-smooth convex optimization problem using Tikhonov/Total Variation regularization:

$$\min_{x \in \mathbb{R}^n} \underbrace{\frac{1}{2} \|A x - y\|_2^2}_{f(x)} + \underbrace{\lambda \|D x\|_1}_{g(x)}$$

- **$f(x)$**: The smooth data-fidelity term. Its gradient is Lipschitz continuous, computed efficiently as $\nabla f(x) = A^\top(Ax - y)$.
- **$g(x)$**: The non-smooth regularization term, where $D$ is the discrete gradient operator. This enforces sparsity in the signal's variations (Isotropic Total Variation). It is handled implicitly via its **Proximal Operator** ($\text{prox}_{\gamma g}$).

## ⚙️ Algorithms & Implementation
To ensure scalability to $10^5+$ variables without memory overflow, black-box solvers were avoided. The optimization routines were fully implemented **from scratch**:

* **Forward-Backward Splitting (ISTA):** Implemented to alternate between explicit gradient descent steps on the smooth data-fidelity term and implicit proximal steps on the TV regularization, achieving an $\mathcal{O}(1/k)$ convergence rate.
* **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm):** An accelerated version leveraging Nesterov's momentum to improve the theoretical convergence rate to **$\mathcal{O}(1/k^2)$**. 
* **Douglas-Rachford Splitting:** An advanced operator splitting method deployed to handle the sum of the convex functions. While computationally heavier per iteration, it is unconditionally stable and highly robust for ill-conditioned inverse problems.
* **Sparse Matrix Operations:** The degradation operator $A$ and finite difference operator $D$ are constructed using `scipy.sparse`. All gradient computations are strictly **matrix-free** (relying on sparse matrix-vector multiplications) to avoid intractable dense matrix inversions.

## 📊 Results & Performance
The algorithmic efficiency was benchmarked across 48 configurations (varying blur kernels, noise levels $\epsilon$, and parameters). The table below reports the **average execution time** for converging on the high-dimensional image grids.

| Metric | Forward-Backward (ISTA) | FISTA | Douglas-Rachford |
| :--- | :--- | :--- | :--- |
| **Convergence Rate** | $\mathcal{O}(1/k)$ | $\mathcal{O}(1/k^2)$ | $\mathcal{O}(1/k)$ |
| **Average Execution Time** | **~41.03 s** | **~41.54 s** | ~102.11 s |

> **Performance Note:** While FISTA takes slightly longer per iteration (and thus has a similar overall time to ISTA), its Nesterov acceleration guarantees a strictly superior objective decay ($\mathcal{O}(1/k^2)$ vs $\mathcal{O}(1/k)$), making it the optimal choice for reaching high-precision stopping criteria.

<p align="center">
  <img src="figures/convergence_plot.png" width="45%" alt="Convergence Plot O(1/k^2)">
  <img src="figures/deblurring_results.png" width="45%" alt="Signal Recovery Results">
</p>
<p align="center"><i>Left: Empirical convergence matching theoretical rates. Right: Restored signal from corrupted input.</i></p>

## 🛠️ Tech Stack
* **Core:** Python 3.10
* **Mathematics & Matrices:** `numpy`, `scipy.sparse` (CSR/CSC formats), `scipy.sparse.linalg`
* **Visualization:** `matplotlib`
