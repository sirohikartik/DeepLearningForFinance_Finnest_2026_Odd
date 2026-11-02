# Neural Network Mathematical Exercises
### Course: Deep Learning for Finance (Finnest 2026)
**Instructor:** Kartik Sirohi — Technical Lead at Finnest

---

## Overview

These four mathematical exercises are designed to bridge rigorous deep learning theory with quantitative finance. Each exercise emphasizes analytical derivation, step-by-step matrix/tensor calculus, and structural intuition.

---

## Table of Contents
1. [Exercise 1: Analytical Forward & Backpropagation for Credit Risk Classification](#exercise-1-analytical-forward--backpropagation-for-credit-risk-classification)
2. [Exercise 2: Recurrent Dynamics, Vanishing Gradients, and Spectral Norms in Financial Time-Series](#exercise-2-recurrent-dynamics-vanishing-gradients-and-spectral-norms-in-financial-time-series)
3. [Exercise 3: Loss Curvature, Hessian Spectrum, and Learning Rate Bounds in Factor Models](#exercise-3-loss-curvature-hessian-spectrum-and-learning-rate-bounds-in-factor-models)
4. [Exercise 4: Physics-Informed & Non-Arbitrage Neural Networks for Option Pricing Surfaces](#exercise-4-physics-informed--non-arbitrage-neural-networks-for-option-pricing-surfaces)

---

## Exercise 1: Analytical Forward & Backpropagation for Credit Risk Classification

### Context
In loan default prediction, we seek to predict whether a borrower will default on a credit facility ($y = 1$) or repay ($y = 0$). Given a small Multi-Layer Perceptron (MLP) with two financial input features, one hidden layer of 2 neurons with ReLU activation, and a scalar sigmoid output neuron trained under Binary Cross-Entropy (Log Loss).

### Model Specification
- **Input vector:** $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \in \mathbb{R}^2$, where:
  - $x_1$: Debt-to-Income (DTI) ratio normalized to $[0, 1]$.
  - $x_2$: Credit card revolving utilization rate $[0, 1]$.
- **Hidden layer (Layer 1):**
  $$z^{[1]} = W^{[1]} x + b^{[1]}, \quad a^{[1]} = \operatorname{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$
  with weight matrix $W^{[1]} \in \mathbb{R}^{2 \times 2}$ and bias $b^{[1]} \in \mathbb{R}^2$:
  $$W^{[1]} = \begin{pmatrix} 0.5 & -0.2 \\ 0.8 & 0.4 \end{pmatrix}, \quad b^{[1]} = \begin{pmatrix} 0.1 \\ -0.3 \end{pmatrix}$$
- **Output layer (Layer 2):**
  $$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}, \quad \hat{y} = a^{[2]} = \sigma(z^{[2]}) = \frac{1}{1 + e^{-z^{[2]}}}$$
  with weight vector $W^{[2]} \in \mathbb{R}^{1 \times 2}$ and scalar bias $b^{[2]} \in \mathbb{R}$:
  $$W^{[2]} = \begin{pmatrix} 1.2 & -0.9 \end{pmatrix}, \quad b^{[2]} = -0.1$$
- **Objective function:** Binary Cross-Entropy (BCE) for sample $(x, y)$:
  $$\mathcal{L}(y, \hat{y}) = -\left[ y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y}) \right]$$

### Questions
1. **Forward Pass Calculation:**
   Given an applicant profile $x = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}$ and true default outcome $y = 1$:
   - Calculate $z^{[1]}$ and $a^{[1]}$.
   - Calculate $z^{[2]}$ and the predicted default probability $\hat{y}$.
   - Compute the scalar loss value $\mathcal{L}(y, \hat{y})$ (to 4 decimal places).

2. **Derivative of Binary Cross-Entropy with respect to Pre-Activation:**
   Prove analytically using the chain rule that:
   $$\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \hat{y} - y$$

3. **Backpropagation Gradients:**
   Compute the exact numerical gradients:
   - $\frac{\partial \mathcal{L}}{\partial W^{[2]}}$ and $\frac{\partial \mathcal{L}}{\partial b^{[2]}}$
   - Backpropagated hidden error $\delta^{[1]} = \frac{\partial \mathcal{L}}{\partial z^{[1]}}$
   - $\frac{\partial \mathcal{L}}{\partial W^{[1]}}$ and $\frac{\partial \mathcal{L}}{\partial b^{[1]}}$

4. **Parameter Update:**
   Using stochastic gradient descent with learning rate $\eta = 0.1$:
   $$\theta \leftarrow \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$$
   Compute the updated parameter values $W^{[1]\prime}$, $b^{[1]\prime}$, $W^{[2]\prime}$, and $b^{[2]\prime}$.

---

### Solution & Step-by-Step Derivation

#### Part 1: Forward Pass
- **Hidden pre-activations:**
  $$z_1^{[1]} = (0.5)(0.6) + (-0.2)(0.8) + 0.1 = 0.30 - 0.16 + 0.10 = 0.24$$
  $$z_2^{[1]} = (0.8)(0.6) + (0.4)(0.8) + (-0.3) = 0.48 + 0.32 - 0.30 = 0.50$$
  $$z^{[1]} = \begin{pmatrix} 0.24 \\ 0.50 \end{pmatrix}$$
- **Hidden activations (ReLU):**
  $$a_1^{[1]} = \max(0, 0.24) = 0.24, \quad a_2^{[1]} = \max(0, 0.50) = 0.50 \implies a^{[1]} = \begin{pmatrix} 0.24 \\ 0.50 \end{pmatrix}$$
- **Output pre-activation:**
  $$z^{[2]} = (1.2)(0.24) + (-0.9)(0.50) + (-0.1) = 0.288 - 0.450 - 0.100 = -0.262$$
- **Predicted probability $\hat{y}$:**
  $$\hat{y} = \sigma(-0.262) = \frac{1}{1 + e^{0.262}} \approx \frac{1}{1 + 1.299526} \approx 0.434873 \approx 0.4349$$
- **Binary Cross-Entropy Loss (with $y=1$):**
  $$\mathcal{L} = -\ln(0.434873) \approx 0.8327$$

#### Part 2: Output Logit Gradient Derivation
Recall $\hat{y} = \sigma(z^{[2]})$ and $\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$.
Using the chain rule:
$$\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}}$$
$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$
$$\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \left( \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right) \cdot \left( \hat{y}(1 - \hat{y}) \right) = \hat{y} - y \quad \blacksquare$$

#### Part 3: Backpropagation Gradients
Let $\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} = \hat{y} - y = 0.434873 - 1 = -0.565127 \approx -0.5651$.
- **Gradients for Layer 2:**
  $$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]} (a^{[1]})^T = -0.565127 \times \begin{pmatrix} 0.24 & 0.50 \end{pmatrix} = \begin{pmatrix} -0.13563 & -0.28256 \end{pmatrix}$$
  $$\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]} = -0.565127$$
- **Hidden layer backpropagation:**
  $$\delta^{[1]} = \frac{\partial \mathcal{L}}{\partial z^{[1]}} = \left( (W^{[2]})^T \delta^{[2]} \right) \odot \sigma'(z^{[1]})$$
  Since $z_1^{[1]} = 0.24 > 0$ and $z_2^{[1]} = 0.50 > 0$, the ReLU derivatives are both $1$:
  $$\delta^{[1]} = \begin{pmatrix} 1.2 \\ -0.9 \end{pmatrix} (-0.565127) \odot \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} -0.678152 \\ 0.508614 \end{pmatrix}$$
- **Gradients for Layer 1:**
  $$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]} x^T = \begin{pmatrix} -0.678152 \\ 0.508614 \end{pmatrix} \begin{pmatrix} 0.6 & 0.8 \end{pmatrix} = \begin{pmatrix} -0.406891 & -0.542522 \\ 0.305168 & 0.406891 \end{pmatrix}$$
  $$\frac{\partial \mathcal{L}}{\partial b^{[1]}} = \delta^{[1]} = \begin{pmatrix} -0.678152 \\ 0.508614 \end{pmatrix}$$

#### Part 4: Parameter Updates ($\eta = 0.1$)
$$W^{[2]\prime} = \begin{pmatrix} 1.2 & -0.9 \end{pmatrix} - 0.1 \begin{pmatrix} -0.13563 & -0.28256 \end{pmatrix} = \begin{pmatrix} 1.2136 & -0.8717 \end{pmatrix}$$
$$b^{[2]\prime} = -0.1 - 0.1(-0.565127) = -0.0435$$
$$W^{[1]\prime} = \begin{pmatrix} 0.5 & -0.2 \\ 0.8 & 0.4 \end{pmatrix} - 0.1 \begin{pmatrix} -0.4069 & -0.5425 \\ 0.3052 & 0.4069 \end{pmatrix} = \begin{pmatrix} 0.5407 & -0.1458 \\ 0.7695 & 0.3593 \end{pmatrix}$$
$$b^{[1]\prime} = \begin{pmatrix} 0.1 \\ -0.3 \end{pmatrix} - 0.1 \begin{pmatrix} -0.6782 \\ 0.5086 \end{pmatrix} = \begin{pmatrix} 0.1678 \\ -0.3509 \end{pmatrix}$$

---

## Exercise 2: Recurrent Dynamics, Vanishing Gradients, and Spectral Norms in Financial Time-Series

### Context
Financial return series $x_1, x_2, \dots, x_T \in \mathbb{R}^d$ exhibit volatility clustering and multi-scale temporal dependencies. When modeling these sequences with a standard Recurrent Neural Network (Elman RNN), backpropagation through time (BPTT) frequently suffers from vanishing or exploding gradients across long market regimes.

### Model Specification
The recurrence equation is:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h), \quad t = 1, \dots, T$$
where $h_t \in \mathbb{R}^m$, $W_{hh} \in \mathbb{R}^{m \times m}$, $W_{xh} \in \mathbb{R}^{m \times d}$.
Let $\mathcal{L} = \ell(h_T)$ be a scalar loss computed at time $T$.

### Questions
1. **Derivation of Temporal Jacobian:**
   Show using the chain rule that the sensitivity of the final hidden state $h_T$ with respect to a past hidden state $h_t$ ($t < T$) is given by:
   $$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$
   Derive the explicit algebraic form of the one-step Jacobian $\frac{\partial h_k}{\partial h_{k-1}}$.

2. **Spectral Bound on Gradient Norm:**
   Let $\| \cdot \|_2$ denote the spectral norm (induced matrix 2-norm), where $\|A\|_2 = \sigma_{\max}(A)$ (the maximum singular value of $A$).
   Prove that:
   $$\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \left( \sigma_{\max}(W_{hh}) \right)^{T - t}$$
   State what happens when $\sigma_{\max}(W_{hh}) < 1$ vs $\sigma_{\min}(W_{hh}) > 1$ as the temporal horizon $(T - t) \to \infty$.

3. **Financial Interpretation:**
   Explain why an RNN with $\sigma_{\max}(W_{hh}) < 1$ fails to capture low-frequency macro signals (e.g., business cycle transitions over 252 trading days) while reacting exclusively to short-term market noise.

4. **LSTM Cell State Constant Error Carousel:**
   Recall the Long Short-Term Memory (LSTM) cell state update:
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
   Compute the partial derivative $\frac{\partial c_t}{\partial c_{t-1}}$.
   Explain mathematically why initializing the forget gate bias $b_f$ to a positive value (e.g., $b_f = 1$ or $2$) mitigates the vanishing gradient problem.

---

### Solution & Step-by-Step Derivation

#### Part 1: Temporal Jacobian
By the multivariable chain rule along the recurrent computation path:
$$\frac{\partial h_T}{\partial h_t} = \frac{\partial h_T}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdots \frac{\partial h_{t+1}}{\partial h_t} = \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}$$
Let $z_k = W_{hh} h_{k-1} + W_{xh} x_k + b_h$, so $h_k = \tanh(z_k)$.
Component-wise, $h_{k, i} = \tanh(z_{k, i})$, so:
$$\frac{\partial h_{k, i}}{\partial z_{k, j}} = \delta_{ij} \left(1 - \tanh^2(z_{k, i})\right)$$
Using matrix calculus:
$$\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\left(1 - h_k^2\right) W_{hh}^T$$
where $\operatorname{diag}(1 - h_k^2)$ is an $m \times m$ diagonal matrix with elements $1 - \tanh^2(z_{k, i}) \in (0, 1]$.

#### Part 2: Spectral Norm Bound
Using the sub-multiplicative property of induced matrix norms $\|A B\|_2 \le \|A\|_2 \|B\|_2$:
$$\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \prod_{k=t+1}^T \left\| \frac{\partial h_k}{\partial h_{k-1}} \right\|_2$$
For each step:
$$\left\| \frac{\partial h_k}{\partial h_{k-1}} \right\|_2 = \left\| \operatorname{diag}(1 - h_k^2) W_{hh}^T \right\|_2 \le \left\| \operatorname{diag}(1 - h_k^2) \right\|_2 \cdot \|W_{hh}^T\|_2$$
Since $\max_i |1 - \tanh^2(z_{k, i})| \le 1$ and $\|W_{hh}^T\|_2 = \|W_{hh}\|_2 = \sigma_{\max}(W_{hh})$:
$$\left\| \frac{\partial h_k}{\partial h_{k-1}} \right\|_2 \le 1 \cdot \sigma_{\max}(W_{hh})$$
Multiplying across all $T - t$ time steps:
$$\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \left( \sigma_{\max}(W_{hh}) \right)^{T - t} \quad \blacksquare$$

- **Consequences:**
  - If $\sigma_{\max}(W_{hh}) < 1$, then as $(T - t) \to \infty$, $\left(\sigma_{\max}(W_{hh})\right)^{T-t} \to 0$ exponentially. The gradient vanishes, rendering earlier inputs $x_t$ ineffective for updating weights.
  - If $\sigma_{\min}(W_{hh}) > 1$ and activations remain in the linear regime ($z_k \approx 0 \implies \tanh'(z_k) \approx 1$), gradients grow exponentially as $\sigma_{\min}^{T-t} \to \infty$, causing numeric overflow or explosive divergence.

#### Part 3: Financial Implication
In financial time series, price dynamics operate across heterogeneous frequencies: high-frequency fluctuations (minutes to days) and low-frequency macro regimes (months to years, such as interest rate cycles). When $\sigma_{\max}(W_{hh}) < 1$, gradient signals from day $T$ attenuate to machine epsilon before reaching $T - 60$ or $T - 252$. The model becomes incapable of attributing losses to long-term regime switches, biasing predictions exclusively toward recent market noise.

#### Part 4: LSTM Error Highway
For an LSTM cell:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
The total derivative with respect to $c_{t-1}$ is:
$$\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) + \left( \frac{\partial f_t}{\partial c_{t-1}} \odot c_{t-1} + \frac{\partial i_t}{\partial c_{t-1}} \odot \tilde{c}_t + \frac{\partial \tilde{c}_t}{\partial c_{t-1}} \odot i_t \right)$$
The dominant linear transmission term is $\operatorname{diag}(f_t)$.
When $b_f$ is initialized to $+1$ or $+2$, the sigmoid activation produces $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \approx 1$.
Consequently:
$$\frac{\partial c_T}{\partial c_t} \approx \prod_{k=t+1}^T \operatorname{diag}(f_k) \approx \prod_{k=t+1}^T I = I$$
This creates an uninterrupted additive gradient highway (the "Constant Error Carousel"), allowing gradient signals to propagate across hundreds of trading days without geometric decay.

---

## Exercise 3: Loss Curvature, Hessian Spectrum, and Learning Rate Bounds in Factor Models

### Context
Multi-factor asset pricing models predict cross-sectional excess asset returns $y \in \mathbb{R}^N$ from a design matrix of $d$ equity factors $X \in \mathbb{R}^{N \times d}$ (e.g., Value, Momentum, Quality, Low Beta, Market Capitalization). We train a linear neural layer with weight vector $w \in \mathbb{R}^d$ under Mean Squared Error loss with $L_2$ regularization (Ridge / Weight Decay).

### Model Specification
$$\mathcal{L}(w) = \frac{1}{2N} \| y - X w \|_2^2 + \frac{\lambda}{2} \| w \|_2^2$$
where $\lambda > 0$ is the regularization parameter, $X \in \mathbb{R}^{N \times d}$ has rank $r \le \min(N, d)$, and $y \in \mathbb{R}^N$.

### Questions
1. **Gradient & Hessian Derivation:**
   Compute the analytical gradient $\nabla_w \mathcal{L}(w)$ and the Hessian matrix $H = \nabla^2_w \mathcal{L}(w)$.

2. **Positive Definiteness & Convexity:**
   Prove that $H$ is strictly positive definite for any $\lambda > 0$, even if $X$ is rank-deficient (collinear financial factors). Explain why this guarantees a unique global minimum.

3. **Maximum Stable Learning Rate in Gradient Descent:**
   Standard gradient descent updates the weights according to:
   $$w^{(k+1)} = w^{(k)} - \eta \nabla \mathcal{L}(w^{(k)})$$
   Let $w^*$ be the unique global minimizer $\nabla \mathcal{L}(w^*) = 0$, and define the error vector $e^{(k)} = w^{(k)} - w^*$.
   - Express $e^{(k+1)}$ in terms of $e^{(k)}$ and $H$.
   - Prove that $\|e^{(k)}\|_2 \to 0$ as $k \to \infty$ for all initializations if and only if:
     $$0 < \eta < \frac{2}{\lambda_{\max}(H)}$$
     where $\lambda_{\max}(H)$ is the maximum eigenvalue of the Hessian.

4. **Optimal Learning Rate and Factor Collinearity:**
   Show that the optimal learning rate $\eta^*$ that maximizes the convergence rate is:
   $$\eta^* = \frac{2}{\lambda_{\max}(H) + \lambda_{\min}(H)}$$
   and show that the optimal convergence factor is $\frac{\kappa - 1}{\kappa + 1}$, where $\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$ is the condition number. How does $L_2$ weight decay $\lambda$ improve the conditioning of multi-collinear financial datasets?

---

### Solution & Step-by-Step Derivation

#### Part 1: Gradient and Hessian
Expand the objective:
$$\mathcal{L}(w) = \frac{1}{2N} \left( y^T y - 2 y^T X w + w^T X^T X w \right) + \frac{\lambda}{2} w^T w$$
Taking the first derivative with respect to $w$:
$$\nabla_w \mathcal{L}(w) = \frac{1}{N} \left( X^T X w - X^T y \right) + \lambda w = \left( \frac{1}{N} X^T X + \lambda I \right) w - \frac{1}{N} X^T y$$
Taking the second derivative:
$$H = \nabla^2_w \mathcal{L}(w) = \frac{1}{N} X^T X + \lambda I \quad \blacksquare$$

#### Part 2: Positive Definiteness
Let $v \in \mathbb{R}^d$ be any non-zero vector ($v \ne 0$). The quadratic form is:
$$v^T H v = v^T \left( \frac{1}{N} X^T X + \lambda I \right) v = \frac{1}{N} (X v)^T (X v) + \lambda v^T v = \frac{1}{N} \|X v\|_2^2 + \lambda \|v\|_2^2$$
Since $\|X v\|_2^2 \ge 0$ and $\|v\|_2^2 > 0$ for all $v \ne 0$:
$$v^T H v \ge 0 + \lambda \|v\|_2^2 = \lambda \|v\|_2^2 > 0 \quad (\text{since } \lambda > 0)$$
Thus, all eigenvalues satisfy $\lambda_i(H) \ge \lambda > 0$. $H$ is strictly positive definite everywhere, which establishes that $\mathcal{L}(w)$ is strictly convex with a unique global minimum at:
$$w^* = \left( X^T X + N \lambda I \right)^{-1} X^T y$$

#### Part 3: Maximum Stable Learning Rate
At the optimum, $\nabla \mathcal{L}(w^*) = H w^* - \frac{1}{N} X^T y = 0$.
The gradient at any point $w^{(k)}$ can be written as:
$$\nabla \mathcal{L}(w^{(k)}) = H w^{(k)} - \frac{1}{N} X^T y = H (w^{(k)} - w^*)$$
Substitute into the gradient descent update:
$$w^{(k+1)} = w^{(k)} - \eta H (w^{(k)} - w^*)$$
Subtract $w^*$ from both sides:
$$e^{(k+1)} = w^{(k+1)} - w^* = (I - \eta H) (w^{(k)} - w^*) = (I - \eta H) e^{(k)}$$
By induction, $e^{(k)} = (I - \eta H)^k e^{(0)}$.
Since $H$ is symmetric real, it has an orthogonal eigen-decomposition $H = Q \Lambda Q^T$. Then:
$$I - \eta H = Q (I - \eta \Lambda) Q^T$$
The error converges to zero for all $e^{(0)}$ if and only if the spectral radius $\rho(I - \eta H) < 1$:
$$\max_{i} |1 - \eta \lambda_i(H)| < 1$$
$$-1 < 1 - \eta \lambda_i(H) < 1 \implies 0 < \eta \lambda_i(H) < 2 \implies 0 < \eta < \frac{2}{\lambda_i(H)} \quad \forall i$$
Since this must hold for all eigenvalues, the tightest upper bound is determined by $\lambda_{\max}(H)$:
$$0 < \eta < \frac{2}{\lambda_{\max}(H)} \quad \blacksquare$$

#### Part 4: Optimal Learning Rate and Conditioning
The worst-case contraction factor per step is:
$$\rho(\eta) = \max \left\{ |1 - \eta \lambda_{\min}(H)|, |1 - \eta \lambda_{\max}(H)| \right\}$$
The function $|1 - \eta \lambda_{\min}|$ decreases with $\eta$, while $|1 - \eta \lambda_{\max}|$ increases once $\eta > 1/\lambda_{\max}$. The minimax optimum occurs at their point of intersection:
$$1 - \eta^* \lambda_{\min}(H) = -(1 - \eta^* \lambda_{\max}(H))$$
$$2 = \eta^* \left( \lambda_{\max}(H) + \lambda_{\min}(H) \right) \implies \eta^* = \frac{2}{\lambda_{\max}(H) + \lambda_{\min}(H)} \quad \blacksquare$$
Plugging $\eta^*$ back into the contraction factor:
$$\rho^* = 1 - \frac{2 \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1}$$
where $\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$.

- **Financial interpretation of $L_2$ regularization:**
  In multi-factor equity modeling, factors are frequently co-linear (e.g., Book-to-Price, Earnings-to-Price, and Cash-Flow-to-Price). Without regularization ($\lambda = 0$), the sample covariance matrix $\frac{1}{N} X^T X$ has $\lambda_{\min} \approx 0$, driving the condition number $\kappa \to \infty$ and $\rho^* \approx 1 - \frac{2}{\kappa} \to 1$. Gradient descent stalls or exhibits violent oscillations across the narrow ravine. Adding $\lambda I$ shifts every eigenvalue uniformly: $\lambda_i(H) = \lambda_i(\frac{1}{N} X^T X) + \lambda$.
  The regularized condition number becomes:
  $$\kappa_{\text{reg}} = \frac{\lambda_{\max}(\Sigma) + \lambda}{\lambda_{\min}(\Sigma) + \lambda} \ll \kappa_{\text{unreg}}$$
  which drastically rounds out the loss landscape, restoring rapid and stable convergence.

---

## Exercise 4: Physics-Informed & Non-Arbitrage Neural Networks for Option Pricing Surfaces

### Context
When pricing European call options $C(S, K, \tau)$ via deep neural networks, a standard unconstrained network may yield arbitrage violations: negative probabilities, negative option prices, or calendar spreads yielding free money. Under Black-Scholes and general no-arbitrage theory:
1. **Non-negative price:** $C(S, K, \tau) \ge \max(S - K e^{-r\tau}, 0) \ge 0$.
2. **Monotonicity in spot:** $\Delta = \frac{\partial C}{\partial S} \ge 0$.
3. **Convexity in strike (Breeden-Litzenberger):** The risk-neutral probability density of the stock price at expiry $S_T$ is given by:
   $$q(K) = e^{r\tau} \frac{\partial^2 C}{\partial K^2} \ge 0 \implies \frac{\partial^2 C}{\partial K^2} \ge 0$$
4. **Black-Scholes PDE Residual:**
   $$\mathcal{N}[C] = \frac{\partial C}{\partial \tau} - \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} - r S \frac{\partial C}{\partial S} + r C = 0$$

### Model Specification
Consider a single-hidden-layer neural pricing network $\hat{C}_\theta(S, K, \tau)$:
$$\hat{C}_\theta(S, K, \tau) = \sum_{j=1}^m v_j g(w_j S + u_j K + p_j \tau + b_j) + b_0$$
where $g(z) = \ln(1 + e^z)$ is the Softplus activation function, and $v_j, w_j, u_j, p_j, b_j, b_0$ are parameters.

### Questions
1. **Mathematical Proof of Arbitrage-Free Strike Convexity:**
   Suppose we constrain $v_j \ge 0$ for all $j \in \{1, \dots, m\}$. Prove mathematically that $\frac{\partial^2 \hat{C}_\theta}{\partial K^2} \ge 0$ for all inputs, ensuring the model never generates negative implied probabilities.

2. **Analytical Derivation of Option Greeks:**
   Using the properties of the Softplus function $g(z)$ and logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$:
   - Prove that $g'(z) = \sigma(z)$ and $g''(z) = \sigma(z)(1 - \sigma(z))$.
   - Derive analytical closed-form expressions for option Delta $\hat{\Delta} = \frac{\partial \hat{C}_\theta}{\partial S}$ and Gamma $\hat{\Gamma} = \frac{\partial^2 \hat{C}_\theta}{\partial S^2}$.
   - Show that if $v_j \ge 0$ and $w_j \ge 0$, then $\hat{\Delta} \ge 0$ and $\hat{\Gamma} \ge 0$.

3. **Physics-Informed Loss Formulation:**
   Formulate a Physics-Informed Neural Network (PINN) loss function that combines market price observation fitting with Black-Scholes PDE residual penalization over a collocation grid of $M$ domain points $\{(S_k, K_k, \tau_k)\}_{k=1}^M$.
   Write the explicit loss $\mathcal{L}_{\text{PINN}}(\theta)$.

4. **PDE Residual Derivative Calculation:**
   Given $m=1$ with parameters $v_1 = 1.0, w_1 = 0.5, u_1 = -0.5, p_1 = 0.2, b_1 = 0, b_0 = 0$, spot $S = 100$, strike $K = 100$, time-to-maturity $\tau = 1$, volatility $\sigma = 0.20$, and risk-free rate $r = 0.05$:
   - Calculate the argument $z_1 = w_1 S + u_1 K + p_1 \tau + b_1$.
   - Compute $\hat{C}$, $\frac{\partial \hat{C}}{\partial \tau}$, $\frac{\partial \hat{C}}{\partial S}$, and $\frac{\partial^2 \hat{C}}{\partial S^2}$.
   - Compute the scalar Black-Scholes PDE residual $\mathcal{N}[\hat{C}]$.

---

### Solution & Step-by-Step Derivation

#### Part 1: Arbitrage-Free Convexity in Strike
Let $z_j = w_j S + u_j K + p_j \tau + b_j$.
Differentiating $\hat{C}_\theta$ with respect to strike $K$:
$$\frac{\partial \hat{C}_\theta}{\partial K} = \sum_{j=1}^m v_j \frac{\partial g(z_j)}{\partial K} = \sum_{j=1}^m v_j u_j g'(z_j)$$
Differentiating a second time:
$$\frac{\partial^2 \hat{C}_\theta}{\partial K^2} = \sum_{j=1}^m v_j u_j^2 g''(z_j)$$
For the Softplus activation $g(z) = \ln(1 + e^z)$:
$$g'(z) = \frac{e^z}{1 + e^z} = \sigma(z)$$
$$g''(z) = \frac{d}{dz} \sigma(z) = \sigma(z)(1 - \sigma(z))$$
Since $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$, $g''(z) = \sigma(z)(1 - \sigma(z)) > 0$.
Furthermore, $u_j^2 \ge 0$ for all $u_j \in \mathbb{R}$.
Therefore, if $v_j \ge 0$:
$$v_j u_j^2 g''(z_j) \ge 0 \implies \frac{\partial^2 \hat{C}_\theta}{\partial K^2} = \sum_{j=1}^m v_j u_j^2 g''(z_j) \ge 0 \quad \forall K \in \mathbb{R} \quad \blacksquare$$
By the Breeden-Litzenberger theorem, the risk-neutral state price density is:
$$q(K) = e^{r\tau} \frac{\partial^2 \hat{C}_\theta}{\partial K^2} \ge 0$$
Hence, non-negative outer weights $v_j \ge 0$ strictly eliminate negative state-price arbitrage.

#### Part 2: Option Greeks
- **Derivatives of Softplus:**
  $$g'(z) = \frac{d}{dz} \ln(1 + e^z) = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} = \sigma(z)$$
  $$g''(z) = \frac{d}{dz} \left(1 + e^{-z}\right)^{-1} = - (1 + e^{-z})^{-2} (-e^{-z}) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$$
- **Delta ($\hat{\Delta}$):**
  $$\hat{\Delta} = \frac{\partial \hat{C}_\theta}{\partial S} = \sum_{j=1}^m v_j w_j g'(z_j) = \sum_{j=1}^m v_j w_j \sigma(z_j)$$
- **Gamma ($\hat{\Gamma}$):**
  $$\hat{\Gamma} = \frac{\partial^2 \hat{C}_\theta}{\partial S^2} = \sum_{j=1}^m v_j w_j^2 g''(z_j) = \sum_{j=1}^m v_j w_j^2 \sigma(z_j)(1 - \sigma(z_j))$$
- **Sign constraints:**
  Since $\sigma(z) > 0$ and $1 - \sigma(z) > 0$:
  - If $v_j \ge 0$ and $w_j \ge 0$, every summand $v_j w_j \sigma(z_j) \ge 0$, ensuring $\hat{\Delta} \ge 0$ (monotonicity with underlying asset price).
  - Since $w_j^2 \ge 0$, $v_j \ge 0$ guarantees $\hat{\Gamma} \ge 0$ (long option position has non-negative gamma).

#### Part 3: Physics-Informed Neural Network (PINN) Loss
Given $N$ observed market prices $\{(S_i, K_i, \tau_i, C_i^{\text{mkt}})\}_{i=1}^N$ and $M$ unsupervised PDE collocation points $\{(S_k, K_k, \tau_k)\}_{k=1}^M$:
$$\mathcal{L}_{\text{PINN}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}}(\theta) + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}}(\theta)$$
where:
$$\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( \hat{C}_\theta(S_i, K_i, \tau_i) - C_i^{\text{mkt}} \right)^2$$
$$\mathcal{L}_{\text{PDE}}(\theta) = \frac{1}{M} \sum_{k=1}^M \left( \frac{\partial \hat{C}_\theta}{\partial \tau_k} - \frac{1}{2} \sigma^2 S_k^2 \frac{\partial^2 \hat{C}_\theta}{\partial S_k^2} - r S_k \frac{\partial \hat{C}_\theta}{\partial S_k} + r \hat{C}_\theta \right)^2$$
and $\mathcal{L}_{\text{shape}}$ optionally penalizes boundary violations (such as $\hat{C} < \max(S - K, 0)$).

#### Part 4: Numerical Calculation
- **Input parameters:**
  $$S = 100, \quad K = 100, \quad \tau = 1, \quad \sigma = 0.20, \quad r = 0.05$$
  $$v_1 = 1.0, \quad w_1 = 0.5, \quad u_1 = -0.5, \quad p_1 = 0.2, \quad b_1 = 0, \quad b_0 = 0$$
- **Argument $z_1$:**
  $$z_1 = 0.5(100) + (-0.5)(100) + 0.2(1) + 0 = 50 - 50 + 0.2 = 0.2$$
- **Softplus and Sigmoid evaluations at $z_1 = 0.2$:**
  $$e^{0.2} \approx 1.221403$$
  $$\hat{C} = g(0.2) = \ln(1 + e^{0.2}) = \ln(2.221403) \approx 0.798138$$
  $$\sigma(0.2) = \frac{1}{1 + e^{-0.2}} = \frac{1}{1 + 0.818731} \approx 0.549834$$
  $$g''(0.2) = \sigma(0.2)(1 - \sigma(0.2)) = (0.549834)(0.450166) \approx 0.247517$$
- **Partial Derivatives:**
  $$\frac{\partial \hat{C}}{\partial \tau} = v_1 p_1 \sigma(z_1) = (1.0)(0.2)(0.549834) = 0.109967$$
  $$\frac{\partial \hat{C}}{\partial S} = v_1 w_1 \sigma(z_1) = (1.0)(0.5)(0.549834) = 0.274917$$
  $$\frac{\partial^2 \hat{C}}{\partial S^2} = v_1 w_1^2 g''(z_1) = (1.0)(0.25)(0.247517) = 0.061879$$
- **Black-Scholes PDE Terms:**
  1. Time decay: $\frac{\partial \hat{C}}{\partial \tau} = 0.109967$
  2. Diffusion term: $\frac{1}{2} \sigma^2 S^2 \frac{\partial^2 \hat{C}}{\partial S^2} = \frac{1}{2} (0.04)(10000)(0.061879) = 200 \times 0.061879 = 12.3758$
  3. Drift term: $r S \frac{\partial \hat{C}}{\partial S} = (0.05)(100)(0.274917) = 5 \times 0.274917 = 1.374585$
  4. Discount term: $r \hat{C} = (0.05)(0.798138) = 0.039907$
- **Total PDE Residual:**
  $$\mathcal{N}[\hat{C}] = 0.109967 - 12.3758 - 1.374585 + 0.039907 = -13.6005$$
  The PDE residual $\mathcal{N}[\hat{C}] \approx -13.6005 \ne 0$ reflects that this untrained single neuron does not yet satisfy Black-Scholes dynamics. Minimizing $\mathcal{L}_{\text{PINN}}$ will adjust $(v_1, w_1, u_1, p_1, b_1)$ until $\mathcal{N}[\hat{C}] \approx 0$.
