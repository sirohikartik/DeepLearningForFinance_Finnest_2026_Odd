# Deep Learning For Finance

**Instructor:** Kartik Sirohi  
**Role:** Technical Lead at Finnest  

---

## Course Overview

Deep Learning for Finance is a rigorous course designed to bridge fundamental neural network mathematics with production-grade quantitative finance applications. Students learn both the analytical foundations (backpropagation, matrix calculus, optimization geometry, loss surface analysis) and applied implementations (fraud detection, credit scoring, time-series forecasting, volatility modeling, and physics-informed non-arbitrage derivative pricing).

---

## Syllabus

### 01 — Fundamentals
- **Machine Learning & Deep Learning Foundations**: Representation learning, inductive bias, and empirical risk minimization.
- **Neural Network Architecture**: Perceptrons, Multi-Layer Perceptrons (MLPs), computation graphs, and matrix formulations.
- **Weights, Biases & Activation Functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Softplus, and GELU.
- **Forward Propagation & Loss Functions**: Mean Squared Error (MSE), Binary Cross-Entropy (BCE), Negative Log-Likelihood, and financial utility losses.
- **Gradients, Backpropagation & Optimization**: Automatic differentiation, chain rule, Jacobian-vector products, Stochastic Gradient Descent (SGD), Momentum, and Adam.

### 02 — Applications in Finance
- **Fraud Detection**: Extreme class imbalance, autoencoders, anomaly detection, and cost-sensitive loss functions.
- **Credit Risk & Loan Default Prediction**: Calibrated probabilities, ROC-AUC, expected loss modeling ($EL = PD \times LGD \times EAD$), and feature attribution.
- **Financial Time-Series Forecasting**: Return predictability, volatility clustering, non-stationarity, and regime switching.
- **Sequential Architectures (LSTMs & Transformers)**: Recurrent neural dynamics, Constant Error Carousel, multi-head self-attention, and temporal fusion.
- **Building Blocks of a Financial AI System**: Data pipelines, point-in-time backtesting, latency constraints, and risk controls.

---

## Core Resources & Reading List

### 1. Andrej Karpathy — Deep Learning from Foundations to LLMs
- **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**  
  A masterclass in deep learning fundamentals starting from first principles:
  - [micrograd](https://github.com/karpathy/micrograd): Building an autograd engine and neural network backpropagation from scratch.
  - [makemore](https://github.com/karpathy/makemore): Autoregressive character-level language modeling (Bigram, MLP, BatchNorm, WaveNet).
  - [Building nanoGPT](https://github.com/karpathy/nanoGPT): Complete from-scratch implementation and training of modern Transformers.
  - [YouTube Playlist: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **[Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/)**  
  Originally designed and instructed by Andrej Karpathy. The course notes provide one of the clearest mathematical introductions to backpropagation, vector calculus, and optimization algorithms available anywhere.  
  - [CS231n Course Notes](https://cs231n.github.io/optimization-2/)
- **Essential Articles by Karpathy**:
  - [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
  - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### 2. Andrew Ng — Stanford Deep Learning & Machine Learning
- **[Stanford CS230: Deep Learning](https://cs230.stanford.edu/)**  
  Taught by Andrew Ng and Kian Katanforoosh at Stanford University. Focuses on deep learning architectures, hyperparameter search, error analysis, and building end-to-end AI applications.
  - [Stanford CS230 Lecture Series (YouTube)](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)
  - [CS230 Cheatsheets](https://stanford.edu/~shervine/teaching/cs-230/)
- **[Deep Learning Specialization (Coursera / DeepLearning.AI)](https://www.deeplearning.ai/courses/deep-learning-specialization/)**  
  The world-renowned 5-course sequence instructed by Andrew Ng:
  1. *Neural Networks and Deep Learning*
  2. *Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization*
  3. *Structuring Machine Learning Projects*
  4. *Convolutional Neural Networks*
  5. *Sequence Models (RNNs, LSTMs, Transformers)*
- **[Stanford CS229: Machine Learning](https://cs229.stanford.edu/)**  
  Foundational mathematical theory of machine learning, statistical learning theory, optimization, and generalized linear models.

### 3. The Theory of Deep Learning Book & Theoretical Foundations
- **[The Principles of Deep Learning Theory: An Effective Field Theory Approach to Understanding Neural Networks](https://deeplearningtheory.com/)**  
  *Daniel A. Roberts, Sho Yaida, and Boris Hanin* (Cambridge University Press / arXiv:2106.10165)  
  - The definitive mathematical treatise developing an exact microscopic theory of deep neural networks using effective field theory from theoretical physics.
  - Explores signal propagation at infinite width, critical initialization, renormalization group flow through depth, and finite-width corrections ($1/N$ expansion) explaining why representation learning occurs in deep networks.
  - Free open-access preprint: [arXiv:2106.10165 [cs.LG]](https://arxiv.org/abs/2106.10165)
- **Additional Theoretical Texts**:
  - *Deep Learning* — Ian Goodfellow, Yoshua Bengio, and Aaron Courville ([MIT Press](https://www.deeplearningbook.org/))
  - *Understanding Deep Learning* — Simon J.D. Prince ([MIT Press](https://udlbook.github.io/udlbook/))
  - *Mathematical Introduction to Deep Learning* — Arnulf Jentzen, Benno Kuckuck, Philippe von Wurstemberger ([arXiv:2012.04516](https://arxiv.org/abs/2012.04516))

---

## Neural Network Mathematical Exercises

For complete derivations and step-by-step solutions, see the [EXERCISES.md](EXERCISES.md) file.

### [Exercise 1: Analytical Forward & Backpropagation for Credit Risk Classification](EXERCISES.md#exercise-1-analytical-forward--backpropagation-for-credit-risk-classification)
- **Domain:** Credit default risk prediction ($y \in \{0, 1\}$).
- **Architecture:** 2 inputs (Debt-to-Income, Credit Utilization), 2 hidden ReLU neurons, 1 Sigmoid output neuron.
- **Core Math:** Forward pass computation, proof that $\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \hat{y} - y$ under Binary Cross-Entropy, exact hidden error propagation $\delta^{[1]} = \left((W^{[2]})^T \delta^{[2]}\right) \odot \mathbf{1}_{z^{[1]}>0}$, parameter gradient calculation, and SGD weight update.

### [Exercise 2: Recurrent Dynamics, Vanishing Gradients, and Spectral Norms in Financial Time-Series](EXERCISES.md#exercise-2-recurrent-dynamics-vanishing-gradients-and-spectral-norms-in-financial-time-series)
- **Domain:** Macro regime and volatility modeling across trading horizons.
- **Architecture:** Recurrent Neural Network (Elman RNN) vs. Long Short-Term Memory (LSTM).
- **Core Math:** Derivation of multi-step Jacobian $\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \operatorname{diag}(1 - h_k^2) W_{hh}^T$, proof of spectral upper bound $\left\|\frac{\partial h_T}{\partial h_t}\right\|_2 \le (\sigma_{\max}(W_{hh}))^{T-t}$, analysis of vanishing gradients on long-memory financial features, and mathematical proof of the LSTM additive Constant Error Carousel $\frac{\partial c_T}{\partial c_t} \approx I$.

### [Exercise 3: Loss Curvature, Hessian Spectrum, and Learning Rate Bounds in Factor Models](EXERCISES.md#exercise-3-loss-curvature-hessian-spectrum-and-learning-rate-bounds-in-factor-models)
- **Domain:** Multi-factor equity return forecasting with collinear predictors.
- **Architecture:** Regularized linear/neural layer under $L_2$ weight decay (Ridge / Tikhonov).
- **Core Math:** Exact Hessian derivation $H = \frac{1}{N} X^T X + \lambda I$, proof of strict positive definiteness for $\lambda > 0$, proof of maximum stable learning rate condition $0 < \eta < \frac{2}{\lambda_{\max}(H)}$, derivation of optimal step size $\eta^* = \frac{2}{\lambda_{\max}(H) + \lambda_{\min}(H)}$, and condition number analysis $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ in financial factor covariance matrices.

### [Exercise 4: Physics-Informed & Non-Arbitrage Neural Networks for Option Pricing Surfaces](EXERCISES.md#exercise-4-physics-informed--non-arbitrage-neural-networks-for-option-pricing-surfaces)
- **Domain:** European option pricing surface $C(S, K, \tau)$ satisfying no-static-arbitrage constraints and Black-Scholes PDE.
- **Architecture:** Shape-constrained neural network with Softplus activation $g(z) = \ln(1 + e^z)$.
- **Core Math:** Proof that non-negative weights $v_j \ge 0$ enforce strike convexity $\frac{\partial^2 C}{\partial K^2} \ge 0$ (Breeden-Litzenberger state-price density non-negativity), analytical derivation of option Greeks (Delta $\hat{\Delta}$ and Gamma $\hat{\Gamma}$), formulation of PINN loss with Black-Scholes PDE residual $\mathcal{N}[C] = \frac{\partial C}{\partial \tau} - \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} - r S \frac{\partial C}{\partial S} + r C$, and concrete numerical evaluation.
