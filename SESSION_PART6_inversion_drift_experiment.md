# Session Part 6 — Inversion-Drift Diagnostic Experiment

*Saved: 2026-05-11*

## Goal

Measure how much a single layer of backward target-propagation distorts the activation distribution, comparing two inversion methods:

- **Method N (Naive)**: Moore-Penrose pseudo-inverse of the linear map, plus a simple non-linearity inverse (forward-mask preservation for ReLU).
- **Method K (K-FAC-A regularized)**: Gaussian posterior inversion with empirical activation covariance Σₐ (= K-FAC A factor) as the prior, plus the same non-linearity inverse.

The diagnostic isolates *inversion fidelity* from training dynamics. No OLS solve, no weight update, no end-to-end training. Just: given a known-good target at layer L (taken from the actual forward pass), how well does back-propagation to layer L−1 recover the actual layer-(L−1) activations?

## Why this is the right next experiment

1. **Decouples the failure modes.** The retrainer's accuracy gap to LoRA could come from (a) inversion drift accumulating bad targets, (b) the OLS solver being insufficient given good targets, or (c) compounding errors across layers. This experiment isolates (a) at single-layer granularity. If naive and K-FAC-A inversions produce similar drift, K-FAC-TP doesn't gain anything from the prior and we revise the approach. If K-FAC-A is meaningfully better, the framework has empirical support before any retraining benchmark runs.

2. **Cheap to run.** No training loop, no convergence question, no hyperparameter sweep. Single forward pass + per-layer inversion + distribution comparison. Probably hours, not days.

3. **Generalizes cleanly.** The same diagnostic runs on MLPs, transformer blocks, full BERT. Pick the cheapest viable substrate first.

## Setup

### Network choices (phased)

**Phase 1 — MLP on MNIST** (fastest iteration):
- 4-layer MLP: 784 → 512 → 256 → 128 → 10, ReLU between hidden layers
- Train to ~98% accuracy with standard SGD/Adam (15-30 min)
- Three retrainable hidden layers to test per-layer drift

**Phase 2 — Single transformer block**:
- Use one BERT transformer block in isolation, fed by frozen embedding output
- Tests attention + LayerNorm + GELU handling
- Reuses your BERT-SST2 data pipeline

**Phase 3 — Full BERT-base last 4 layers**:
- The actual target deployment context
- Connects directly to your existing N=4 retrainer benchmark

Do Phase 1 end-to-end before starting Phase 2. The methodology decisions you make in Phase 1 will need refinement for transformer-specific layers, and doing them in the simpler setting first saves debugging time.

### Data

- 4096 samples from MNIST (Phase 1) or SST-2 train set (Phase 2-3)
- Single forward pass with hooks to capture activations at every layer of interest

## The single-step drift measurement

For a target layer L with input dimension $d_{in}$, output dimension $d_{out}$, linear weight $W$, bias $b$, and activation function $f$:

**Ground truth from forward pass:**
- $a^*_{L-1}$ = actual layer-(L−1) output activations (the input to layer L), shape: (n_samples, $d_{in}$)
- $a^*_L$ = actual layer-L output activations after the non-linearity, shape: (n_samples, $d_{out}$)

**The setup:** treat $a^*_L$ as a "perfect target" — it's known to be a real, achievable activation distribution because it came from real data. Back-propagate it to produce $\hat a_{L-1}$. Compare $\hat a_{L-1}$ to $a^*_{L-1}$.

### Method N (Naive)

Pre-activation target: $t_{\text{pre}} = f^{-1}(a^*_L)$ where for ReLU:
- $t_{\text{pre}}[i] = a^*_L[i]$ where $a^*_L[i] > 0$
- $t_{\text{pre}}[i] = a_{\text{pre-forward}}[i]$ where $a^*_L[i] = 0$ (forward-mask preservation; alternative: zero)

Input target via pseudo-inverse: $\hat a_{L-1} = W^+ (t_{\text{pre}} - b)$ with damped pinv: $W^+ = (W^\top W + \lambda I)^{-1} W^\top$ for $d_{in} < d_{out}$, $W^+ = W^\top (W W^\top + \lambda I)^{-1}$ for $d_{in} > d_{out}$, regularizer $\lambda$ small (1e-4 of mean singular value squared).

### Method K (K-FAC-A regularized)

Same $t_{\text{pre}}$ from the non-linearity inverse.

Compute $\mu_a = \text{mean}(a^*_{L-1})$ and $\Sigma_a = \text{cov}(a^*_{L-1})$ over the batch (the K-FAC A factor).

Kalman posterior mean:
$$
\hat a_{L-1} = \mu_a + \Sigma_a W^\top \left(W \Sigma_a W^\top + \sigma^2 I\right)^{-1} \left(t_{\text{pre}} - W \mu_a - b\right)
$$

Choose $\sigma^2$ as a small fraction of $\text{trace}(W \Sigma_a W^\top) / d_{out}$ to start — sweep one order of magnitude in each direction in a later run.

## Drift metrics

Compute all of these for both methods, per layer, per sample/across samples:

### Per-sample fidelity

- **Relative L2 error per sample**: $\frac{\|\hat a_{L-1}^{(j)} - a^{*\,(j)}_{L-1}\|_2}{\|a^{*\,(j)}_{L-1}\|_2}$, then take mean and 95th percentile across samples
- **Cosine similarity per sample**: $\frac{\hat a^{(j)} \cdot a^{*\,(j)}}{\|\hat a^{(j)}\| \|a^{*\,(j)}\|}$, mean and 5th percentile

### Distributional fidelity (the key K-FAC-A claim)

- **Mean drift**: $\|\hat\mu - \mu^*\|_2 / \|\mu^*\|_2$ where $\hat\mu = \text{mean}(\hat a)$, $\mu^* = \text{mean}(a^*)$
- **Covariance Frobenius**: $\|\hat\Sigma - \Sigma^*\|_F / \|\Sigma^*\|_F$
- **Eigenvalue spectrum match**: sort eigenvalues of $\hat\Sigma$ and $\Sigma^*$ descending, compute Pearson correlation and KL divergence between sorted vectors (treat as discrete distributions)
- **Symmetric KL divergence between Gaussian fits**: $\frac{1}{2}\left[\text{KL}(\mathcal{N}(\hat\mu, \hat\Sigma) \| \mathcal{N}(\mu^*, \Sigma^*)) + \text{KL}(\mathcal{N}(\mu^*, \Sigma^*) \| \mathcal{N}(\hat\mu, \hat\Sigma))\right]$ — this is the cleanest single distributional summary

### Manifold proximity

- **Nearest-neighbor distance in $a^*$**: for each $\hat a^{(j)}$, find nearest $a^{*\,(k)}$ in the batch, report distance. Tests whether back-propagated points land on or off the activation manifold.
- **Wasserstein-2 distance between $\{\hat a^{(j)}\}$ and $\{a^{*\,(j)}\}$** (entropic Sinkhorn approximation for tractability). Direct distributional distance.

### Functional fidelity (most directly relevant)

- **Downstream forward pass match**: feed $\hat a_{L-1}$ through the *remaining* network (layers L, L+1, ...) and measure
  - prediction loss vs original labels
  - logit MSE vs the predictions from feeding $a^*_{L-1}$
  - accuracy

This is the *load-bearing* metric. The covariance preservation argument is theoretical justification; the functional metric tells you whether covariance preservation actually buys downstream usefulness.

## Hypotheses (to test, not to assume)

**H1 — Naive inversion has high per-sample error in under-determined cases.** When $d_{in} > d_{out}$ (typical going backward from a narrower to wider layer), the pseudo-inverse minimum-norm solution is fundamentally biased toward the row space of $W$. We expect per-sample relative error >50% in this regime for Method N.

**H2 — K-FAC-A inversion has lower distributional drift.** The Kalman posterior is constructed precisely to preserve the prior distribution. Covariance Frobenius and KL divergence should be smaller for Method K. *This is the central testable claim of the K-FAC-TP framework.*

**H3 — Functional fidelity tracks distributional fidelity more than per-sample fidelity.** When you feed the back-propagated activations through the rest of the network, the network's behavior depends on the *distribution* of inputs more than on per-sample exactness. K-FAC-A should win on functional metrics even when per-sample error is comparable.

**H4 — Drift grows with $d_{in}/d_{out}$ ratio.** Both methods should drift more when the layer is more "lossy" (more dimensions to recover than constraints to do so). K-FAC-A's advantage should grow with this ratio.

**H5 — ReLU dead-unit handling matters.** Layers with high fraction of dead units (post-ReLU zeros) will show larger drift in Method N. K-FAC-A with conditional Gaussian on dead units should be flatter as a function of dead-unit fraction.

**The decisive question:** if Method K and Method N produce indistinguishable functional fidelity (H3 fails), the K-FAC-A prior is computationally interesting but doesn't pay off — and the K-FAC-TP project's distinguishing technical contribution evaporates. If Method K wins meaningfully on functional fidelity, the prior is doing useful work and the project has empirical support to proceed.

## Suggested implementation outline

```python
# In a new file: benchmark/inversion_drift.py

def measure_one_layer_drift(model, data, layer_idx, method='naive', lambda_reg=1e-4):
    """Returns dict of drift metrics for back-propagating one layer."""
    # 1. Forward pass with hooks
    activations = forward_with_hooks(model, data)
    a_in = activations[layer_idx - 1]   # ground truth input
    a_out = activations[layer_idx]      # "target" = forward output
    layer = model.layers[layer_idx]
    
    # 2. Invert the non-linearity
    if isinstance(layer.activation, nn.ReLU):
        t_pre = a_out.clone()
        # For dead units, restore forward pre-activation
        pre_forward = layer.linear(a_in)
        dead_mask = (a_out == 0)
        t_pre[dead_mask] = pre_forward[dead_mask]
    # ... handle Tanh, GELU separately
    
    # 3. Invert the linear map
    W = layer.linear.weight  # shape (d_out, d_in)
    b = layer.linear.bias    # shape (d_out,)
    
    if method == 'naive':
        # Damped pseudo-inverse
        W_pinv = damped_pinv(W, lambda_reg)
        a_back = (t_pre - b) @ W_pinv.T  # shape (n, d_in)
    
    elif method == 'kfac_a':
        # Kalman posterior mean
        mu_a = a_in.mean(dim=0)
        Sigma_a = torch.cov(a_in.T)  # shape (d_in, d_in)
        residual = t_pre - mu_a @ W.T - b  # shape (n, d_out)
        gram = W @ Sigma_a @ W.T + lambda_reg * torch.eye(W.shape[0])
        chol = torch.linalg.cholesky(gram)
        adjust = torch.cholesky_solve(residual.T, chol).T  # shape (n, d_out)
        a_back = mu_a + adjust @ W @ Sigma_a.T  # shape (n, d_in)
    
    # 4. Compute drift metrics
    return {
        'sample_rel_err': sample_relative_error(a_back, a_in),
        'cov_frob': covariance_frobenius(a_back, a_in),
        'kl_gauss': gaussian_kl_symmetric(a_back, a_in),
        'eig_spectrum_corr': eigenvalue_correlation(a_back, a_in),
        'functional_loss': downstream_loss(model, layer_idx, a_back, target_labels),
    }

# Run across all layers, both methods, plot drift as function of layer depth
```

## Expected outputs

For each phase, a plot with:
- x-axis: layer index (deeper = further from output)
- y-axis: drift metric (one plot per metric)
- two curves: Method N (naive) and Method K (K-FAC-A)

What we're looking for:
- Does K curve sit below N curve? By how much?
- Does the gap widen with layer depth?
- Does the gap correlate with $d_{in}/d_{out}$ ratio (H4)?
- Does the functional-fidelity curve agree with the distributional-fidelity curves (H3)?

## Phase 2-3 extensions

After Phase 1 establishes methodology:

**Multi-step drift chain:** back-propagate through 2, 3, ..., k layers in sequence. Measure how drift compounds. Each layer's output is the next layer's target (no "ground truth" reset). This directly tests the "compounding errors kill from-scratch TP" failure mode and whether K-FAC-A mitigates it.

**Different non-linearities:** GELU, Tanh, LayerNorm. Each has its own inverse. Test the methodology on each.

**Attention layers:** for transformer blocks, the attention mechanism is structurally different. Decide whether to (a) treat the whole attention block as a single non-invertible layer and skip it, (b) invert the linear projections only (Q, K, V, O) and pass through attention as a fixed nonlinear map, or (c) develop attention-specific inversion. (a) is the cheapest; (b) is the most principled mid-ground; (c) is research-grade.

**Hyperparameter sweep on $\sigma^2$ (K-FAC-A damping):** the Kalman posterior depends on the noise term. Sweep two orders of magnitude in each direction from the heuristic starting point to find the sweet spot.

## Go/no-go criteria

After Phase 1 (MLP on MNIST) is complete:

- **Strong go**: Method K shows meaningfully lower (>2× reduction) covariance-Frobenius drift AND meaningfully lower functional-loss drift than Method N. Worth scaling to Phase 2-3.
- **Weak go**: Method K wins on distributional metrics but ties Method N on functional metrics. The theoretical story holds but the practical impact is unclear — worth one more careful look at the functional metrics before committing.
- **No-go**: Method K and Method N produce indistinguishable drift, or Method N is better. The K-FAC-A prior isn't doing useful work — revise the framework (possibly the prior is wrong, the metric is wrong, or the underlying retrainer doesn't actually need better inversion).

The cleanest possible outcome: Method K wins on distributional metrics by a factor of 5-10x and on functional metrics by a clear margin, and the gap grows with layer depth. That would be strong empirical support for the K-FAC-TP direction before committing to the larger retraining experiment.
