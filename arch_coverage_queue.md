# Architecture Coverage — Experiment Queue

Maintained by TeamLeader. Updated after each completed experiment.

| # | Exp | Status | Notes |
|---|-----|--------|-------|
| E1 | DeepMLP-10L, MNIST | ✅ COMPLETE | GREEN. H5, H6 confirmed. Surprising: no-mom >> mom-match at depth 10. |
| E2 | LNMLP-10L (LNReLU), MNIST | ✅ COMPLETE | GREEN. H1, H4, H5, H6 confirmed. K>N (unlike E1). Monotone decline in B-iter (vs E1's collapse). |
| E3 | LeNet CNN, MNIST | ✅ COMPLETE | GREEN (run complete). MaxPool breaks the chain. FC-only works; conv+pool fails catastrophically. Q8 opened. |
| E4 | MLP autoencoder, MNIST recon | ✅ COMPLETE | GREEN. Continuous targets work from random (MSE 212 vs 798). naive>K-FAC (reversed from classification). Distill degrades for autoencoders. Q9 opened. |
| E7 | 1D-CNN, synthetic time-series | 🔒 P-B | 1D conv (similar to E3). Decide Q8 first. |
| E5 | ResNet-8, CIFAR-10 | ✅ COMPLETE | GREEN. Train 71.24%, distill 71.25% (skips don't break distillation), chain-retrain 33.06% (skip-decomp partially works, pooling wall again). Q1 resolved (decompose y=x+f(x)). |
| E6 | Small U-Net, MNIST denoising | ✅ COMPLETE | GREEN. Train MSE=0.0070, distill MSE=0.0070 (Gate B1 PASS: concat-skips don't break distillation), chain-retrain MSE=0.0135 (5.2x improvement from random, 1.9x from teacher — pool+concat chain noise). |
| E8 | Tiny transformer | ✅ COMPLETE | Train 100% (majority)/99.98% (pointer), distill 100%/99.98% (exact). Retrain-probe: majority=77% (22.6 pp wall), pointer=57.6% (42.4 pp wall — clean attention wall isolator). Q2+Q10 resolved. |
| E9 | Tiny GRU (optional) | 🔒 P-F | Only if P-A through P-E give sufficient intuition. |

**Phase gate:** P-A (E1+E4) ✅ COMPLETE. P-B (E2+E3) ✅ COMPLETE. P-C (E5+E6) and P-D gated on open questions.

**Open questions (ordered by urgency):**  
- Q6: At what chain depth does moment-matching switch from beneficial to harmful? (between 4 and 10 layers)  
- Q7: Does E2's monotone-decline B-iter (vs E1 collapse) reflect LayerNorm normalizing drifting targets?  
- Q8: Can MaxPool inversion be made exact via stored argmax indices? Would this fix E3-class conv chains?  
- Q9: For autoencoders, should TP distillation use original inputs rather than model outputs?  
- Q1: ✅ RESOLVED — ResNet block inverted by additive decomposition y=x+f(x): t_f = ReLU⁻¹(t_out) − x_block_input; recurse input target through f's first conv. BN omitted (no norm-inversion primitive); identity skips only; maxpool reuses E3 switch-unpool.
- Q2: ✅ RESOLVED — attention treated as opaque/non-invertible (like max-pool). Q/K/V/O are OLS-fit as Linears; softmax mixing is replayed forward, never inverted. Label-retrain therefore cannot reach Q/K/V (the "attention wall").
- Q10: ✅ RESOLVED — pointer task (d_model=128, std=0.1 pos init) achieves 99.98% train; retrain-probe drops to 57.6% (42.4 pp wall). Clean attention wall measure confirmed. See Q10 resolution section in findings.

