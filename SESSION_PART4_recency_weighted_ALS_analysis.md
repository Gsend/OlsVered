# Session Part 4 — Recency-Weighted ALS for Recommenders: Honest Analysis

*Saved: 2026-05-11*

## TL;DR — revised view, contradicting last turn

When I pitched recency-weighted ALS as the highest-EV thread, I implied two things that don't survive a careful look at the math:

1. **"Practitioners do recency via a mathematically wrong hack."** Partly wrong. The standard practitioner approach — decay raw interaction counts $r_{ui}$ by an age factor before feeding them to standard implicit ALS — is *mathematically equivalent* to the correct interaction-recency confidence formulation. It's not a hack in the "biased solution" sense; it's a clean reduction.
2. **"Algorithm 3 unlocks ALS variants that the textbook approach can't tractably express."** True but narrower than I implied. The textbook precompute-and-patch shortcut extends to a wider class of weighted ALS than I credited. The shortcut only truly breaks for **non-factorable per-(user,item) confidence**, which is a real but smaller niche.

The pitch survives, but in a narrower form. This document works through the math, classifies which weighted-ALS variants actually need Algorithm 3, and revises the economic ranking.

---

## 1. The HKV shortcut, restated precisely

Implicit-feedback ALS (Hu/Koren/Volinsky 2008) targets:

$$
\min_{X, Y} \sum_{u,i} c_{ui} (p_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2)
$$

with $p_{ui} \in \{0, 1\}$ and confidence $c_{ui} = 1 + \alpha r_{ui}$. Per-user normal equation:

$$
x_u = (Y^\top C_u Y + \lambda I)^{-1} Y^\top C_u p_u
$$

The shortcut exploits the algebraic identity

$$
Y^\top C_u Y = Y^\top Y + Y^\top(C_u - I) Y
$$

which is only useful when **$C_u - I$ is sparse** — i.e., when $c_{ui} = 1$ for all items the user didn't touch. Under standard HKV, this holds: untouched items have $r_{ui} = 0 \Rightarrow c_{ui} = 1 \Rightarrow (C_u - I)_{ii} = 0$. So the second term is a sum over only the $n_u = |N(u)|$ items the user actually interacted with.

**Per-user cost:**
- Precompute $Y^\top Y$ once per half-iteration: $O(NF^2)$ amortized over $M$ users → negligible per-user.
- Form $Y^\top(C_u - I) Y$: $O(n_u F^2)$.
- Solve $F \times F$ system: $O(F^3)$ (Cholesky on the sum).
- **Total per user:** $O(n_u F^2 + F^3)$.
- **Per half-iteration:** $O(\text{nnz} \cdot F^2 + M F^3)$ where $\text{nnz} = \sum_u n_u$ is total interactions.

This is what makes ALS tractable at web scale. The cost scales with sparsity, not with catalog size.

---

## 2. The crucial question: when does $C_u - I$ stop being sparse?

Whenever $c_{ui} \ne 1$ for items the user *didn't* interact with. Concretely, the shortcut breaks if the confidence model assigns a non-unit baseline to non-interactions.

This is the real partition. Let me classify the five "weighted ALS" variants I listed in the prior turn:

### Case A — Interaction-recency (decay weight of touches by age)

$$c_{ui} = 1 + \alpha \cdot w_{ui} \cdot r_{ui}, \quad w_{ui} = \exp(-\ln 2 \cdot \Delta t_{ui} / \tau)$$

**Untouched items:** $r_{ui} = 0 \Rightarrow c_{ui} = 1$. Shortcut works fine.

**Practitioner equivalence:** Define $r_{ui}^{\text{decayed}} = w_{ui} r_{ui}$ and run vanilla HKV ALS on the decayed counts. The resulting confidence is identical: $c_{ui}^{\text{hack}} = 1 + \alpha r_{ui}^{\text{decayed}} = 1 + \alpha w_{ui} r_{ui}$. Same formula, same solution.

**Verdict:** *Practitioners are doing this correctly via preprocessing.* Algorithm 3 offers nothing structural. **My earlier pitch was wrong.**

(Minor caveat: if a user touched item $i$ at multiple timestamps $t_1, \dots, t_k$, the right effective count is $r_{ui}^{\text{eff}} = \sum_j w(\Delta t_j)$ rather than $w_{ui} \cdot r_{ui}$ with a single $w_{ui}$. But this is still a preprocessing fix; the solver is untouched.)

### Case B — Factorable item-side priors (item-aging, inverse-popularity, exposure-by-item)

$$c_{ui} = d_i + \alpha r_{ui}, \quad d_i \text{ depends only on } i$$

**Untouched items:** $c_{ui} = d_i$. Generally $d_i \ne 1$, so $(C_u - I)_{ii} = d_i - 1$ on every untouched item. The shortcut *appears* to break.

**But:** $D = \text{diag}(d_1, \dots, d_N)$ is the same for every user. Precompute $Y^\top D Y$ once per half-iteration instead of $Y^\top Y$. The user-dependent part is still sparse: $C_u - D$ is non-zero only on touched items, contributing $Y^\top (C_u - D) Y = \alpha \sum_{i \in N(u)} r_{ui} y_i y_i^\top$.

**Per-user cost:** identical to standard HKV — $O(n_u F^2 + F^3)$. The shortcut extends naturally.

**Verdict:** A trivial generalization of the standard solver handles this. Algorithm 3 offers nothing structural here either.

### Case C — Factorable user-side priors (user-confidence scaling, user trust level)

$$c_{ui} = e_u (1 + \alpha r_{ui}), \quad e_u \text{ depends only on } u$$

**Per-user cost:** $C_u = e_u \cdot \tilde{C}_u$ where $\tilde C_u$ has the standard structure. The $e_u$ factor scales out of both sides of the normal equation up to the regularizer; absorb it into $\lambda$. Trivial.

**Verdict:** Cheap reformulation. No solver innovation needed.

### Case D — Multiplicative factorable both sides

$$c_{ui} = f_u \cdot g_i \cdot (1 + \alpha r_{ui} h_{ui})$$

**Per-user cost:** Combine Cases B and C. Precompute $Y^\top \text{diag}(g) Y$ once; scale per-user by $f_u$; add sparse user-specific update. Still $O(n_u F^2 + F^3)$ per user.

**Verdict:** Even general factorable structure submits to the standard shortcut.

### Case E — Genuinely non-factorable $C_u$

$$c_{ui} \text{ does not factor as } f(u) \cdot g(i) \cdot \text{(sparse interaction term)}$$

**Examples that are actually non-factorable:**
- **Per-session context weighting**: $c_{ui}$ depends on the session $u$ is currently in (mobile vs desktop, time-of-day, current page), and this interacts non-multiplicatively with item attributes.
- **Trust-graph weighting with dense graph**: $c_{ui} = \sum_{v \in \text{friends}(u)} s(u,v) \mathbb{1}[v \text{ liked } i]$ — depends on both user-side social structure and per-item social signal, doesn't factor.
- **Counterfactual / IPS weighting**: $c_{ui} = 1/\hat{\pi}(i | u)$ where $\hat\pi$ is a learned exposure model that doesn't factor.
- **Non-diagonal $W$**: $C_u$ is not even diagonal — e.g., $W_{ij}$ encodes similarity-aware confidence smoothing.

**Per-user cost without shortcut:**
- Forming $Y^\top C_u Y$ requires touching every item: $O(NF^2)$ per user.
- Solve: $O(F^3)$ per user.
- **Per half-iteration:** $O(M N F^2 + M F^3)$ — catalog-size $\times$ user-count $\times$ $F^2$. **This is brutal.** For Netflix-scale ($M = 10^8$, $N = 10^5$, $F = 128$) it's $\sim 10^{17}$ flops per iteration vs $\sim 10^{12}$ for sparse HKV. Five orders of magnitude.

**Verdict:** This is where the shortcut truly breaks. And it breaks so hard that *no solver, including Algorithm 3, makes this tractable at scale by itself.*

---

## 3. So where does Algorithm 3 actually help?

Two places, both narrower than I previously claimed:

### Win 1 — Numerical robustness for the small-$n_u$ regime (long-tail users)

Even within standard HKV, the per-user system $Y^\top C_u Y + \lambda I$ becomes ill-conditioned when $n_u$ is small relative to $F$. Forming the Gram explicitly squares the condition number of $Y_{N(u),:}$ (the row-restriction of $Y$ to touched items). For a user who touched 3 items with $F = 128$, the Gram is severely rank-deficient and only $\lambda I$ saves it; the resulting $x_u$ is largely driven by the prior, with high variance on the data-driven component.

Algorithm 3's LU on the augmented matrix structure sidesteps the conditioning damage of forming $Y^\top C_u Y$ explicitly. Concretely: instead of computing $A = Y_{N(u),:}^\top C_{u,N(u)} Y_{N(u),:} + \lambda I$ and Cholesky-factoring, you factor the augmented $[Y_{N(u),:}; \sqrt{\lambda} I_F | p_{u,N(u)}; 0]$ via LU and back-substitute.

**Expected effect:** Better $x_u$ estimates for users in the bottom decile of interaction count. Translates to **long-tail NDCG / coverage metrics**, not to head-of-distribution accuracy. Plausibly 1–3 points of long-tail NDCG@20 improvement, by analogy with the QR-vs-Normal-Equations difference in linear regression literature. Worth benchmarking but not a 10×.

This is a real, defensible economic argument. Long-tail performance is the single most-watched metric at companies running recommenders precisely because head items get recommended regardless of algorithm.

### Win 2 — The non-factorable regime, *if you can bound $N$*

For Case E above, the $O(NF^2)$ per-user cost is fatal at full catalog scale, but it's tolerable in restricted settings:

- **Within-category recommenders**: $N$ restricted to a single product category (~$10^4$ items). $MNF^2$ with $M = 10^7$, $N = 10^4$, $F = 64$ is $\sim 4 \times 10^{15}$ flops, ~hours on a GPU — feasible.
- **Candidate-restricted re-ranking**: ALS as a re-ranker over ~1000 candidates from a fast retrieval stage. Now $N = 10^3$, very tractable.
- **Session-bounded recommenders**: $N$ = items shown in current session.

In these settings, Algorithm 3's ability to absorb non-diagonal $W$ natively, with numerical robustness, is a genuine functional capability. The pitch is: *"ALS for re-ranking and session-bounded settings with rich context-aware confidence."*

This is a smaller market than "all ALS" but a real one — re-rankers are everywhere, and the limitation that re-rankers usually have to be neural (because traditional ALS can't express context) is precisely what Algorithm 3 could relax.

### Where Algorithm 3 does *not* help

- Standard HKV with vanilla $c_{ui} = 1 + \alpha r_{ui}$. (Cholesky on the F×F Gram is already non-inverting and well-conditioned for typical $n_u$.)
- Cases A–D above — these reduce to standard HKV by reformulation.
- Beating `implicit` on MovieLens-style benchmarks at head metrics. (You'll tie or lose by 10–20%.)
- Web-scale non-factorable ALS. (No solver fixes the $O(MNF^2)$ data-motion cost.)

---

## 4. Revised economic ranking

Last turn I ranked recency-weighted ALS #1. With the corrected analysis, the ranking shifts:

| Rank | Thread | EV story | Confidence |
|------|--------|----------|------------|
| **1** | Long-tail numerical robustness | Better cold-user embeddings → measurable long-tail NDCG/coverage lift. Applies to *every* implicit ALS deployment. Big addressable market, small per-deployment delta. | Medium-high |
| **2** | Context-aware re-ranker ALS | Algorithm 3 unlocks non-factorable $C_u$ in candidate-restricted settings where $N$ is bounded. Re-rankers are everywhere; current ones are usually neural because traditional ALS can't express context. | Medium |
| **3** | OlsSMLayerRetrainer integration | You already have `retrain_lora_als` in production. Algorithm 3 could replace the inner solver there. Lower market reach but you own the codebase and the wins are visible in your existing benchmark suite. | High (you'd see the result fast) |
| 4 | Differentiable ALS (hybrid models) | Niche research. Reputation value, weak direct EV. | Medium |
| 5 | "Recency-weighted ALS" as standalone pitch | The framing I sold last turn. Most cases reduce to standard HKV via preprocessing. Not a defensible wedge on its own. | I was wrong |

**The single most surprising correction:** option 3 is probably the *fastest path to a clean result*. You already have an ALS-LoRA codepath that works (84.98% / 85.21% from PART2). Swapping in Algorithm 3's weighted generalized inverse as the solver — with non-uniform damping per layer based on activation statistics — is a one-week experiment in code you control. The result is directly comparable to your existing benchmark numbers. No new benchmark infrastructure, no recommender data wrangling, no comparison with `implicit`'s years of optimization.

---

## 5. Concrete next experiment, if you want one

If you want a real test of Algorithm 3's value in an ALS-shaped problem, the cheapest credible experiment is:

**In `OlsSMLayerRetrainer`, replace the per-layer ridge solve with Algorithm 3 using *per-row activation-magnitude-weighted* confidence.**

Currently `_single_layer_retrain` does a uniform ridge solve. The hypothesis: rows of $X$ (input activations) with extreme magnitudes are unreliable evidence and should be downweighted. Define $w_n = 1/(\|x_n\|^2 + \epsilon)$ or similar. This is exactly the regime Algorithm 3 was designed for, and you'd see immediately whether weighted-LU beats uniform Cholesky on your existing N=1, N=2, N=4 BCD benchmarks.

Concrete predictions:
- N=1 OLS baseline (currently 84.98%): expect ~85.0–85.5% with row-weighting if the hypothesis holds.
- Weighted-LU should produce more stable BCD sweeps under your λ-scaling regime — the conditioning argument from §3 Win 1 maps directly to BCD divergence cases.
- If neither materializes, that's evidence Algorithm 3 isn't load-bearing for ALS-as-solver applications either, and the recommender angle is even weaker than this analysis suggests.

This is a 1–2 day spike that decisively informs whether to invest in the larger recommender direction.

---

## 6. What I'd tell you if you asked the original EV question fresh

The question wasn't "where can Algorithm 3 win" — it was "what's the economic value of using Algorithm 3 in recommenders." With the corrected math, the honest answer is:

> **Small but real, concentrated in long-tail metrics and non-factorable-$C_u$ niches. Not a paradigm shift. Worth a focused 2–4 week experiment if and only if the row-weighted ALS spike in your existing retrainer benchmark shows a measurable effect. If that spike comes back flat, the recommender direction is unlikely to pay off and you should stay with K-FAC work.**

The expected value of the spike is high — it tells you in days whether the larger investment is worth it, in code you already own.
