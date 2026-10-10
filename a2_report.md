## Abstract

Default LM unless stated: course d8 model (width 512, 8 layers), AdamW, batch B = 64, WD .1, 1% warmup + linear decay, final val loss. Seed noise ≈ 0.0017 (A1 replicates).
Cite as: **"In Problem X (model, budget, what was swept), [number] vs [number], so [claim]."**

| Part | What we ran | Key numbers | Takeaway |
|---|---|---|---|
| **1a** LR vs budget | Supplied AdamW LR sweeps at 153.6M / 307.2M / 614.4M tokens; quadratic-in-log-LR fit per budget, then power law in D | Optimal LR +25% (153.6M→307.2M), only +7% (307.2M→614.4M) | Optimal LR *rises* with budget (opposite of 1/√T) and the rise slows: not one power law |
| **1b** Predict larger budgets | 1a law vs supplied sweeps at 1.2288B / 1.8432B / 2.4576B | Predicted .0038 / .0041 / .0044 vs fitted .0039 / .0034 / .0029; loss gap 0 / +.0005 / +.0030 | Optimum rises then falls, so the prediction degrades with D |
| **1c** Larger-three vs all-six fit | 5 runs at **4.9152B**: grid .0015/.003/.006 + two predictions | Larger-three (exp −0.41) LR .00223 → **2.7166** (best of all 5); all-six (exp +0.10) LR .00373 → 2.7185; grid best .003 → 2.7172 | Local fit wins, but the ≈.002 gain is within noise (≈.003) |
| **1d** Hyperball vs AdamW | Hyperball (norm-preserving Adam, WD 0) at **1.2288B**; LRs .007 / **.0105 (predicted)** / .015 / .021 | Predicted .0105 best (2.8272; neighbours +.009); beats AdamW at the same budget by **.0106** | Smoother LR trend and lower loss, but sharper U (less LR-tolerant) |
| **2a** Joint LR–WD optimum | Supplied LR×WD grids at 4 budgets (153.6M–1.2288B); paraboloid in (log LR, log WD) | Best WD .85 → .16 (∝ D^−0.82, R² .994); LR ∝ D^+0.22 (R² .68); gain from tuning WD shrinks **26×** | WD follows a clean power law, LR does not; WD matters most at small D |
| **2b** LR–WD compensation | From each optimum: 2× LR with ½ WD, at 4 budgets | Loss cost .0376 / .0098 / .0136 / **.0020**; valley angle 67° / 62° / 70° / **48°** (45° = product line); η*λ* ∝ D^−0.59 | Keeping ηλ fixed works increasingly well at large D |
| **2c** Test the product law | **2.4576B**: product-law LR .003, WD .0927 vs LR .003 × WD {.05, .1, .2} vs reused 153.6M optimum (.0015, 1.6) | **2.7670** vs best baseline 2.7683 (WD .1) vs 2.8536 | Product rule beats reusing the small-budget recipe by **.087** |
| **3.1a** NQM SGD | 2D NQM H = diag(1, 10), σ = 1, N = 8,192; LR sweeps at B = 1–64, predict 256, extend to 512 | LR* ∝ B^**0.99**; at B = 256 predicted .146 vs tuned .124 (loss ×1.27) | Linear scaling holds up to B ≈ 64; then the stability limit 2/h_max = .2 caps LR |
| **3.1b** NQM RMSProp/Adam | Same NQM, LR tuned per B | Exponent RMSProp .894, Adam .886; B = 256: RMSProp LR off 9% (loss ×1.02), Adam off 2.2× (loss ×1.06) | Similar exponents but different transfer: Adam's floor is flat (×2.6 LR range within 5%) |
| **3.1c** Signal vs noise | σ ∈ {1, 10, 100, 300}, 2D and scalar NQM | Adam exponent .88 (σ = 1) → .50 (σ ≥ 100); σ = 10 transitions (transfer loss ×1.23–2.06) | Noise-dominated v ⇒ √B rule; signal-dominated ⇒ exponent toward 1 |
| **3.1d** NQM momentum | B = 16 vs 256; SGD μ ∈ {0, .5, .9, .95, .99}; Adam β1 sweep (β2 = .95); LR retuned | B = 16: ~15% gain; B = 256: up to 1.85× (SGD) and 6.8× (Adam); best μ = .5, β1 = .8; .9 hurts at B = 256 | Momentum helps more at large B; with only 32 updates, high β1 is too slow |
| **3.2a** LM LR vs batch | **614.4M tokens**, WD .1, LR sweeps at B = 8 / 16 / 32 / 64 (B = 16/32 reused A1, B = 64 from P1) | LR* .00101 / .00181 / .00290 / .00319; local exponents .84 / .68 / .14; best loss 2.9337 / 2.9238 / **2.9189** / 2.9256 | LR scaling flattens past B ≈ 32, unlike NQM's clean √B |
| **3.2b** Scale LR or WD with B | Fit on B = 8–64, then 3 runs (×½, ×1, ×2) per hypothesis at **B = 128, 256** (12 runs) | (i) WD .1, LR ∝ B^0.57 → .0054 / .0080: gap **+.028 / +.161**. (ii) LR .0015, WD ∝ B^0.98 → .95 / 1.88: fitted .938 / 1.80, gap **0 / 0** | Scaling WD ∝ B (fixed ηλT/2 ≈ 3.3–3.8) transfers; scaling LR fails |
| **3.2c** LM momentum ablation | B = 8 (LR .001, WD .1) and B = 256 (LR .0015, WD 1.9), β2 .95, β1 ∈ {0, .5, .8, .9, .95, .98, .99}, LR fixed | Gain over β1 = 0: **−.047** at B = 8 (best .98, 2.9248) vs **−.246** at B = 256 (best .95, 2.9671); β1 = .99 costs +.006 vs +.175 | Momentum matters ~5× more at large batch |
| **3.2d** NQM vs LM | Horizon test: NQM B = 16, 512 → 4,096 updates | β1 = .99 goes from ×11 worse to the best setting (×0.87) | NQM's β1 blow-up is a short-horizon artifact |
| **Batch-switch example** | B 64 → 128 at 307.2M of 614.4M tokens; LR .0015, WD 1.6 | Answer **A** (2.93867 vs B 2.94538): WD 1.6 gives n ≈ 11 vs optimum ≈ 3 at 614M | Run is over-decayed, so halving per-token decay (doing nothing) helps |
| **4.0** Alignment derivation | 4 combinations of update alignment α and readout alignment ω_move | Full/full (α = ω = 1) recovers μP: hidden LR η/m, readout multiplier 1/m, readout init 1/n₀ | α sets hidden LR and readout multiplier; ω sets readout init only |
| **4.1a** Width transfer, 5 steps | 2-layer stress model, 5 Adam updates, constant LR, FP32, n₀ = 512; widths **640 / 2560 / 5120** | LR*: Kaiming .0014 / .00015 / .000057 (∝ m^−1.5) vs μP .0019 / .0018 / .0017; best loss Kaiming 6.822 / 7.161 / 7.313 vs μP 6.773 / 6.654 / 6.661 | μP transfers its LR *and* gets lower loss |
| **4.1b** Width probes | 4.1a runs at best LR | μP M₅ 1.12 / 1.02 / 1.02 vs Kaiming 1.14 / .64 / .54; μP initial logits .89 / .45 / .32 (∝ n^−½); cross-width α ≈ .93–1.07; ω_move ≈ .5 | Update alignment ≈ full; readout alignment ≈ none |
| **4.1c** Depth transfer, 5 steps | Width 64, BF16, depths **2 / 100 / 1000** (r = L/2), 7-point LR grid .003–.1 | LR* vs source .0158: standard μP ×.87 / ×.76; Depth-μP ×1.05 / ×1.00; CompleteP ×1.02 / ×.97. Loss vs 6.851: μP +.21 / +.22; Depth-μP 0 / +.11; CompleteP −.04 / +.02 | Depth-μP and CompleteP transfer; only CompleteP keeps loss flat |
| **4.1d** Depth probes | 4.1c runs at best LR (.0173) | Residual RMS after 5 updates: μP **797 / 7,334** vs Depth-μP 5–8, CompleteP 6.5–6.7 (source 9.8); last FFN branch: Depth-μP .54–.76 vs CompleteP 4.5–5.4 (source 4.6) | Normalization hides residual blow-up; Depth-μP blocks go lazy |
| **4.2a** Width transfer, long run | **153.6M tokens**, widths 128 / 256 (SP vs μP, n₀ = 512), source LR .003 (P1 best at 512) | μP cross-width α ≈ 1 through warmup → .75–.8 at end (o/down .5–.6); ω_move ≈ .50 throughout | μP's α = 1 assumption holds early only |
| **4.2b** Transfer vs tuned | Same runs, local quadratic fits | LR* SP .0081 / .0040 / .0024 (∝ n^−0.88) vs μP .0030 / .0020 / .0024; .003 costs SP **+.065** (w128) vs μP ≤ +.010; best loss equal (3.627 vs 3.629) | Better transfer ≠ lower tuned loss |
| **4.2c** Held-out width 1024 | Power law on widths 128–512; 6 runs per prescription at width 1024 | SP: predicted .00126 vs fitted .00117 (gap 0); .003 gap **+.142**. μP: predicted .00193 vs .00237 (gap +.0025); .003 gap +.004. Best: μP 3.0992 vs SP 3.1041 | Fitting rescues SP; μP needs no fit |
| **4.2d** Depth transfer, long run | Width 512, depths **4 / 16** (r = L/8, source = depth 8), LR .0015 / .003 / .006 (+.00075) | LR*: μP .0024 / .0019; Depth-μP .0018 / .0028; CompleteP .0025 / .0020. Gap at .003: μP 0 / +.020; Depth-μP **+.024** / 0; CompleteP 0 / +.012 | Residual multiplier ≈ no-op (tiny embedding + pre-norm); Depth-μP over-corrects |
| **4.2e** Regime diagnostics | All 4.2 runs at transferred and best LR | Update 1: embedding 90–99% of grad², clip coefficient .03–.26. Bad-LR residual spike: SP w1024 **628** vs 71; μP L16 **180** vs 35. Fixed-WD timescale n: μP 1.41 (w128) → .18 (w1024) vs optimum ≈ 1.8 | LR shift alone ≠ instability; residual RMS is the tell; fixed WD confounds |

## Problem 1-when can training on less data guide large-scale runs

### a-learn from small training budgets

The optimal learning rate goes up as the training budget grow, which is the opposite of the classic “$1/\sqrt{T}$” rule. 

Also, the linear pattern does not seem to fit, as the optimum rises about 25% from 153.6M to 307.2M, but only about 7% from 307.2M to 614.4M.

!image.png

### b-predict before increasing the budget

**T**he optimum rises, then falls. Since its shows non-linearity, no power law can fit them.

As a results, the gap between our predicted lr and the optimum increases as D grows, which in turn increases |loss_at_predicted lr - loss_at_fitted_optimum|.

!image.png

| D | Predicted LR | Fitted optimal LR | Fitted / predicted | Loss at predicted LR − fitted min |
| --- | --- | --- | --- | --- |
| 1.2288B | 0.00380 | 0.00388 | 0.98 | 0.0000 |
| 1.8432B | 0.00414 | 0.00343 | 1.21 | 0.0005 |
| 2.4576B | 0.00440 | 0.00291 | 1.51 | 0.0030 |

### c-compare and test two scaling runs (larger-three vs all six)

I expect the larger-three fit to have the lower loss and be closer to the optimal learning rate because it better captures the local, down-falling pattern trending at larger N.

The optimal lr inherits the tail pattern of observations: it continues to fall as we double N from 2.5B to 4.9B.

From the bias-variance trade-off perspective, the larger-three fit has high variance but less local bias, whereas the all-six fit has lower variance but more local bias.

Also, **`the gain from having a more accurate prediction is very small`**: the loss difference between the better larger-three fit and the worse all-six fit is on the scale of **`0.002`**, which is within the range of noise (0.003).

!image.png

| LR | Run type | Loss |
| --- | --- | --- |
| 0.0015 | grid | 2.7175 |
| **0.00223** | **"larger three" prediction** | **2.7166**, the lowest of all five |
| 0.003 | grid | 2.7172, best of the three grid LRs |
| 0.00373 | "all six" prediction | 2.7185 |
| 0.006 | grid | 2.7278 |

Compare the performance of the two predictions

| Rule | Predicted LR | Predicted ÷ fitted optimum | Loss gap vs. best grid LR (.003) |
| --- | --- | --- | --- |
| Larger three (exponent −0.409) | 0.00223 | **1.01** | **−0.0006** (better) |
| All six (exponent +0.099) | 0.00373 | 1.69 | +0.0013 (worse) |

### d-Hyperball vs AdamW

Hyperball’s optimal lr changes more smoothly than AdamW’s. Hyperball’s RMS in loss space is 1%, in contrast to AdamW’s 4%, both relative to their own loss scale. Hyperball’s fitting curve also has a smaller slope.

Hyperball’s predicted lr leads to the lowest loss, which is ~0.008 lower than nearby lr.

Hyperball also leads to a lower loss than AdamW at the same budget, and the difference 0.0106 is very non-trivial.

But Hyperball’s sharper U-shape curve is less tolerate to error (i.e., scaling the optimal lr by the same factor would leads to a bigger loss increase).

Caveat: AdamW also shows a linear trend across the first 4 budgets but then curves down beyond 1.2B. It is unknown whether Hyperball would encounter a similar problem. [TODO: 3 runs at a larger budget to see if the declining trend holds]

!image.png

!image.png

**Hyperball’s Fitting Results at 1.2288B tokens:**

| LR | Loss | Difference from best |
| --- | --- | --- |
| 0.007 | 2.8362 | +0.0090 |
| **0.0105 (predicted)** | **2.8272** | best sampled |
| 0.015 | 2.8360 | +0.0088 |
| 0.021 | 2.8567 | +0.0295 |

### e-explore schedule variants

[TODO] Worth trying constant and WSD

## Problem 2-how should learning rate and weight decay scale together

### a-let LR and WD vary together

- As (D) scales up, the optimal learning rate increases while the optimal weight decay decreases.
- At a fixed scale, higher learning rates generally pair with lower weight decay, as suggested by the orientation of the level curves.
- The joint optimal LR is lower than Problem 1's at every budget (because it's paired with a higher WD, and higher WD favours lower LR). Both trends rise with the same exponent (about +0.22), but problem 1's is much cleaner (R² = 0.97 against 0.68)
- **Described by a single power law?** WD can, LR CANNOT.

!image.png

!image.png

!image.png

- The gain of tuning WD shrinks as D scales, about 26× across our test range. It's large at small budgets, where the best WD (0.85) is far from .1, and roughly gone by 1.2288B, where the best WD (0.16) is close to .1.
- [TODO]: redo the plots below but tuned WD at each scale (need 4 more runs) to see if the trends of the norm shows the common up and down pattern. If this is true, then we can claim

!image.png

### b-from coupling to a joint scaling law (product of lr and wd)

- **Compensation kinda works, which supports NQM’s cumulative-decay-invariant  argument**
    - compensation partly works for the smallest three budgets, and works almost perfectly for the largest budge (1.2B). This suggest that, when fixing the token budget and batch size(and thus the total number of optimization steps), the optimal $\eta\lambda$ product stays constant. This supports the hypothesis that the optimal cumulative WD shrinkage $\approx \exp(-\eta\lambda D/B)$ is fixed given a token budget.
- At small budges, the valley is steeper than the product line: 2x LR change takes roughly 4-5x WD change.
- Valley grows narrower as budget grows: from 4x to 16x

| D | Valley angle (product line = 45°) | Loss cost of 2× LR with ½ WD | 2× LR alone | 2× WD alone | Valley narrowness |
| --- | --- | --- | --- | --- | --- |
| 153.6M | 67° | 0.0376 | 0.0468 | 0.0195 | 4× |
| 307.2M | 62° | 0.0098 | 0.0148 | 0.0069 | 5× |
| 614.4M | 70° | 0.0136 | 0.0196 | 0.0055 | 6× |
| 1.2288B | **48°** | **0.0020** | 0.0086 | 0.0072 | **16×** |

!image.png

- WD is tightest, then the product, then LR.
- The produce shares the same declining patter as WD and does not possess most of LR’s variations. This indicates that the loss surface is more sensitive to WD than LR.

| Quantity | Variation across budgets | Exponent | R² | Scatter around fit (RMS, log) |
| --- | --- | --- | --- | --- |
| Optimal LR, η* | ×1.68 | +0.22 | 0.68 | 0.119 |
| **Optimal WD, λ*** | ×5.22 | **−0.82** | **0.994** | **0.051** |
| Product, η*·λ* | ×3.60 | −0.59 | 0.967 | 0.084 |

!image.png

### c-test the joint rule

- **Yes**, the product rule gives an effective recipe?
- The gain of 0.08 compared with the reused recipe from 150M is substantial.

| Run | LR | WD | Loss | Prediction minus run |
| --- | --- | --- | --- | --- |
| **Product-law prediction** | .003 | 0.0927 | **2.7670** | — |
| Baseline (ii) | .003 | .05 / .1 / .2 | 2.7734 / **2.7683** / 2.7722 | **−0.0013** vs the best (WD .1) |
| Baseline (i), reused 153.6M optimum | .0015 | 1.6 | 2.8536 | **−0.0866** |

!image.png

## Problem 3.1

### a-SGD

- The analysis of NQM suggest that when $\eta h_\text{max}\ll 1$, the final error is proportional to $\eta/B$. So, if $\eta^{\star} = \argmin L(B,\eta)$, then $k\eta^{\star} = \argmin L(kB,\eta)$.
- Our empirical observation that, at small $B$ (1-64), the optimal LR grows linearly (fitted power~0.99), supports this.
- The rule is useful up to about **B ≈ 64.** Beyond it the predicted optimal learning rate is over 0.1, making $\eta^{\star}h_{\text{max}} >1$. When the produce is >1, the descent in the steep direction diverge. so SGD no longer converges to the global optimal. The increasing batch loss in rightmost panel supports this.

!image.png

!image.png

### b-adaptive updates

- **Do they follow the same rule?**
    - RMSProp and Adam nearly match the power(0.894 vs 0.886), but neither matches SGD's approximate linear power of 0.99 or the square-root rule's 0.5.
    - The fit is also curved: as we can see the points being fitted slightly concave up, meaning that the power grows as $B$ increases
- **Does a similar exponent mean similar transfer accuracy?**
    - No
    - **RMSProp transfers well.** Its LR is off by 9%, costing only 2% in loss.
    - **Adam's LR is off by 2.2×, yet the loss is only 6% worse.** This is beacause its B=256 loss curve has a wide, bumpy floor (×2.6 in LR within 5% of the best loss), so "the tuned LR" is barely defined there.
- The best loss of both methods deviates from the optimal loss faster than SGD, as shown by the right most panel of the second figure.
    - (Claude gives an useful explanation)
    - Under the NQM, the second moment splits roughly as **v ≈ h²·E[w²] (signal, dominating) + σ²/B (noise)**.
    - With σ = 1 and initial gradients of order 1–3, the signal term dominates for most of training, and increasingly so as B grows.
    - Also, in this regime, Adam and RMSProp behave like **sign descent.**
    - When we scale B, larger batches stop buying much better directions. They mainly just remove optimization steps. With normalized, nearly fixed-length updates, fewer steps means less total distance traveled, so the optimum can become unreachable within the fixed example budget.
- Prediction for the LM setting
    - **Adam's optimal LR should rise with B, but sublinearly,** with an exponent somewhere between 0.5 and 1.
    - **The rule should stop working past a critical batch size.** Beyond some B, likely around 128–256 sequences here, the optimal LR should flatten out, and the best loss at a fixed token budget should get worse because of fewer steps. For Adam, that degradation may start earlier than for SGD, because normalized steps don't benefit from lower gradient noise.
    - **A fitted exponent can transfer poorly for optimal LR, even when the loss doesn't suffer.** Where the loss–LR curve is flat, a wrong LR costs little; where it's sharp, as near instability, it costs a lot.

| Method | Fitted exponent (B = 1–64) | Predicted LR at 256 | Tuned LR at 256 | Predicted ÷ tuned LR | Loss at predicted ÷ loss at tuned | LRs within 5% of best loss |
| --- | --- | --- | --- | --- | --- | --- |
| SGD (part a) | B^0.992 | 0.146 | 0.124 | 1.18 | 1.27 | narrow |
| RMSProp | B^**0.894** | 0.087 | 0.095 | 0.91 | **1.02** | 0.088–0.100 (×1.14) |
| Adam | B^**0.886** | 0.083 | 0.181 | **0.46** | **1.06** | 0.088–0.228 (**×2.6**) |

!image.png

!image.png

### c-explore the assumptions

- **Does the signal-to-noise ratio change the scaling rule?**
    - As we raise σ, noise now dominates the split of the second moment:**v ≈ h²·E[w²] (signal) + σ²/B (noise, now dominating).** The normalizer $\sqrt{v} = \sigma/\sqrt{B}$ is now independent of the current gradient. Adam then is SGD with effective LR $\eta*\sqrt{B}/\sigma$, meaning the optmial raw LR should get divided by $\sqrt{B}$.
    - Since SGD has a linear scaling function $\eta^{\star}_{\text{SGD}} = CB$, the above analysis suggests $\eta^{\star}_{\text{Adam}} = CB/\sqrt{B} = CB^{0.5}$.
    - The left panel of the first figure shows the fitted exponent converges to 0.5 as noise increases, thus confirming the above analysis.
- $\sigma=10$ is the most interesting case, as it shows the transition from being noise dominated to signal dominated.
    - At small batch sizes, noise dominates, so the sqrt law holds
    - But at around B=32-64, the signal term over takes the loss term, and the scaling rule of LR begins to behave like the linear rule as in $\sigma=1$. That’s why we significantly underestimate the optimal learning rate at B=256.

| Model | σ | B_c = σ²/(h²·Var w₀) | RMSProp p | Adam p | SGD p | Loss at predicted ÷ tuned LR, B=256 (RMSProp / Adam) |
| --- | --- | --- | --- | --- | --- | --- |
| 2D | 1 | 0.1 | 0.89 | 0.88 | 0.99 | 1.03 / 1.04 |
| 2D | 10 | 10 | 0.54 | 0.52 | 1.00 | **1.28 / 1.23** |
| 2D | 100 | 1,000 | 0.50 | 0.50 | 1.00 | 1.00 / 1.01 |
| 2D | 300 | 9,000 | 0.50 | 0.49 | 1.00 | 1.00 / 1.01 |
| scalar | 1 | 0.5 | 0.97 | 0.95 | 1.00 | 1.19 / 1.23 |
| scalar | 10 | 50 | 0.57 | 0.54 | 1.00 | **2.06 / 1.50** |
| scalar | 100 | 5,000 | 0.50 | 0.49 | 1.00 | 1.00 / 1.00 |
| scalar | 300 | 45,000 | 0.47 | 0.46 | 1.00 | 1.00 / 1.00 |

Note: B_c is the batch size below which the noise term exceeds the largest initial squared gradient.

!image.png

!image.png

- When noise dominates, the best losses of all three methods stay close to each other. But when the signal takes over, the two adaptive methods becomes worse.
    - We can see this phenomenon from the following figures. For $\sigma=1$,  Adam and RMSProp diverges from SGD as soon as we increase B. For $\sigma=10$, Adam and RMSProp diverges when $B$ surpasses the signal-noise equilibrium point. For $\sigma=100,300$, the equilibrum point is not yet reached in our sweep so they stay close to each other throughout.
    - This happens for the same reason as I explained in the previous part: when signal dominates, the adaptive methods degenerates to fixed-length step size, loosing the self regularizing benefit of raw gradient that the gradient magnitudes shrinks as we approach a optimum.

!image.png

- **Which conclusions depend on the curvature model?**
    - The main signal-vs-noise story is robust. The sensitive parts are the following.
    - Exponent at low noise: 0.89 in 2D against 0.97 for the scalar. With two curvatures.
    - Cost of a bad transfer: at σ = 10 a wrong LR costs 1.5–2.1× on the scalar model against about 1.25× in 2D, because the loss curves have different shapes.
    - Where SGD's rule breaks: SGD's η*/B at B=512 is 0.57× its small-batch value in 2D but 0.85× on the scalar. The breakdown is set by the stability limit 2/h_max, which is 10× larger for the scalar model.

### d-Momentum at small and large batches

- **How does the benefit of momentum depend on batch size**
    - B=16 (noise-dominated): a modest gain of about 15%, and momentum mostly rescales the LR. The tuned SGD LRs keep the effective LR η/(1−μ) nearly fixed: 0.0093 (μ=0), 0.0093 (μ=.5), 0.0089 (μ=.9), 0.0083 (μ=.95). That's the classical NQM claim: once the LR is tuned, momentum adds little. The remaining ~15% comes from averaging noise over more steps.
    - B=256 (few steps, curvature-limited): a much larger gain, up to 1.85× for SGD and 6.8× for Adam, consistent with the prediction that momentum helps more at large batches. Momentum lets SGD exceed plain SGD's stability limit (2/h_max = 0.2). At μ=.9 the effective LR is 0.27, faster than SGD alone can go.
    - **But the best coefficient is much smaller at B=256** (μ=.5, β₁=.8), and the usual 0.9 hurts (SGD 1.69×, Adam 1.12×). The cause is that we only have a total of 32 updates.

!image.png

- **Learning curves:** at B=256, the best momentum runs pull ahead over the second half and finish clearly below the no-momentum runs

!image.png

- **Likely to carry over:** momentum should be worth more at larger batches, because gradient noise is lower there, so dropping
- **Likely not to carry over:** the optimal value of the momentum coefficient. The finite-horizon penalty that made β₁ = .9 lose here. But LM rus have much more updates that give a lot more time for momentum to warmup.

## Problem 3.2

### a-tune lr with WD

- **Scaling law of LR—flattens as B increases, does not comply with any NQM setting**
    - With WD fixed at 0.1, the optimal peak learning rate increases with batch size, but the scaling is not described by a single clean power law. I suspect the flattening occurs because the LM has more complex curvature that Adam’s diagonal preconditioning cannot fully neutralize as in NQM, causing the optimal LR to approach a stability limit.
- **Loss—down and up, does not comply with any NQM setting**
    - best loss initially improves with batch size, from B=8 through B=32, but worsens slightly at B=64. These does not match any of our NQM setting where the best loss only increases when B grows. I suspect that the initial decrease in best loss is attributed to a LM specific issue that’s not explained by NQM.

| B | Updates | Optimal LR | Best measured loss | Local exponent to the next B |
| --- | --- | --- | --- | --- |
| 8 | 75,000 | 0.00101 | 2.9337 | 0.84 |
| 16 | 37,500 | 0.00181 | 2.9238 | 0.68 |
| 32 | 18,750 | 0.00290 | **2.9189** | **0.14** |
| 64 | 9,375 | 0.00319 | 2.9256 | — |

!image.png

## b-compare two scaling hypotheses

- **Scaling Law of WD vs B—almost perfectly linear, supports NQM’s cumulative-decay-invariant argument**
    - The best WD is almost exactly proportional to B. This keeps the cumulative weight decay shrinkage, $(1-\eta\lambda)^{D/B} \approx \exp(-\eta\lambda D/B)$, nearly constant: 3.3–3.8 at every batch size. Combine with our earlier observation in problem2 that at a fixed token budget D, good LR–WD pairs approximately preserve the product $\eta\lambda$, this further suggests a broader principle: **for a fixed training budget, the optimal cumulative weight-decay shrinkage is approximately invariant.**

|  | Fit over B = 8–64 | Prediction at B=128 | Prediction at B=256 |
| --- | --- | --- | --- |
| **(i)** WD .1, scale the LR | LR $\propto$ B^0.57 | LR **0.0054** | LR **0.0080** |
| **(ii)** LR .0015, scale the WD | WD $\propto$ B^**0.98** | WD **0.95** | WD **1.88** |

!image.png

- **Hypothesis (ii)—scaling WD while fixing a stable LR—wins**
- **The reasons why (i) fails and cause loss overshoot by very far**
    - The optimal LR stops growing as the curvature stability point is reached
    - WD=0.1 is under-decayed at large B.
- Best loss vs B for hypothesis (B)—perfect down-and-up
    - 2.9337 (B=8), 2.9238 (16), 2.9189 (32), 2.9178 (64), 2.9319 (128), 2.9684 (256). Gains flatten by B=64 and loss rises after that, suggesting that we are past the critical BS here.

| B | Hypothesis | Predicted | Fitted optimum | Predicted ÷ fitted | Loss at prediction minus best at this B |
| --- | --- | --- | --- | --- | --- |
| 128 | (i) | LR .0054 | ≈ .0035 (fitted) | 1.56 | +0.028 |
| 128 | (ii) | WD .95 | .938 | 1.01 | 0 |
| 256 | (i) | LR .008 | ≤ .004 | ≥ 2 | +0.161 |
| 256 | (ii) | WD 1.9 | 1.80 | 1.05 | 0 |

!image.png

### c-ablate momentum

- **Most findings are consistent with NQM**
    - Momentum helps at both batch sizes, with the impact on the large batch size much more pronounced
    - Too small and too large are both degenerating. But the degeneration caused by a too large momentum is not as severe as NQM

!image.png

### d-What does NQM explain

|  | NQM prediction | LM measurement | Verdict |
| --- | --- | --- | --- |
| **LR vs B** (Adam, noise-dominated) | sqrt law when noise dominates | Nonlinear, lr scaling fattens as B grows.  | **Fails.** LM has more complex curvature that Adam’s diagonal preconditioning cannot fully neutralize as in NQM, causing the optimal LR to approach a stability limit. |
| **Best loss vs B** (fixed tokens) | Can only get worse as B grows (fewer updates, same total noise) | improve initially but then degrade | **Partly holds**: the rise after B=64 agrees; the improvement from 8 to 64 contradicts it. The reason behind the initial improvement is unclear to me |
| **WD vs B** at fixed LR | The NQM has no WD. Its step-counting logic (T = N/B) implies the optimal cumulative decay is invariant | WD and B are almost proportional (i.e., linear) | **Holds** |
| **Momentum benefit vs B** | Grows with B | also grows with B | **Holds** in direction |
| **Best β1 vs B** | Falls as B increase: .95 → .8 | Also slightly falls .98 → .95 | **Holds** in direction |
| **β1 above the optimum √β2 ≈ .975** | Catastrophic: β1 = .99 is ×13 worse (B=16) | B=8: .98 is the best (2.9248) and .99 costs only +0.006 | **Fails**, because of the NQM's short run. |
- The disagreement in the last row (the catastrophic effect when $\beta_1$ goes beyond $\sqrt{\beta_2}$ in NQM but not in LM), is cause by out NQM’s being short horizon. When I increase DQM training steps by 8x, β1=0.99 becomes the best setting.

### Example Question (Hard)

I think doubling WD would achieve the best loss. This maintains the cumulative decay while also maintaining the optimal LR.

Doubling LR also maintains the cumulative decay, but deviates from the best LR. The outcome should match those when we compensate LR WD for each other to maintain the optimal product. When we compensate, the alternative config is acceptable but not perfect.

The other options deviate from the optimal cumulative decay and should for sure have worse loss

## Problem 4.0

No (readout param, readout feature) alignment

- read out now need a+b=1/2 for readout logit to be $\theta(1)$, and a+c=1 for weight update to be $\theta(1)$. abc=(1,-1/2, 0).

No (param update, input) alignment

- now we need a+c=1/2 to have $\theta(1)$ weight update. So
- hidden now need a+c=1/2 and a+b=1/2, which yield abc=(0, 1/2, 1/2)
- read out need a+c=1/2 and a+b=1, which yield abc=(1/2, 1/2, 0)

Both have no alignment

- read out need a+c=1/2 and a+b=1/2, which yield abc=(1/2, 0, 0)

### ABC choices

| Combination (α, ω_move) | Input | Hidden | Readout | Initial logit RMS n^(1/2−a−b) |
| --- | --- | --- | --- | --- |
| 1: (1, 1), μP | (0, 0, 0) | (0, 1/2, **1**) | (**1**, **0**, 0) | n^(−1/2), vanishes |
| 2: (1, 1/2) | (0, 0, 0) | (0, 1/2, **1**) | (**1**, **−1/2**, 0) | Θ(1) |
| 3: (1/2, 1) | (0, 0, 0) | (0, 1/2, **1/2**) | (**1/2**, **1/2**, 0) | n^(−1/2), vanishes |
| 4: (1/2, 1/2) | (0, 0, 0) | (0, 1/2, **1/2**) | (**1/2**, **0**, 0) | Θ(1) |

## Scaling rules

| Combination $(\alpha, \omega_{\mathrm{move}})$ | Input: multiplier, init variance, LR | Hidden: multiplier, init variance, LR | Readout: multiplier, init variance, LR |
| --- | --- | --- | --- |
| **1:** $(1, 1)$ | $1,\ \frac{1}{d},\ \eta$ | $1,\ \frac{1}{k},\ \frac{\eta}{m}$ | $\frac{1}{m},\ \frac{1}{n_0},\ \eta$ |
| **2:** $(1, \tfrac{1}{2})$ | $1,\ \frac{1}{d},\ \eta$ | $1,\ \frac{1}{k},\ \frac{\eta}{m}$ | $\frac{1}{m},\ \frac{m}{n_0},\ \eta$ |
| **3:** $(\tfrac{1}{2}, 1)$ | $1,\ \frac{1}{d},\ \eta$ | $1,\ \frac{1}{k},\ \frac{\eta}{\sqrt{m}}$ | $\frac{1}{\sqrt{m}},\ \frac{1}{m n_0}=\frac{1}{n},\ \eta$ |
| **4:** $(\tfrac{1}{2}, \tfrac{1}{2})$ | $1,\ \frac{1}{d},\ \eta$ | $1,\ \frac{1}{k},\ \frac{\eta}{\sqrt{m}}$ | $\frac{1}{\sqrt{m}},\ \frac{1}{n_0},\ \eta$ |

## Problem 4.1

### a-MuP transfers better and also achieves a lower loss

!image.png

### b-probe width scaling

- **Feature learning—μP stays width-invariant, while Kaiming weakens with width**
    - Under μP, final-feature movement is almost identical across widths: $M_5 \approx 1.12, 1.02, 1.02$ for widths $640, 2560, 5120$. Under Kaiming it falls from $1.14$ to $0.54$, suggesting that the smaller tuned LR required at large width suppresses feature learning.
- **Logit scaling—matches the μP prediction**
    - μP's initial logits shrink approximately as $n^{-1/2}$: $0.89 \rightarrow 0.45 \rightarrow 0.32$, consistent with $a+b=1$. After training, logits return to the same $O(1)$ scale across widths. Kaiming instead starts with $O(1)$ logits, as expected from $a+b=1/2$.
- **Weight-update alignment—cross-width scaling is close to full alignment under μP**
    - Although the single-width effective exponents are only around $0.7$, fitting $R=\Theta(n^\alpha)$ across widths gives $\alpha\approx0.93\!-\!1.07$ for most μP hidden matrices and $\alpha\approx0.94\!-\!1.00$ for the readout. Thus the update alignment scales approximately like the full-alignment assumption $\alpha=1$.
    - Kaiming is much less interpretable across width because its tuned LR changes by roughly $20\times$, so the prefactor in $R$ is not held fixed.
- **Readout–feature alignment—close to no alignment, contradicting the full-alignment assumption**
    - $\omega_{\mathrm{move}}$ stays around $0.5\!-\!0.6$ under both parameterizations, indicating that the initial readout acts on learned feature movement roughly like an independent random matrix. This disagrees with the $\omega_{\mathrm{move}}=1$ assumption used in the standard μP readout derivation.
- **Overall—μP's update-alignment and feature-learning predictions work, but its readout-alignment assumption does not**
    - The experiments support $\alpha\approx1$, width-independent $O(1)$ feature movement, and $n^{-1/2}$ initial-logit scaling. The main mismatch is the readout–feature-change alignment, which empirically behaves much closer to $\omega_{\mathrm{move}}=1/2$ than $1$.

!image.png

!image.png

### c-transfer across depths

- **Transfer:** Depth-μP and CompleteP keep the source LR to within 5% out to r = 500. Standard μP drifts down about 24%, far less than the 1/r or 1/√r shift its unscaled residual branches would naively suggest.
- **Better transfer DOES NOT mean lower loss, but means less degradation.** Depth-μP and CompleteP transfer equally well, their loss stays nearly flat or even lightly grow as we scale depth. Standard μP transfers worse and pays about 0.21 already by depth 100.

!image.png

### d-probe depth scaling

- **Residual-stream scaling—standard μP blows up with depth, while both depth-scaled prescriptions remain bounded**
    - At initialization, standard μP grows approximately as $\sqrt{r}$: the final pre-norm RMS increases from $1.34$ at depth 2 to $9.4$ and $33$ at depths 100 and 1,000. After five updates it grows much more strongly, reaching $797$ and $7{,}334$, consistent with increasingly coherent accumulation across blocks.
    - Depth-μP and CompleteP instead keep the pre-norm residual stream at $O(1)$ with depth: roughly $5\!-\!8$ after five updates, compared with $9.8$ at the depth-2 source.
- **Normalization—can completely hide residual-stream blow-up**
    - The final RMSNorm output remains around $0.95\!-\!0.97$ for every prescription and depth, even when standard μP's pre-norm residual reaches $7{,}334$.
    - Likewise, final normalized feature movement remains $\Theta(1)$ everywhere, with $M_5 \approx 1.09\!-\!1.26$. Thus normalized activations alone do not reveal whether the underlying residual stream is stable.
- **Per-block feature learning—Depth-μP becomes increasingly lazy, while CompleteP preserves source dynamics**
    - Depth-μP keeps total feature movement nonvanishing, but its last-block FFN branch barely changes at large depth: RMS $0.54\!-\!0.76$ after five updates versus $4.6$ at the source. Its $r^{-1/2}$ block LR therefore suppresses learning inside each individual block.
    - CompleteP keeps the same branch dynamics as the source: $4.5\!-\!5.4$ versus $4.6$. Its unscaled block LR preserves $O(1)$ per-block learning, while the $1/r$ residual multiplier keeps the global sum bounded.
- **LR transfer vs. loss—bounded global signals are enough for transfer, but good loss also requires nonvanishing per-block learning**
    - Depth-μP and CompleteP both bound the residual stream and both transfer LR well. Standard μP only approximately transfers because normalization controls what each block sees even though the underlying residual stream grows strongly.
    - CompleteP also gives the best depth scaling of loss because it preserves both bounded global dynamics and source-like per-block learning. Depth-μP transfers LR but loses performance as its blocks become lazy, while standard μP suffers from distorted early dynamics caused by residual growth.
- **Readout–feature alignment—unchanged with depth and close to no alignment**
    - $\omega_{\mathrm{move}}\approx0.50$ across all depths, updates, and prescriptions. As in the width experiments, the **initial** readout acts on learned feature movement approximately like an independent random matrix.

## Problem 4.2

### a-transfer the reference LR

- **Update alignment—close to full alignment early in training, but weakens substantially later**
    - Under μP, the cross-width exponent is around $\alpha \approx 1$ during the first few updates and remains near full alignment through warmup. After warmup it steadily decreases, ending around $0.75\!-\!0.8$ for most matrices and around $0.5\!-\!0.6$ for the attention output and FFN down-projection.
    - If the late-training value $\alpha \approx 0.75$ dominated optimization, the $\eta/m$ rule would make feature updates shrink roughly as $n^{-0.25}$ at larger width, suggesting the optimal LR should drift upward. Empirically, this does not happen over widths $128$ to $1{,}024$: the fitted optima remain around $0.002\!-\!0.003$. This suggests that getting the early-training scaling right is sufficient for good LR transfer, even though the alignment assumptions become less accurate later.
- **Per-width alignment—peaks near warmup and then decays**
    - The single-width effective alignment rises from roughly $0.65\!-\!0.85$ in the first few updates to $0.85\!-\!0.97$ near the end of warmup, then falls to about $0.55\!-\!0.65$ by the end of training. The layers behave similarly, with relatively narrow variation across depth.
- **Readout–feature alignment—remains at no-alignment throughout training**
    - $\omega_{\mathrm{move}} \approx 0.50$ at every width and throughout training under both μP and the baseline.

!image.png

!image.png

### b-compare transfer and tuned performance

- **LR transfer—μP keeps the optimum nearly fixed, while the baseline follows the expected width dependence**
    - The baseline optimum shifts strongly with width: $0.0081 \rightarrow 0.0040 \rightarrow 0.0024$ from widths $128 \rightarrow 256 \rightarrow 512$, fitting roughly $\mathrm{LR}^\star \propto n^{-0.88}$. The power of $-0.88$ roughly matches the $\eta/m$scaling of hidden layer LR that the basline missed. Transferring the source LR $0.003$ costs as much as $0.071$ loss at width $128$.
    - Under μP, the fitted optima stay in a narrow range, $0.0030, 0.0020, 0.0024$, with no clear width trend. The worst transfer penalty of using $0.003$ is only $0.01$.
- **Tuned performance—better LR transfer does not imply better final loss**
    - After independently tuning LR, μP and the baseline reach essentially the same loss: $3.627$ vs $3.629$ at width $128$ and $3.407$ vs $3.408$ at width $256$.
- **Baseline scaling—long training recovers the expected $1/m$ like behavior**
    - The baseline LR shift is much milder than in the five-update stress test: about $m^{-0.88}$ here versus $m^{-1.5}$ in 4.1(a). Warmup, clipping, and LR decay remove the severe early instability that previously forced a steeper LR reduction.

| Width (m) | Baseline LR* | Baseline best measured loss (at LR) | Baseline loss at .003 | μP LR* | μP best measured loss (at LR) | μP loss at .003 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 (¼) | .0081 | 3.627 (.006) | 3.692 (+0.065) | .0030 | 3.629 (.003) | 3.629 (+0.000) |
| 256 (½) | .0040 | 3.407 (.003) | 3.407 (+0.000) | .0020 | 3.408 (.0015) | 3.418 (+0.010) |
| 512 (1) | .0024 | 3.229 (.003) | 3.229 (+0.000) | .0024 | 3.229 (.003) | 3.229 (+0.000) |

!image.png

### c-predict a held-out width

- **Fitting helps the baseline a lot but not MuP**
- Baseline’s fitting prediction is more accurate, but MuP has a wider valley of good LR
- The tuned performance are still quite close, with MuP being 0.005 smaller (1.66x seed noise)

|  | Power law (fit on widths 128–512) | Prediction ÷ fitted optimum | Gap at prediction | Gap at .003 | Best Loss Measured |
| --- | --- | --- | --- | --- | --- |
| Baseline (SP) | 0.562 · n^(−0.88) → .00126 | 1.07 | **+0.0000** | **+0.142** | 3.1041 |
| μP | 0.00588 · n^(−0.16) → .00193 | 0.81 | +0.0025 | +0.0040 | 3.0992 |

!image.png

### d-transfer across depth

- **Transfer: neither correction helps consistently**
    - Depth-MuP transfers perfectly to depth 16, but only because the default 0.003 LR collides with Depth-MuP’s optimal LR that grows with depth.
    - CompleteP leaves the block LR unscaled, so it transfers about as well as standard μP: perfect at depth 4, +0.012 vs +0.020 at depth 16.
- Tuned Performance: neither improve, Depth-MuP slight worsen at depth 4

!image.png

### e-identify the training regime

- **Early-training finite-width effects—clipping and embedding gradients violate the clean μP assumptions**
    - During the first few updates, the embedding contributes $90\!-\!99\%$ of the squared gradient norm and clipping is severe, with coefficients as low as $0.03\!-\!0.26$. Because the clipping factor changes during warmup, it reweights Adam's early moment estimates differently across widths.
    - These effects disappear quickly: by update $23$ the embedding contribution is negligible, and clipping is inactive by the end of training.
- **Feature learning—nonvanishing everywhere, but not sufficient to identify a good LR**
    - Normalized hidden features move by $\Theta(1)$ under every width, depth, prescription, and LR tested. Initial μP logits follow the expected $m^{-1/2}$ scaling, but by the end all runs reach similar logit scales.
    - Thus normalized feature movement confirms feature learning, but does not distinguish well-tuned runs from unstable ones.
- **Residual-stream transient—the clearest diagnostic of an excessively large LR**
    - Bad transferred LRs produce large pre-norm residual spikes during warmup: SP at width $1{,}024$ reaches $628$ versus $71$ at its tuned LR, while standard μP at depth $16$ reaches $180$ versus $35$.
    - RMSNorm largely hides these transients from normalized activations. Moderate residual growth can still train well, but very large peaks—roughly $6\!-\!11\times$ the reference—consistently accompany worse final loss.
- **Alignment—full early in training, partial later; readout alignment remains absent**
    - Update alignment is close to full during warmup, with cross-width $\alpha \approx 1$, but decays later to roughly $0.5\!-\!0.8$. The pattern depends mainly on training stage rather than width, depth, or parameterization.
    - Readout–feature alignment stays at $\omega_{\mathrm{move}}\approx0.5$ throughout, again contradicting the full-alignment readout assumption.
- **Optimal-LR shift—not by itself evidence of leaving the stable μP regime**
    - The baseline's width-dependent optimum, Depth-μP's depth-dependent shift, and μP's small width-256 dip can all occur while feature movement, residual scale, clipping, and alignment remain healthy.
- **Fixed weight decay—confounds the measured LR-scaling trends**
    - AdamW decay depends on $\eta\lambda$, so changing the effective LR across width or depth also changes the cumulative decay timescale when $\lambda$ is held fixed. All runs are increasingly under-decayed at larger width, and μP's $\eta/m$ hidden LR strengthens this effect.
    - Because the loss valley approximately follows constant $\eta\lambda$, fixed WD can push the apparent optimal LR upward and partially mask the intrinsic LR-transfer trend. A cleaner μP test would scale hidden-layer WD so that $\eta\lambda T$ remains approximately constant.