# Problem 4: a better update across time — parts (a)–(c)

Figures are in `experiments/a4_stability/plots/p4/`. Numbers come from `results/a4_stability/p4/summary.json`
(`plot_p4.py`), the outcomes from `p4_outcomes.py`, and the checkpoint probes from `p4_probe.py`.

**Setup.** All models are the modern 4-layer RNN from Problem 2. Its depth residuals, LayerNorm pre-norm and
SwiGLU MLPs are unchanged, and so is the shared recipe (width 512, vocab 4096, context 1024, batch 64, 614.4M
tokens, AdamW with WD 0.1, 1% warmup then linear decay, clipping at 1, seed 42). Only the recurrent update
h_t = φ(h_{t-1}, x_t) changes, and every variant carries only h across time. The tanh baseline is the supplied
`a4-rnn` sweep. Probes use P1's 8 fixed held-out sequences in FP32, at initialization (seed 42), step 100 and
final. Each model is probed at its best LR. "dL_T/dh_t" is the RMS of the last-token loss's gradient with
respect to each layer's state at time t. Ratios between gradients are averaged over the 4 layers in log space.
"Context gap at k" is the last-token loss with only BOS plus the last k tokens, minus the full-context loss, on
64 validation rows. Without BOS, truncated inputs push the tanh RNN far off-distribution (+50 nats), as in P2,
so that version is shown only in the right panel of `context.png`.

## Results

| update φ | best val loss | Δ vs GRU | Δ vs tanh | LRs (val loss) | params |
|---|---|---|---|---|---|
| tanh RNN (supplied) | 3.516 | +0.326 | — | 0.001: 3.523, **0.002: 3.516**, 0.004: 3.681 (+2 lower) | 17.32M |
| **GRU** | **3.190** | — | −0.326 | 0.001: 3.273, **0.003: 3.190**, 0.009: 3.438 | 21.52M |
| GRU, gates see x_t only | 3.233 | +0.043 | −0.283 | 0.003 only | 19.42M |
| GRU, no reset (r = 1) | 3.284 | +0.094 | −0.232 | 0.003 only | 19.42M |
| GRU, fixed update (z = 0.5) | 3.383 | +0.193 | −0.133 | 0.003 only | 19.42M |
| GRU, no update (z = 1, no carry) | 3.416 | +0.226 | −0.100 | 0.003 only | 19.42M |
| (b) pre-norm GRU | 3.205 | +0.015 | −0.311 | 0.003: 3.211, **0.006: 3.205** | 21.52M |
| **(b) pre-norm GRU, no tanh** | **3.188** | **−0.002** | **−0.328** | **0.003: 3.188**, 0.006: 3.195 | 21.52M |

The ablations were trained only at the GRU's best LR, 0.003. Their 0.001 and 0.009 attempts crashed on
infrastructure: a reused Modal container hit torch.compile's recompile limit. These are not training
outcomes and are excluded. Tuning can only lower an ablation's loss, so each "Δ vs GRU" for an ablation is an
upper bound on what that component is worth. The no-reset run resumed from a mid-training checkpoint after
one of those crashes. All results are single-seed.

## (a) Tuned GRU and ablations: which recurrent components help?

**The GRU beats the tuned tanh RNN by 0.326 nats** (3.190 vs 3.516). Its LR optimum (0.003) sits inside its
grid. Each ablation removes one component and keeps the rest:

- **Update gate is the most important component.** Removing it (z = 1, so h_t = tanh(W x + U(r⊙h))) costs
  0.226. Replacing it with a constant carry (z = 0.5) still costs 0.193. A fixed leaky carry recovers only
  0.033 of the gap, so the gain comes from the carry being *data-dependent*, not from carrying h at all.
- **Reset gate is worth 0.094.**
- **Gates that see h_{t-1} add only 0.043.** Computing r and z from x_t alone (only the candidate reads
  h_{t-1}) keeps most of the GRU's gain. Most of the gating decision is driven by the current token.
- **Parameters don't explain it.** The four ablations have identical parameter counts (19.42M), yet span
  3.233–3.416. The GRU has 24% more parameters than the tanh RNN, but z = 1, which has 12% more than the tanh
  RNN, is still 0.10 better than tanh. A 0.18 spread between equal-size models comes from the update rule.

**Checkpoint diagnostics** (`time.png`, `stages.png`, `gates.png`, `context.png`):

1. **The update gate makes the temporal gradient survive.** Log10 of dL_T/dh_{T-lag} / dL_T/dh_T at the
   final checkpoint:

   | | lag 16 | lag 64 | lag 256 | lag 1022 |
   |---|---|---|---|---|
   | tanh RNN | −3.8 | −12.5 | −16.0 | −22.2 |
   | GRU, z = 1 | −1.1 | −2.1 | −5.6 | −17.6 |
   | GRU, z = 0.5 | −1.0 | −1.7 | −3.4 | −10.4 |
   | GRU, x-only gates | −0.6 | −0.9 | −1.4 | −2.5 |
   | GRU | −0.6 | −0.8 | −1.2 | −2.4 |
   | GRU, r = 1 | −0.6 | −0.8 | −1.0 | −1.5 |

   Every model with a learned z loses only ~2 decades over the whole 1022-token window. Without it, the
   gradient falls 10–22 decades. For the tanh RNN the collapse is concentrated in the top two layers: −18.7 and
   −27.7 decades at lag 64, against −1.7 and −1.9 in layers 1–2. `stages.png` shows that learning creates this
   reach. At init the GRU's gradient already underflows (< 1e-30) beyond ~250 tokens. By step 100 it spans the
   whole sequence within ~2 decades. With z = 1 it is still dead beyond ~150 tokens at step 100.

2. **The gates learn a depth-ordered memory hierarchy** (`gates.png`). The mean update gate in the GRU falls
   with depth: z̄ = 0.73, 0.60, 0.40, 0.27 for layers 1–4. Median per-channel timescales 1/z̄ are 1.4, 1.5, 3.8
   and 61 tokens. In layer 4, 59% of channels have timescales > 20 tokens and 36% > 100. The bottom layers
   behave like nearly memoryless token features, and the top layers integrate over long spans. With x-only
   gates the median channel stays fast (~1.7 tokens) in every layer, but a long-memory tail remains (layer 4:
   31% > 20 tokens). That is enough to keep the gradient reach (row above) and most of the loss.

3. **The reset gate prevents saturation.** In the GRU, r is mostly closed (mean 0.15–0.19 per layer). The
   candidate reads only a small fraction of the old state, so its tanh rarely saturates:
   Pr(|h| > .95) = 0.008. With r = 1 the full U·h_{t-1} enters the tanh, and saturation jumps to **0.40, the
   same as the tanh RNN (0.40)**. Its gradient still reaches far, because the carry (1−z)h bypasses the tanh.
   So the reset gate's 0.094 is not about gradient flow. It keeps the candidate in a regime where it can write
   graded, non-binary values.

4. **Context use follows the carry.** Context gaps at k = 64 / 256 (± s.e., 64 rows; the last-token metric
   is noisy):

   | | k = 64 | k = 256 |
   |---|---|---|
   | tanh RNN | −0.003 ± 0.016 | 0.000 ± 0.000 |
   | GRU, z = 1 | 0.022 ± 0.020 | −0.001 ± 0.002 |
   | GRU, z = 0.5 | 0.015 ± 0.028 | 0.004 ± 0.004 |
   | GRU | 0.102 ± 0.050 | 0.033 ± 0.018 |
   | GRU, x-only gates | 0.091 ± 0.061 | 0.029 ± 0.019 |
   | GRU, r = 1 | 0.116 ± 0.056 | 0.027 ± 0.015 |

   The tanh RNN and both carry-less or constant-carry variants make the same prediction from 64 tokens as from
   1023. Every model with a learned update gate still uses tokens more than 64 (and, weakly, more than 256)
   back. This matches the gradient-reach table: large gradient reach comes with measurable use of distant
   context.

## (b) A new update: pre-norm across time

**Proposal.** Problem 2 moved the transformer's depth tricks into the RNN's depth stack, but the temporal
update is still a raw recurrence. We apply the same pre-norm rule across time. Every recurrent map reads a
normalized state N(h_{t-1}), with a new LayerNorm per layer, while the carry stays raw, exactly as each depth
branch reads N(x) while the residual stream passes through untouched:

    n = LayerNorm(h_{t-1});  z = σ(W_z x + U_z n + b_z);  r = σ(W_r x + U_r n + b_r)
    pre-norm GRU:            h_t = (1 − z)⊙h_{t-1} + z⊙tanh(W x + U(r⊙n) + b)
    pre-norm GRU, no tanh:   h_t = (1 − z)⊙h_{t-1} + z⊙(W x + U(r⊙n) + b)

The no-tanh version tests a prediction from (a): once the recurrent read is normalized, the candidate no
longer needs a squashing nonlinearity to stay bounded. h is a convex combination of candidates whose recurrent
part has fixed scale. Only h is carried, and the change adds 4,096 parameters (four LayerNorms).

**Results.** The no-tanh version reaches **3.188 at LR 0.003: 0.328 better than the tuned tanh baseline and
0.002 better than the tuned GRU.** At LR 0.006 it reaches 3.195, only 0.007 worse, whereas the GRU degrades from
3.190 at 0.003 to 3.438 at 0.009. The proposal's LR curve looks flatter at the high end, but there is no GRU
run at 0.006 to compare directly. 0.002 is well within single-seed noise, so the honest claim is
**parity with the GRU, not a clear win**. The pre-norm GRU with tanh is slightly worse than the GRU (3.205,
best at 0.006). Removing the tanh improves it by 0.023 at 0.003 and 0.010 at 0.006, the only consistent
ordering across both LRs.

**Comparison with the tanh baseline and the GRU (diagnostics).**
- **Longest gradient reach of all models.** The no-tanh version loses only 0.4 decades at lag 64, 0.6 at
  lag 256 and 0.9 across the whole window. The GRU loses 0.8, 1.2 and 2.4. At lag 64, layers 2–4 lose at most
  0.3 decades (GRU layer 2: 0.8).
- **Longest memory.** It has the slowest update gates: median channel timescales 79 and 76 tokens in layers
  3–4 (GRU: 4 and 61), with 47% / 43% of those channels over 100 tokens.
- **Context use is no larger than the GRU's.** Its context gaps (0.122 ± 0.078 at k = 64, 0.028 ± 0.022 at
  k = 256) are indistinguishable from the GRU's. Stronger gradient reach did not turn into more measured use of
  distant tokens, which fits the loss being at parity rather than better.

## (c) What the diagnostics explain, and how they guided the design

- **Why the GRU beats tanh.** The tanh RNN's top layers lose 19–28 decades of gradient within 64 tokens, and
  40% of its state sits in saturation. It neither uses nor can learn from context beyond ~64 tokens. A
  data-dependent carry fixes both. Gradients survive through (1−z)h independently of the tanh, and the gates
  let upper layers specialize into slow channels (timescales of 60+ tokens) while lower layers stay fast.
  Constant carries (z = 0.5) give some gradient reach (−3.4 decades at lag 256) but no measurable long-range
  context use, and recover only 0.033 of the loss. Data-dependent *selection* is what matters.
- **What the reset gate does.** r = 1 has GRU-like gradient reach but tanh-RNN-like saturation (0.40) and
  loses 0.094. The reset gate's job is scale control of the candidate's recurrent input. That suggested
  doing the scale control explicitly, by normalizing the recurrent read, which is the (b) proposal.
- **Why the pre-norm GRU with tanh underperforms, and the fix.** Normalizing h to unit RMS before U
  *raises* the candidate's input scale: GRU states have RMS ≈ 0.47. Saturation rises from 0.008 (GRU) to
  0.21. This is the r = 1 failure mode again, from a different cause. Dropping the tanh removes the saturating
  nonlinearity while normalization keeps the update bounded. State RMS settles at ≈ 0.80, with no divergence
  at either LR, and the loss improves at both LRs.
- **Limits of the evidence.** The headline gain over the GRU (−0.002) is within single-seed noise. Ablations
  were run at one LR. The context gaps have s.e. comparable to their size at k = 256. The direction suggested by
  the diagnostics is that the remaining gap is not gradient reach: the no-tanh model already has the flattest
  temporal gradients but no extra context use. Further gains are more likely from what the slow channels
  store, given a fixed 512-wide state, than from making gradients travel further.
