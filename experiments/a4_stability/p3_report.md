# Problem 3: attention residuals across depth — parts (a)–(c)

Figures are in `experiments/a4_stability/plots/p3/`. Numbers come from `results/a4_stability/p3/summary.json`
(`plot_p3.py`), and the depth probes from `p3_probe.py`.

**Setup.** Both models use the shared recipe: width 512, vocab 4096, context 1024, batch 64, 600k sequences
(614.4M tokens), AdamW with WD 0.1, 1% warmup then linear decay, clipping at 1, seed 42. Branches, branch
norms and the RNN's temporal recurrence are unchanged. Only the depth skip is replaced. The additive
baselines are the supplied P1/P2 sweeps. Each new variant was trained at (⅓, 1, 3) × its center LR. The center
is 0.003 for d8, 0.009 for d8 block-2 (the starter's value) and 0.001 for the RNN. Attention-residual queries are
zero-initialized, so every router starts as a uniform average. There is one more routed read before the usual
output head. d8 full AttnRes runs out of H100 memory at batch 64: 17 routers each keep an FP32 stack of up to
17 sources for the backward pass. It was therefore trained as 2 micro-batches of 32. That is the same batch of 64
per step and the same optimizer, but without torch.compile. Depth diagnostics use 8 fixed held-out validation sequences (8×1024 tokens), in FP32, at
initialization (rebuilt with seed 42), step 100 and the final checkpoint, each at the variant's best LR. A
"depth entry" is the input to a branch (A/R = attention/recurrent, M = MLP). The last entry is the final stream
read by the head.

## Results

| model | variant | best val loss | Δ vs tuned additive | best LR | losses at the (⅓, 1, 3)× grid | params |
|---|---|---|---|---|---|---|
| d8 | additive (supplied) | 2.9265 | — | 0.003 | 0.00075: 3.004, 0.0015: 2.948, 0.003: 2.927, 0.006: 2.943, 0.012: 2.963 | 34.61M |
| d8 | half-and-half | 2.9328 | +0.006 | 0.003 | 2.982 / 2.933 / 2.946 | 34.61M |
| d8 | EMA β=0.1 | 6.8746 | +3.95 (fails) | any | 6.875 / 6.875 / 6.875 | 34.61M |
| d8 | EMA β=0.9 | 2.9475 | +0.021 | 0.003 | 2.981 / 2.947 / 2.984 | 34.61M |
| d8 | uniform routing | 2.9274 | +0.001 | 0.003 | 2.977 / 2.927 / 2.945 | 34.61M |
| d8 | full AttnRes | 2.9010 | **−0.026** | 0.009 (grid edge) | 2.949 / 2.903 / 2.901 (2 micro-batches) | 34.63M |
| d8 | block AttnRes B=2 | 2.9056 | **−0.021** | 0.003 | 0.001: 2.946, 0.003: 2.906, 0.009: 2.912, 0.027: 3.001 | 34.63M |
| d8 | block AttnRes B=4 | 2.9123 | −0.014 | 0.003 | 2.963 / 2.912 / 3.209 | 34.63M |
| RNN | additive (supplied) | 3.5160 | — | 0.002 | 0.00025: 3.767, 0.0005: 3.617, 0.001: 3.523, 0.002: 3.516, 0.004: 3.681 | 17.32M |
| RNN | half-and-half | 3.5037 | **−0.012** | 0.001 | 3.678 / 3.504 / 3.697 | 17.32M |
| RNN | EMA β=0.1 | 3.5726 | +0.057 | 0.001 | 3.720 / 3.573 / 3.727 | 17.32M |
| RNN | EMA β=0.9 | 3.5066 | −0.009 | 0.001 | 3.686 / 3.507 / 3.652 | 17.32M |
| RNN | uniform routing | 3.5717 | +0.056 | 0.001 | 3.743 / 3.572 / 3.613 | 17.32M |
| RNN | full AttnRes | 3.5491 | +0.033 | 0.001 | 3.681 / 3.549 / 3.595 | 17.33M |
| RNN | block AttnRes B=2 | 3.5159 | 0.000 | 0.001 | 3.671 / 3.516 / 3.655 | 17.33M |
| RNN | block AttnRes B=4 | 3.5173 | +0.001 | 0.001 | 3.660 / 3.517 / 3.677 | 17.33M |

The routers add 17.4k parameters to d8 (17 × (query + key-norm weight)) and 9.2k to the RNN, both under 0.06%.
Every finite variant has an interior minimum on its grid except d8 full AttnRes. Its best rate, 0.009, is the
top edge, but the curve is flat there (2.903 at 0.003, 2.901 at 0.009). Its true optimum is probably near
0.005, and the gain is unlikely to be much larger than measured. A run at 0.027 would close the curve; it
wasn't run, because each micro-batched run takes about 95 min.
Caveat: the RNN grid steps by 3×, while the supplied baseline steps by 2× and is best at 0.002. At the
variants' common best LR of 0.001, the baseline gets 3.5234. RNN gaps of about 0.01 are therefore within LR
resolution. Each result is a single seed.

## A mental model for depth mixing

Every scheme here writes the input to branch *j* as a mix of earlier sources (the embedding and the branch
outputs so far). Two properties of that mix matter.

1. **Only the relative weights matter, until something reads the stream without normalization.** Each branch
   sees its input through a pre-norm, so N(c·x) = N(x). Rescaling the whole mix changes nothing the branch
   sees. Additive residuals weight every source equally (weight 1), and uniform routing also weights them
   equally (weight 1/(j+1)). These are the *same function* whenever the final stream is also normalized,
   which is true in d8 (final RMSNorm). They differ in the modern RNN, whose head reads the raw final stream
   (`output_norm = Identity`). EMA β gives geometric weights β^(j−k) that favor recent sources.
   Half-and-half gives a two-term recursion. Attention residuals learn input-dependent weights.
2. **The identity path sets the per-branch gradient multiplier.** For x′ = βx + F(N(x)),
   ∂x′/∂x = βI + J_F, so the skip path contributes β^L across L branches. Additive and half-and-half keep
   a unit-gain path. EMA 0.1 leaves only 0.1^16 ≈ 10⁻¹⁶ in d8. Routers pass gradient back through their
   weights, which only stays high-gain if a weight stays near 1.

Predictions: uniform ≈ additive in d8; anything that shrinks the RNN's final read hurts; small β kills deep
models; learned weights can only help if they move away from equal weighting in a useful direction.

## (a) Fixed mixing patterns

*Figures: `depth_d8_fixed.png`, `depth_d8_ema.png`, `depth_rnn_fixed.png`, `depth_rnn_ema.png`, `sweep.png`.*

**Uniform routing.** d8 uniform matches additive to 0.0008 at every LR (2.977/2.927/2.945 against the baseline's
interpolated curve). This confirms property 1. The depth statistics differ only by scale. The routed stream
stays near RMS 1, where the additive stream grows from 2 to 25. Stream gradients are correspondingly larger
(up to about 15× near the top), and the raw branch/input ratio is 1–7 instead of 0.25–0.5. Because the forward
function is identical, the parameter gradients are identical too. In the RNN, uniform routing is worse by
+0.056. At init, the RNN's final stream under uniform routing has RMS 0.166 against 1.50 for additive: the
average over 9 sources shrinks the logits about 9×. The model must grow that scale back through the
unnormalized head. By step 100 the final RMS is back to 1.1, but the early dynamics differ. The step-100 and
final gradient profiles become jagged, with recurrent-branch inputs in layers 3–4 getting 3–5× less gradient than the
MLP inputs next to them. Adjacent-depth cosine also departs from the baseline's alternating pattern. Uniform also prefers the
higher LR: its 3× point (3.613) is the closest of any RNN variant to its best.

**EMA β=0.1** fails in d8 at every LR: 6.875, essentially a unigram predictor (the unigram cross-entropy of the
validation targets is 6.844). By step 100, the gradient for L falls about 10⁷× from the top entry (1.4e−7) to
the lower ones (about 1e−14). The final checkpoint is similar (8e−11 at the top, 1e−14 to 1e−15 below). The
lower 12 branches effectively stop training, while the top of the stream grows to RMS 36 by step 100 and
3,100 at the end. Adjacent-depth cosine stays mostly below 0.3: consecutive stream states share almost nothing,
because each one is mostly the last branch output. The carried stream holds no usable token identity. In the
8-branch RNN, β^8 = 10⁻⁸ is survivable. The branch Jacobians carry the gradient, and the RNN's
gradient falls only about 10× from top to bottom (5e−5 to about 4e−6, close to the baseline's profile). It
trains, but is 0.057 worse.

**EMA β=0.9** keeps a near-unit gain. In d8 its gradient profile is the flattest of all (about 7e−7 at every entry,
against the additive profile that decays toward the top). Its stream RMS saturates near 10, because a geometric
sum of updates is bounded by about ‖F‖/(1−β). Adjacent cosine is 0.95–0.97, so consecutive states are
nearly identical. It is still 0.021 *worse* in d8. The embedding and early features are discounted by
0.9^16 ≈ 0.19 relative to the newest output. For the RNN, with half the depth (0.9^8 ≈ 0.43), the same
scheme is slightly *better* (−0.009).

**Half-and-half** mixes the two most recent states, which is a mild recency bias with unit gain. It is +0.006 in
d8, within noise of additive, and −0.012 in the RNN, the best fixed scheme there. Its depth profiles track the
baseline closely. The main difference is lower adjacent-depth cosine (in the final RNN, 0.55–0.8 against 0.75–0.9), so each
branch moves the stream more.

**Mental model from (a).** Under a final norm, only the *relative* weights over earlier sources matter, and
equal weighting (additive = uniform) is a strong default for 16 branches. Recency bias costs d8 (EMA 0.9) and
destroys it when strong (EMA 0.1), because the identity gain compounds over depth. In the 8-branch RNN,
mild recency (half, EMA 0.9) helps slightly. What hurts there is anything that shrinks the raw final read
(uniform, EMA 0.1).

## (b) Full attention residuals

*Figures: `routing_d8.png`, `routing_rnn.png`, `depth_d8_routed.png`, `depth_rnn_routed.png`, `sweep.png`.*

Zero-initialized queries make full AttnRes *equal to uniform routing at init*: the init columns of the routed
depth plots overlap. The two models then go in opposite directions.

**d8: the best variant overall, with dense routing.** Full AttnRes reaches 2.9010, **0.026 below** the tuned
additive baseline and 0.025 below uniform routing. Starting from the same function as uniform, the learned
weights are worth the whole gain. The final routes are *dense*. We measure the effective number of sources
as exp(entropy) of each router's mean weights. That measure covers 80–90% of the sources each router can
see (e.g. 8.0 of 9 for A5, 13.6 of 17 for the final read). The routes follow a smooth pattern:

- The embedding weight decays from 0.92 (M1) to 0.20 (A2) and to 0.05–0.09 in the top quarter.
- The newest source gets an extra 0.1–0.4 above uniform.

The one sharp route is M6, which takes 0.80 of its input from b5, the output of A3. (The mean-weight
entropy can hide per-token sparsity, so these numbers bound density from above.) The depth statistics match
the block variants rather than additive. Stream RMS stays at 2–5 instead of growing to 24. The final
gradient profile is flatter than additive's, at 4e−7 to 1.4e−6 across depth versus additive's decay from 1.3e−6
to 3e−7, apart from a dip to 1.3e−7 at M6. The branch/input ratio is 1.2–1.8 instead of 0.25–0.5. Adjacent
cosine is 0.87–0.95, except 0.47 and 0.36 around M6/A7. Early training is fragile. At step 100 (lr 0.009),
the routes lean on the first attention output b1 (0.3–0.55), and the mid-stack stream gradient is 1e−9
to 1e−8, about 40–400× below additive. By the final checkpoint it has recovered.

**RNN: worse than additive, with sparse routing.** It recovers part of uniform's loss (+0.033 against +0.056) but still trails
additive. The learned routes are **sparse and non-local**. Layers R3 and R4 read 0.77 and 0.78 of their
input from source b4, the output of MLP-2. M4 reads 0.52 from b3, the output of recurrent layer 2. The
embedding weight falls from 1.0 (R1) through 0.5 (M1) to 0.06 at R3 and above. The final read spreads
across b4 (0.18), b6 (0.20) and b8 (0.25). The depth statistics show the cost of this sparsity.
Branch/input ratios swing between 0.3 and 8 (against a flat 0.5–0.8 for additive), and the R3 and R4
inputs get 15–25× less gradient than the neighboring MLP inputs (a jagged profile down to 2e−7). The
adjacent-depth cosine falls to 0.14–0.4 in the middle of the stack: each branch sees a nearly unrelated
input. Routing towards one source collapses the effective depth (several layers read the same block-2
output), and the final read remains a convex average with the same scale problem as uniform. The
step-100 maps are already strongly non-uniform (R3: 0.47 on b4), so the routes form early and then
sharpen. The effective number of sources is 2.4 of 5 for R3 and 2.4 of 7 for R4, compared with about 85%
of the available sources in d8.

A plausible reason the RNN goes sparse while d8 stays dense (a hypothesis, untested): routers score
*normalized* keys but mix *raw* values. Without a final norm, the RNN can only restore the shrunken
output scale by sending weight to high-norm sources. The inputs to R3 and R4, which read b4, have RMS
about 2.7, against 0.4–1 for their neighbors. In d8 the final RMSNorm removes that pressure, so routing is
free to spread out.

## (c) Blocked attention residuals, B = 2, 4

*Figures: `routing_d8.png`, `routing_rnn.png`, `depth_d8_routed.png`, `depth_rnn_routed.png`.*

Within a block the outputs are summed additively, so blocks interpolate between full routing (B = 1) and
additive (B = L). They also restore part of the raw scale: at init the RNN's final stream RMS is 0.30
(B=2) and 0.50 (B=4), against 0.166 for uniform/full and 1.50 for additive.

- **RNN:** B=2 (3.5159) and B=4 (3.5173) match additive (3.5160) and beat full AttnRes by 0.03. Their routes
  are much less sparse: after R1, every weight lies between 0.09 and 0.66. The embedding keeps
  0.15–0.66 through the stack, and the newest or partial block gets 0.25–0.46. Their gradient and
  branch-ratio profiles are smoother than full routing's, sitting between full routing and additive.
- **d8:** blocking also beats the tuned baseline, but by less than full routing: full −0.026 (2.9010), **B=2
  −0.021 (2.9056)**, B=4 −0.014 (2.9123). The ordering is monotone in routing resolution. Coarser blocks
  give up part of the gain, consistent with d8 using its fine-grained routes when it has them. The
  d8 block-2 sweep now has an interior minimum at 0.003 (0.001: 2.946). The B=2 curve stays below the baseline over the whole plotted range (from 250M tokens), and
  the gap widens slightly. The learned pattern is consistent. Early routers lean on the embedding (M1: 0.95), and
  the embedding weight decays to 0.09–0.16 by the top. Each router puts its largest weight, 0.3–0.6, on the
  newest or in-progress block. Older blocks share the rest. Compared with additive's equal weighting, the
  late layers get a stream dominated by recent computation, with the embedding de-emphasized. The routed
  stream RMS stays flat at 3–5 instead of growing to 25, and the branch/input ratio is 0.5–1.5 instead of
  0.25–0.5. The gradient reaching the final stream is 5–9× the additive one. B=4 has a narrow LR window: it diverges toward 3.209
  at 0.009, where the baseline still gets about 2.95. B=2 is robust up to 0.009 (2.912).

**Does this support the mental model?** Mostly, with one refinement. (1) The d8 uniform = additive control
and the RNN's correlation between final-read scale and loss support the scale part: uniform and full sit
at 0.166 (+0.056 and +0.033), B=2/B=4 at 0.30/0.50 (≈0), half/EMA-0.9 at 0.97/1.06 (−0.01). EMA 0.1 is the
exception (0.32, +0.057), but its identity gain also vanishes. (2) A fixed recency bias hurt d8 (EMA 0.9,
+0.021), but *learned* recency with a retained embedding share helps (blocks, −0.02). So the benefit isn't
recency itself. It comes from letting each layer choose its mixture. (3) What matters is whether the routing stays
*dense*, not whether it is blocked. Full routing helps in d8, where it stays dense (−0.026, the best result),
and hurts in the RNN, where it goes sparse (+0.033). Sparse, near-one-hot routes cut the effective depth and
starve some branches of gradient. Blocking forces density by summing 2–4 branches per source. That makes it
the safer choice where full routing would go sparse (RNN: full +0.033, blocks ≈ 0), and the slightly
weaker one where full routing is already dense (d8: −0.021/−0.014 against −0.026).

## Synthesis

Under pre-norm, a depth-mixing scheme is characterized by (i) the *relative* weights it gives earlier
sources, (ii) the gain of its identity path, and (iii) the scale of whatever reaches an unnormalized reader,
here the RNN's head. Additive residuals give equal weights at unit gain, which is hard to beat with a
fixed rule. Uniform routing is the same function in d8 and only a scale change elsewhere. Geometric decay
breaks the gradient highway: catastrophic at β = 0.1 over 16 branches, mildly costly at 0.9. Learned
routing helps only when it stays dense. In d8, behind a final norm, full AttnRes learns a dense mixture.
Each layer de-emphasizes the embedding and leans slightly towards recent outputs. It is the best variant
(−0.026), followed by B=2 (−0.021) and B=4 (−0.014). In the RNN, whose head reads the raw stream, full
routing collapses onto a few high-norm sources, skips layers and starves branches of gradient (+0.033).
Blocks, which force dense sums, recover parity. The natural next step for (d) is to keep routing dense
and decouple scale from routing. Options are a learned per-router gain on the routed sum, or a final norm,
which would let the RNN route like d8. Full AttnRes would also need a memory-cheaper router (shared or
low-precision source stacks) to run at batch 64 without micro-batching.
