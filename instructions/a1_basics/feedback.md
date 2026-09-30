## Problem 2
Mostly yes. The allocation is sensible, but I would keep (a), revise (b) slightly, and redesign (c) to be adaptive rather than committing 18 runs up front.
The handout explicitly asks for roughly 20 / 10 / 20 runs in (a)/(b)/(c): four d4–d9 scaling curves in (a), slope-changing-but-still-linear interventions in (b), and interventions that actually destroy the power-law form in (c). Stanford CS 312
(a): Codex's plan is basically right
The current handout really does specify exactly these four variants:
\[
\begin{aligned}
&\text{baseline: } &&\eta=0.003,\ \text{linear},\ \text{dropout}=0\\
&\text{constant LR: } &&\eta=0.003,\ \text{constant}\\
&\text{dropout: } &&\eta=0.003,\ \text{linear},\ \text{dropout}=0.2\\
&\text{large LR: } &&\eta=0.03,\ \text{linear}.
\end{aligned}
\]and depths \(d=4,\ldots,9\). Stanford CS 312
So mathematically there are
\[
4\times6=24
\]points. If three d8 points already exist from Problem 1, then 21 new runs is correct. The handout's “20 runs” is clearly an approximate budget, not an exact requirement.
The pre-registration procedure should be slightly more precise than Codex describes. For each curve, fit only
\[
d4,d5,d6,d7
\]and save
\[
\hat L_{d8},\qquad \hat L_{d9}
\]before revealing the held-out points. The assignment explicitly says to make predictions before training d8/d9. Stanford CS 312
For the three configurations whose d8 results you've already seen, you obviously cannot honestly preregister d8 anymore. I would simply say that in the writeup:
d8 had already been observed during Problem 1, so only the d9 prediction was genuinely held out for these configurations.

Don't pretend otherwise. For dropout, if you haven't seen d8, you can do the intended d8+d9 test.
Also, remember that the fit's \(x\)-axis should be the body-compute proxy
\[
C \approx 6ND,
\]with \(D\) fixed. Since their ladder has width \(=64d\),
\[
N\approx14.5\,d(64d)^2\propto d^3,
\]so
\[
C_d\propto d^3.
\]The handout later explicitly defines this FLOPs proxy. Stanford CS 312
(b): good idea, but I'd change the model-size experiment
The data-scaling part is quite good.
At fixed d6, comparing
\[
B=64
\quad\text{vs}\quad
B=128
\]over
\[
D\in\{0.25,0.5,1,2\}D_0
\]has a clear mechanism. Since
\[
\#\text{optimizer steps}
=
\frac{D}{B},
\]up to the fixed context-length factor, doubling batch size halves the number of updates at a given token budget. Thus the impact of \(B=128\) can plausibly depend systematically on how many tokens you train for. This is exactly the kind of thing part (b) is asking for: different slopes, but hopefully still smooth scaling curves. The handout explicitly tells you to consider both model-size scaling and data scaling. Stanford CS 312
I am less enthusiastic about:
“Problem 1's best recipe (LR \(10^{-2}\), warmup 0.1) across d4–d9.”

The idea is valid, but the hypothesis already says
“the best LR shrinks as width grows.”

If that's the phenomenon you're trying to study, using one fixed aggressive LR \(0.01\) risks crossing from case 2: different slope into case 3: nonlinear/unstable at larger models.
That would make it a good part-(c) experiment, but possibly a bad part-(b) experiment.
A cleaner model-size experiment would be to compare the baseline fixed LR against a width-scaled LR recipe:
\[
\boxed{
\eta(d)=0.003\frac{8}{d}
}
\]because width is \(64d\), so this is
\[
\eta\propto\frac1{\text{width}}.
\]This is particularly well motivated because the handout's later engineering-scaling example explicitly uses
\[
\eta\propto C^{-1/6},
\]while depth and width scale as \(C^{1/6}\); equivalently,
\[
\eta\propto\frac1{\text{width}}.
\]They even give
d4:\eta=.006,\qquad
d8:\eta=.003,\qquad
d16:\eta=.0015.
\] :chatgpt-content-reference{index="5"}


So I would prefer something like

\[
\eta(d)=0.024/dversus fixed \(\eta=.003\). That gives you a very interpretable question:
Does scaling the LR with model width change the model-size scaling exponent while preserving a clean power law?

That's more directly aligned with part (b).
Your data-scaling \(B=64\) vs \(B=128\) experiment can stay.
That would put (b) around 10–12 new runs, depending on reuse, which is perfectly reasonable.
(c): this is where I would make the biggest change
I wouldn't commit to three interventions × six depths immediately.
The objective isn't to produce three full curves. It's to find examples that actually break scaling. The handout says to find potentially radical interventions leading to strong departures from the power-law form and determine where that happens. Stanford CS 312
If you spend 18 runs on three guesses and all three remain approximately linear, you've spent almost the whole budget without accomplishing the actual objective.
Your proposed candidates also have an interpretability problem:
Proposed intervention	My assessment
No QK-norm + LR \(10^{-2}\)	Interesting, but changes two things simultaneously
60k unique × 10 epochs	Very strong candidate
No warmup + LR \(10^{-2}\)	Again changes two things simultaneously


The assignment explicitly recommends “change one thing at a time” where possible. Stanford CS 312
The repeated-data experiment is especially good. You're holding total processed tokens near 614M while changing the number of unique tokens drastically. Larger models have more capacity relative to the fixed unique dataset, so you have a credible mechanism for a regime transition from beneficial scaling to overfitting/data limitation. The hyperparameter table explicitly supports num_epochs > 1 for repeating data. Stanford CS 312
I'd make (c) a screen → expand design instead:
\[
\text{candidate intervention}
\quad\longrightarrow\quad
d4,\ d6,\ d9
\]first.
For example, screen four interventions with three depths each:
\[
4\times3=12\text{ runs}.
\]Good candidates could be repeated data, no QK norm, zero warmup with otherwise-default LR, and some extreme optimizer setting suggested by your Problem 1 observations.
Then look at the curves. Take the two candidates that actually show curvature/regime change and fill in
\[
d5,d7,d8
\]for each:
\[
2\times3=6
\]more runs.
You're at 18 runs and now have two six-point curves selected specifically because they show the phenomenon the question asks about. The last ~2 runs can test a severity level or replicate an anomalous endpoint.
That is much more compute-efficient.
So I'd modify the overall plan to
\[
\boxed{
\begin{array}{c|c|l}
&\text{new runs}&\text{strategy}\\
\hline
(a)&21&\text{essentially Codex's plan}\\
(b)&\sim10\!-\!12&
\text{LR-vs-width scaling + BS data scaling}\\
(c)&\sim18\!-\!20&
\text{screen several break mechanisms, then expand the winners}
\end{array}
}
\]One conceptual distinction is worth keeping very sharp throughout the experiments:
\[
\boxed{
\text{Part (b): slope changes, functional form survives}
}
\]versus
\[
\boxed{
\text{Part (c): functional form itself fails}.
}
\]For (b), a result like
\[
L_A(C)=a_A C^{-\alpha_A}+\epsilon_A,
\qquad
L_B(C)=a_B C^{-\alpha_B}+\epsilon_B,
\qquad
\alpha_A\neq\alpha_B
\]is exactly what you want.
For (c), merely getting a much worse loss is not sufficient. You want systematic curvature, instability, plateauing, reversal, etc., such that no single \(aC^{-\alpha}+\epsilon\) gives a convincing description across the scaling range.
That distinction is the main thing I'd optimize the compute plan around.



## Problem 3
The plan is close for (a) and (b), but I would definitely change (c). The biggest issue is that the proposed d6 experiment goes in the opposite direction from what the assignment explicitly asks you to study.
The handout defines the three natural sources as initialization, data order/choice, and hardware nondeterminism, then asks: (a) vary all three jointly, (b) isolate them, and (c) test whether hyperparameters change variability and specifically ask what happens for more capable, larger/longer-trained models with lower loss. Stanford CS 312
(a) Joint variation: basically good
Three new runs with
\[
(\text{model seed},\text{data seed})=(1,1),(2,2),(3,3)
\]and deterministic=False is exactly the kind of experiment they want. Initialization changes, data order changes, and GPU nondeterminism remains active.
One small correction: I would not treat both existing seed-42 runs as two ordinary samples of the full joint distribution. They share the same initialization and data ordering, so their difference only reflects nondeterministic execution. You can certainly plot them, but statistically they are not equivalent to two more independent draws of all three sources.
A cleaner full-variation sample is
\[
s\in\{1,2,3,42\},
\]one run per paired seed. If you happen to have multiple seed-42 replicates, use those separately to illustrate within-seed noise.
Also, with 3–4 observations you cannot meaningfully characterize the shape of a distribution. You can estimate its rough scale/range. The assignment itself only budgets 3 runs here, so that's fine—just don't overstate what you infer about Gaussianity or tails. The handout asks for the general order of magnitude and training-curve variability with only three suggested runs. Stanford CS 312
(b) Isolation: conceptually right, but restructure the hardware portion
These two are clean:
\[
\boxed{\text{initialization only}}
\]model_seed = 1,2,3,...
data_seed = 42
deterministic = True
and
\[
\boxed{\text{data order only}}
\]model_seed = 42
data_seed = 1,2,3,...
deterministic = True
That's exactly how I would isolate them.
The deterministic duplicate runs are also useful as a negative control: with all seeds/configuration fixed and deterministic execution requested, there should be essentially no run-to-run variation on a fixed setup. The handout explicitly gives two identical deterministic reference runs for this purpose. Stanford CS 312
I would change the hardware design, though. Codex currently proposes:
- 2 nondeterministic H100
- 1 nondeterministic A100
- 1 deterministic A100
The problem is that an H100-vs-A100 difference is not quite the same thing as run-to-run hardware nondeterminism. It also changes the hardware architecture itself.
A cleaner experiment is:
\[
\begin{array}{c|c|c|c}
&\text{seeds}&\text{deterministic}&\text{replicates}\\
\hline
\text{H100}&42,42&\text{False}&2\\
\text{A100}&42,42&\text{False}&2
\end{array}
\]Then you can ask two separate questions:
1. Within one GPU type, how different are repeated runs with everything else identical?
2. Across H100 vs A100, are the resulting trajectories/endpoints systematically different?
The assignment explicitly says to use at least two machines such as H100/A100 when investigating hardware nondeterminism. Stanford CS 312
The deterministic A100 cross-hardware run is interesting but lower priority. If you're compute constrained, I'd trade it for a second nondeterministic A100 replicate.
So I would make (b) something like:
\[
3\text{ init-only}
+3\text{ data-only}
+2\text{ deterministic references}
+4\text{ hardware-noise runs}
=12.
\]That's exactly the top end of their suggested 6–12. Stanford CS 312
(c) is where I'd change the plan
Codex proposes:
- high LR,
- small batch,
- d6 less-capable model.
The first two are excellent. In fact, the assignment's own example quiz compares exactly
\[
B=16
\]and
\[
\eta=0.009
\]against the standard recipe, using ten paired seeds and asking whether either more than doubles the sample standard deviation. Stanford CS 312
So using high LR and small batch is extremely aligned with what they want you to learn.
But this:
less capable model: d6

does not answer the second half of part (c). The assignment explicitly asks:
“For more capable (larger/longer training) models with lower losses, how does the run-to-run variation change?” Stanford CS 312

I would replace d6 with d9 baseline.
You already have a d9 seed-42 baseline from Problem 2(a), so run
\[
d9,\quad s=1,2,3
\]with paired seeds. Then you have four samples:
\[
s\in\{1,2,3,42\}.
\]Now you can directly compare
\[
\sigma_{\text{d8}}
\quad\text{vs}\quad
\sigma_{\text{d9}},
\]where d9 is larger, has lower loss, and is therefore the “more capable” direction the question requests.
An even stronger experiment, if you have a few spare runs, would be longer-trained d8, e.g.
\[
D=2D_0,
\]because then architecture is fixed and only capability/training horizon changes. That lets you distinguish
\[
\text{larger model}
\]from
\[
\text{longer training}.
\]But you don't need both to satisfy the prompt.
I would therefore use this plan
Part	Experiment	New runs
(a)	joint seeds 1–3, nondeterministic	3
(b)	initialization only	3
	data order only	3
	deterministic references	2
	hardware nondeterminism: 2 H100 + 2 A100	4
(c)	LR \(0.009\) or your \(0.01\), paired seeds 1–3	3
	batch 16, paired seeds 1–3	3
	d9 baseline, paired seeds 1–3	3


That's 24 new runs, exactly the same total as Codex's proposal.
And conveniently, for the three (c) conditions you can reuse one seed-42 run each:
\[
\begin{aligned}
\text{high LR}:&\quad \text{Problem 1},\\
B=16:&\quad \text{Problem 1},\\
d9:&\quad \text{Problem 2}.
\end{aligned}
\]giving \(n=4\) for each comparison.
One more statistical point
For part (b), don't expect the variances to add directly:
\[
\sigma_{\text{joint}}
\stackrel{?}{=}
\sigma_{\text{init}}
+\sigma_{\text{data}}
+\sigma_{\text{hardware}}.
\]If the sources were independent and additive in their effect on final loss, it would instead be the variances that approximately add:
\[
\boxed{
\sigma_{\rm joint}^2
\approx
\sigma_{\rm init}^2+
\sigma_{\rm data}^2+
\sigma_{\rm hw}^2.
}
\]But deep-network training is nonlinear and chaotic, so this decomposition need not hold. In fact, discovering that tiny hardware perturbations get amplified differently depending on initialization/data is part of the conceptual bridge into Problem 4—the handout explicitly describes chaotic dynamics as an amplifier of existing randomness rather than an independent random source. Stanford CS 312
So for (b), I'd report both the SDs,
\[
\sigma_{\rm init},\quad
\sigma_{\rm data},\quad
\sigma_{\rm hw},\quad
\sigma_{\rm all},
\]and the corresponding fraction of total variance,
\[
\boxed{
\frac{\sigma_i^2}{\sigma_{\rm all}^2}.
}
\]With only 3–4 samples these estimates will be noisy, but conceptually that is the right quantity when the assignment asks what “fraction of the full variation” each source produces.

## Problem 4-6
Problem 4 is mostly well designed, Problem 5 is fine with a few tweaks, but Problem 6 is substantially under-planned. The current rendered PDF says “Problem 6: Activations and gradient norms (20 runs)” and part (c) itself asks for 20 runs manipulating these statistics. So the proposed 2-run plan misses the main experimental component of the problem. Stanford CS 312
Problem 4 — good structure, but fix the perturbation implementation
The handout wants two things: first, show that a microscopic data perturbation can eventually produce a macroscopic difference; second, characterize divergence as a function of both perturbation magnitude and time remaining after the perturbation. Stanford CS 312
Your proposed grid does exactly that conceptually:
\[
\epsilon \times T
\]with several perturbation magnitudes and several injection times. Eleven new runs is reasonable because the handout suggests 1–5 for (a) and 10–14 for (b). Stanford CS 312
I would make four technical changes.
First, use a fixed perturbation direction across the magnitude sweep. Suppose for parameter tensor \(W\),
\[
W' = W + \epsilon\,\mathrm{RMS}(W)\,Z,
\qquad Z_{ij}\sim\mathcal N(0,1).
\]Generate \(Z\) with a fixed perturbation seed, and reuse the same \(Z\) when comparing \(\epsilon=10^{-6}\), \(10^{-3}\), etc. Then the only change is amplitude:
\[
\epsilon_1 Z,\quad \epsilon_2 Z.
\]Otherwise different random noise draws introduce another uncontrolled variable.
Second, be very careful with the claimed \(10^{-8}\) “below bf16 precision” point. If the actual parameter tensor you're modifying is bf16, adding
\[
10^{-8}\times \mathrm{RMS}(W)
\]will simply round away. In fact even \(10^{-6}\) or sometimes \(10^{-3}\) relative perturbations can be largely quantized away depending on parameter magnitude. A null perturbation teaches you nothing about chaotic amplification.
After injection, explicitly measure
\[
\frac{\|W'-W\|_2}{\|W\|_2}
\]and verify it is nonzero. If weights are FP32 under mixed precision, you're fine; if they're bf16, rethink the perturbation mechanism.
Third, T=0 needs special handling. If your callback runs after an optimizer step, then it cannot actually perturb at step 0. Either inject before training begins, or call the earliest callback point \(T=1\) and describe it accurately.
Fourth, I would log one additional quantity beyond loss:
\[
\boxed{
\Delta_\theta(t)
=
\frac{\|\theta^{\rm perturbed}_t-\theta^{\rm ref}_t\|_2}
{\|\theta^{\rm ref}_t\|_2}
}
\]if convenient. The handout asks about loss divergence, so loss remains the primary metric, but parameter-space divergence lets you directly see the amplification process before it becomes visible in loss.
Your central plots should be
\[
|\Delta L_{\rm train}(t)|,\qquad
|\Delta L_{\rm val}(t)|
\]on a log y-axis if values span many orders of magnitude. You may see something qualitatively like
\[
\Delta(t)\approx \Delta(T)e^{\lambda(t-T)}
\]for an initial regime before saturation. You don't need to claim true Lyapunov dynamics, but that's the phenomenon you're probing.
I would keep the 10-run grid, perhaps like this:
\[
\epsilon\in\{10^{-6},10^{-3}\},\quad
T\in\{0,2000,5000,8000\}
\]= 8 runs, plus one genuinely tiny/null control and one large-perturbation control.
So Problem 4: yes, essentially keep it.
Problem 5 — 4 new runs is reasonable, but don't waste runs duplicating P1
The handout explicitly says to use your previous hyperparameter runs for LR, batch size, and schedule, and spend up to five new runs mainly to fill gaps such as momentum. Parts (b) and (c) explicitly suggest zero new runs. Stanford CS 312
So the philosophy should be:
P5 is mainly an analysis problem, not another large sweep.

The two \(\beta_1\) runs are the most valuable:
\[
\beta_1=0.5,\qquad
\beta_1=0.95,
\]compared against default
\[
\beta_1=0.9.
\]That gives you the missing axis explicitly named by the assignment: momentum.
The \(\beta_2=0.99\) run is also worthwhile. It can change how quickly Adam's second-moment estimate reacts to changing/noisy gradients:
\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]Larger \(\beta_2\) makes \(v_t\) smoother/slower-moving, so it is a sensible experiment for microstructure.
I am less convinced you need a new bs256 run if Problem 1 already contains one. The quiz example uses bs256, but that doesn't imply you need to rerun it. The handout specifically encourages looking back at existing curves. Stanford CS 312
If you have one spare run, I'd rather test another momentum extreme, perhaps
\[
\beta_1=0
\]than duplicate bs256. Then you get a very interpretable sequence:
\[
\beta_1=0,\quad0.5,\quad0.9,\quad0.95.
\]The main analysis should separate two ideas:
Macrostructure: overall descent envelope,
\[
L_t^{\rm smoothed}.
\]Look at warmup shape, rate of descent, curvature, late-training annealing, plateaus, instability.
Microstructure: fluctuations around that envelope,
\[
r_t=L_t-\operatorname{Smooth}(L)_t.
\]You can quantify smoothness rather than relying only on visual judgment:
\[
\sigma_{\rm local}
=
\operatorname{std}(r_t)
\]over comparable training windows.
That gives you a clean way to say, for example, that larger batches reduce short-time loss variance without necessarily changing the macroscopic shape in the same way.
So Problem 5: 3–4 new runs is perfectly adequate.
Problem 6 — this plan needs a major expansion
This is the big issue.
The currently rendered assignment says:
Problem 6: Activations and gradient norms (20 runs)

and for part (c):
“…figure out how to manipulate these using hyperparameters. Can you make the parameter, gradient or activation statistics larger or smaller uniformly? what about just at the start or end of training? (20 runs)” Stanford CS 312

So:
No QK norm + WD=0

is a useful starting point, but it is nowhere near the intended experiment.
The goal isn't simply to find two settings that alter internal statistics. It is to acquire a causal dictionary from hyperparameters → internal training statistics.
You should first define a small set of statistics you care about. I would choose perhaps:
\[
\boxed{\text{parameter RMS}}
\]\[
\boxed{\text{gradient RMS or global gradient norm}}
\]\[
\boxed{\text{residual-stream activation RMS}}
\]and, if available,
\[
\boxed{\text{Q/K RMS or attention-logit RMS}}.
\]Then inspect them along two axes:
\[
\text{training time}
\]and
\[
\text{layer depth}.
\]For example, select early/mid/late checkpoints,
\[
t\in\{0,\ T/2,\ T\},
\]and several representative layers,
\[
\ell\in\{1,4,8\}.
\]That directly answers part (a), which asks you to understand how parameter, activation, and gradient statistics evolve across both trajectory and depth. Stanford CS 312
Then use part (c) as controlled interventions
I'd organize the ~20 runs into families rather than twenty random settings.
A strong design would be something like:
Intervention family	Example values	What it should probe
learning rate	\(3\!\times10^{-4},10^{-3},3\!\times10^{-3},9\!\times10^{-3}\)	gradient/update scale, instability
weight decay	\(0,0.01,0.1,1.0\)	parameter norms over training
warmup	\(0,0.01,0.1\)	early-training gradient/activation behavior
batch size	\(16,64,256\)	gradient noise / smoothness
QK norm	on/off	Q/K and attention-logit magnitude
grad clipping	None, 1.0, perhaps lower threshold	directly manipulate gradient norms


A lot of the default/control values already exist, so this need not literally require 20 new jobs if reuse is accepted. But you should have roughly twenty experimental conditions represented in the analysis.
The key is to make predictions before looking.
For example:
Weight decay
AdamW applies
\[
\theta_{t+1}
=
(1-\eta_t\lambda)\theta_t-\eta_tu_t.
\]So increasing \(\lambda\) gives you a direct hypothesis:
\[
\lambda\uparrow
\quad\Rightarrow\quad
\|\theta\|_{\rm RMS}\downarrow
\]especially later in training.
This is a beautiful P6 intervention because it primarily targets parameter statistics.
Learning rate
Increasing \(\eta\) does not directly alter the raw gradient,
\[
g_t=\nabla_\theta L,
\]but larger parameter updates move the system into different states, potentially generating larger gradient and activation excursions later.
So you can ask whether high LR primarily changes
\[
\text{gradient RMS},
\]\[
\text{activation RMS},
\]or both.
Gradient clipping
This is almost tailor-made for the assignment because it gives you a direct control knob:
\[
g'_t
=
g_t\min\left(1,\frac{c}{\|g_t\|}\right).
\]Changing \(c\) should have an immediately visible effect on post-clipping gradient norms. That's a much cleaner example of “make a statistic smaller” than some indirect hyperparameter changes.
QK norm
Also excellent, but don't just inspect “activations” generically.
Without QK norm,
\[
q^\top k
\]depends directly on \(\|q\|\|k\|\), whereas normalized queries/keys constrain that scale. So log:
\[
\operatorname{RMS}(q),\quad
\operatorname{RMS}(k),\quad
\operatorname{RMS}\left(\frac{qk^\top}{\sqrt{d}}\right).
\]That makes the intervention mechanistically interpretable.
The P6 question I would explicitly try to answer
For each hyperparameter \(h\), determine whether its effect looks like:
\[
\textbf{uniform in time:}
\qquad
S_h(t)\approx c\,S_{\rm base}(t)
\]or
\[
\textbf{early-only:}
\qquad
S_h(t)\neq S_{\rm base}(t)
\text{ near }t=0,\ 
S_h(t)\to S_{\rm base}(t)
\]or
\[
\textbf{late-emerging:}
\qquad
S_h(0)\approx S_{\rm base}(0),
\quad
|S_h(t)-S_{\rm base}(t)|\uparrow.
\]That phrasing maps almost exactly onto the assignment's question of whether you can manipulate statistics “uniformly” versus “just at the start or end of training.” Stanford CS 312
My revised budget
I'd therefore use approximately:
Problem	New runs	Assessment
P4	11	Good; fix perturbation precision/seed details
P5	3–4	Good; primarily reuse and analyze P1
P6	perhaps 8–12 new + extensive reuse	Need substantially more coverage than 2


You don't necessarily need 20 new P6 runs, because your P1 sweeps already contain many of the required interventions. But the final P6 analysis should feel like a 20-condition investigation, not a two-run appendix.
The most useful next step is probably to inventory your exact P1 runs and map them onto P5/P6. We can likely cut quite a lot of redundant compute while still covering what the assignment asks.





  











ChatGPT can make mistakes. Check important info.