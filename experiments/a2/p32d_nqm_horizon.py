"""Problem 3.2(d): does the NQM's beta1 = .99 blow-up (3.1d) come from its short horizon?

Adam on the 2D NQM (sigma = 1, beta2 = .95, LR retuned per beta1) at B = 16 with the 3.1(d) budget
(N = 8192, 512 updates) and with 8x and 64x more updates. Prints the best loss relative to beta1 = 0.
Runs locally on CPU (~10 s).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.a2.nqm import tune

BATCH = 16
BETA1S = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99)
STEPS = (512, 4096, 32768)
COARSE_LRS = np.geomspace(1e-5, 1.0, 26)


def _job(job):
    steps, beta1 = job
    _, _, _, lr, best, edge = tune("adam", BATCH, COARSE_LRS, beta1=beta1, n_samples=256,
                                   n_examples=steps * BATCH, seed=1)
    return job, (lr, best, edge)


def main():
    with ProcessPoolExecutor() as pool:
        res = dict(pool.map(_job, [(s, b1) for s in STEPS for b1 in BETA1S]))
    for s in STEPS:
        base = res[(s, 0.0)][1]
        print(f"B = {BATCH}, {s} updates (sigma = 1, beta2 = .95, LR retuned per beta1)")
        for b1 in BETA1S:
            lr, best, edge = res[(s, b1)]
            tau = 1 / (1 - b1)
            print(f"  beta1 {b1:<5} 1/(1-beta1) = {tau:>4.0f} steps ({tau / s:6.2%} of run)  lr* {lr:.2e}  "
                  f"loss {best:.3e}  x{best / base:.3f} vs beta1 = 0" + ("  (LR at grid edge)" if edge else ""))


if __name__ == "__main__":
    main()
