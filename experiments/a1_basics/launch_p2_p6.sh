#!/usr/bin/env bash
# Launch all Problem 2-6 runs (86 runs across 12 Modal apps).
# Usage: bash experiments/a1_basics/launch_p2_p6.sh [--dry-run]
# Every launcher skips runs whose finished model already exists, so rerunning this after
# everything has *finished* is safe; rerunning while jobs are still queued duplicates them.
set -uo pipefail

cd "$(dirname "$0")/../.."
extra_args=("$@")
log="experiments/a1_basics/launch_p2_p6_$(date +%Y%m%d-%H%M%S).log"
failed=()

launch() {
    local module=$1 part=$2
    echo "=== ${module} ${part}" | tee -a "$log"
    if ! uv run python -m "experiments.a1_basics.${module}" "$part" "${extra_args[@]+"${extra_args[@]}"}" 2>&1 \
        | grep -v "Could not preflight" | tee -a "$log"; then
        failed+=("${module} ${part}")
    fi
}

# Deterministic references first: Problem 4 compares against them.
launch p3_measuring_variation b
launch p4_amplification a
launch p4_amplification b
launch p2_scaling_law_reliability a
launch p2_scaling_law_reliability b
launch p2_scaling_law_reliability c
launch p3_measuring_variation a
launch p3_measuring_variation b-h100
launch p3_measuring_variation c
launch p5_loss_curve_augury a
launch p6_activation_gradient_norms c
# Last: the only launch that uses a different GPU type (fix A100_GPU if Modal rejects it).
launch p3_measuring_variation b-a100

echo "Log: $log"
if ((${#failed[@]})); then
    echo "FAILED launches: ${failed[*]}" | tee -a "$log"
    exit 1
fi
if [[ " ${extra_args[*]+"${extra_args[*]}"} " == *" --dry-run "* ]]; then
    echo "Dry run only: nothing was launched." | tee -a "$log"
else
    echo "All launches submitted." | tee -a "$log"
fi
