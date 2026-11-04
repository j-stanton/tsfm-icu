#!/bin/bash
#SBATCH --job-name=tsfm-few-shot-sim
#SBATCH --partition=stampede
#SBATCH -w mscluster40
#SBATCH --time=46:00:00
# ^ This job is deliberately CAPPED, not the full uncapped factorial --
# job 65732 (mortality, strategy=similarity, run_few_shot.slurm
# UNMODIFIED, no caps) ran the full 48h wall-clock limit and only got
# through 1500/5762 eval candidates on k=1 (the CHEAPEST of the four k
# values, and the first cell run) before being killed. That's ~31s/eval
# patient under strategy="similarity" vs ~2.1s/eval patient implied by
# strategy="random"'s own timing (job 63743: ~3340s for 5762 patients) --
# roughly a 15x per-patient cost multiplier, consistent with
# run_few_shot_evaluation's own docstring: strategy="similarity" does a
# fresh k-nearest-neighbour selection AND refits Approach A's probe PER
# EVAL QUERY, not once per replicate the way strategy="random" does.
#
# At that rate the full uncapped run (4 ks x 5 seeds x 5762 patients,
# probe-only -- mortality has no Approach-B rule) would take on the
# order of 4 x 5 x 5762 x 31s =~ 10-11 DAYS. Not a bigger --time problem,
# a scope problem -- see the --ks / --max-patients choices below instead.
#
# SECOND ATTEMPT, after job 68173 (this same capped scope, WITHOUT
# --resume) ran the full 46h again and STILL lost everything: the real
# per-patient cost came in at ~37.9s (worse than the ~31s/patient
# estimate above, which was based on a smaller, earlier partial sample),
# so it only got through k=1 (all 5 seeds, clean) plus 3 of 5 seeds of
# k=5 before being killed -- and because run_few_shot.py only
# wrote its output JSON ONCE, at the very end of the whole --ks sweep,
# that entire ~46h of real GPU compute produced zero bytes on disk.
# Fixed properly this time: run_few_shot.py now checkpoints
# results to --output after EVERY (task,approach,strategy,k) cell (see
# its _write_checkpoint), and --resume (added below) makes a THIRD
# attempt, if one is ever needed, skip whatever cells already completed
# rather than re-paying for them. This run still starts from scratch
# (nothing was saved from job 68173 to resume from), but is now safe
# against losing partial progress to a future timeout.
#
# SUBMIT THIS FROM ~/tsfm-icu-early-warning/few_shot/:
#   sbatch run_few_shot_similarity_capped.slurm
# (no positional args -- this is a one-off scoped probe, not a general
# per-(task,strategy) driver like run_few_shot.slurm)
#SBATCH --output=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/sim-%j-output-%N.txt
#SBATCH --error=/home-mscluster/%u/tsfm-icu-early-warning/few_shot/results/sim-%j-error-%N.txt

# SCOPE CHOICES, all sized off job 65732's real ~31s/patient rate:
#   --ks 1 5            k=1 and k=5 only (not 10, 20) -- k=1 isolates the
#                        DEGENERATE-probe baseline (cheap to confirm it
#                        still hits the neutral 0.5 fallback under
#                        similarity the same way it does under random);
#                        k=5 is the cheapest k that actually exercises a
#                        fitted (if uncalibrated) probe. Skipping 10/20
#                        roughly halves the cost for comparatively little
#                        extra information, since random-strategy's own
#                        k=10/20 cells didn't show a clean k-trend either
#                        (see phase3_full_mortality_random.json).
#   --max-patients 500   5 seeds x 2 ks x 500 patients x 31s/patient =~
#                        43.1 hours -- leaves real margin under this job's
#                        46h --time and the confirmed 72h stampede max.
#                        500 is NOT the full 5762-patient test set, so
#                        this cell's bootstrap CI will be wider than
#                        strategy="random"'s -- expected and worth noting
#                        in the write-up as a scope limitation specific
#                        to the similarity-strategy numbers, not a flaw
#                        in the comparison itself (same 500 patients,
#                        same seeds, same task -- a fair apples-to-apples
#                        timing/signal check against what strategy=
#                        "random" would show on an equivalently-sized
#                        slice).
#   --approaches probe   mortality has no Approach-B rule anyway (see
#                        TASK_DECISION_RULES), so this is really a no-op
#                        here -- included for clarity/symmetry with how
#                        aki/sepsis's eventual similarity jobs will need
#                        to make the same choice explicitly, since THOSE
#                        tasks DO have a rule approach and rule-approach
#                        cells don't refit a probe at all (cheaper -- see
#                        run_few_shot_evaluation) but still pay the
#                        per-query k-NN selection cost.
#   --n-seeds 5           kept at the full Section 3.4 spec (5) rather
#                        than cut further -- seeds are what the bootstrap
#                        CI needs, and cutting --max-patients already
#                        bought back the time budget without sacrificing
#                        seed count.

sleep 20

echo "=== Useful Debugging Information ==="
echo "Date: "$(date)
echo "Username: "$USER
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Home Directory: "$HOME
echo "Node: "$HOSTNAME
echo "Current Working Directory: "$PWD
echo "Job Submitted From Machine: "$SLURM_SUBMIT_HOST
echo "Job Submitted From Directory: "$SLURM_SUBMIT_DIR
echo "Number of Nodes: " $SLURM_JOB_NUM_NODES
echo "List of Nodes: "$SLURM_JOB_NODELIST
echo "Number of Tasks: "$SLURM_NTASKS
echo "======"

cd $SLURM_SUBMIT_DIR
mkdir -p results

source /etc/profile.d/lmod.sh
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00

SIF_PATH="/usr/local/containers/pytorch/pytorch-cuda12.9-ubuntu24.04-00.sif"

REPO_ROOT="$SLURM_SUBMIT_DIR/.."
COHORT_ROOT="$REPO_ROOT/external/YAIB-cohorts/data"
SPLIT_FILE="$REPO_ROOT/splits/shared_split.csv"

apptainer exec --nv "$SIF_PATH" bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tsfm-adapter

    set -e

    export PYTHONUNBUFFERED=1
    # Same buffering fix as run_few_shot.slurm -- see that file's
    # comment for the full job-61982 history. Kept here unchanged since
    # it's a correctness/visibility fix independent of this job's scope.

    cd $SLURM_SUBMIT_DIR

    echo '--- Confirming real data paths exist before running ---'
    test -f '$COHORT_ROOT/mortality24/mimic/dyn.parquet' && echo 'mortality24/mimic: OK' || { echo 'MISSING: mortality24/mimic/dyn.parquet'; exit 1; }
    test -f '$SPLIT_FILE' && echo 'shared_split.csv: OK' || { echo 'MISSING: shared_split.csv'; exit 1; }

    echo ''
    echo '--- Phase 3 CAPPED similarity run: task=mortality, strategy=similarity, k in {1,5}, 5 seeds, max-patients=500 ---'
    echo '--- --resume is set: if results/phase3_similarity_capped_mortality.json already has'
    echo '--- completed (task,approach,strategy,k) cells from an earlier attempt, they are'
    echo '--- skipped rather than re-run (see run_few_shot.py --resume help text).'
    python3 -u run_few_shot.py \
        --cohort-root '$COHORT_ROOT' \
        --split-assignment '$SPLIT_FILE' \
        --tasks mortality \
        --ks 1 5 \
        --strategies similarity \
        --approaches probe \
        --n-seeds 5 \
        --max-patients 500 \
        --resume \
        --output '$SLURM_SUBMIT_DIR/results/phase3_similarity_capped_mortality.json'
"

echo ""
echo "Done. Check results/phase3_similarity_capped_mortality.json for the results table."
echo "This is a SCOPED result (max-patients=500, ks={1,5} only) -- not directly"
echo "comparable in CI width to phase3_full_mortality_random.json's full-5762-patient"
echo "numbers, but DOES give a fair, same-sized, same-seeds comparison of"
echo "strategy=similarity vs strategy=random's AUROC/AUPRC/ECE on the same slice."
echo "If you want that direct comparison, rerun strategy=random with the same"
echo "--max-patients 500 --ks 1 5 flags for an apples-to-apples pair."
echo ""
echo "NOTE (see few_shot_pipeline.py's _fit_probe_on_demos / run_few_shot_evaluation):"
echo "Approach A's probe is trained on ZERO-SHOT forecast features of the"
echo "demonstrations but scored on ICL-CONDITIONED forecast features of the query --"
echo "a real train/inference feature-distribution mismatch, not a crash-level bug."
echo "This affects probe numbers under BOTH strategies, but is worth keeping in mind"
echo "if this job's k=5 AUROC looks unexpectedly weak or below-chance, the way"
echo "aki/random's k=5/k=10 did (see phase3_full_aki_random.json)."
