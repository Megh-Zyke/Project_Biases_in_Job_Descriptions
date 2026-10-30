#!/bin/bash
#SBATCH --job-name=jtest_qwen38_27b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=06:00:00
#SBATCH --nice=300
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/meghss/Project_Biases_in_Job_Descriptions/Judge
#SBATCH --output=/home/meghss/Project_Biases_in_Job_Descriptions/Judge/logs/%x-%j.out
# Two-stage judge (judge_test.py): Qwen3.8-27B (bf16, 52 GB, hybrid attention) — tensor-parallel over 2 A6000s; text-only.
# Submit:  sbatch run_judge_test_qwen3.8_27b.sh

set -uo pipefail
export HF_HUB_OFFLINE=1   # weights are in /shared/4/models; never download
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  Start: $(date)"

busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$busy" ]]; then
    echo "FATAL: GPU already in use on $(hostname) — node is dirty:"; echo "$busy"; exit 1
fi

# vLLM spawns tensor-parallel workers; kill the whole process group, then sweep
# any of our processes still holding these GPUs (orphans reparented to PID 1).
cleanup() {
    [[ -n "${PGID:-}" ]] && { kill -TERM -"$PGID" 2>/dev/null; sleep 10; kill -KILL -"$PGID" 2>/dev/null; }
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [[ "$(ps -o user= -p "$pid" 2>/dev/null)" == "$USER" ]] && kill -KILL "$pid" 2>/dev/null
    done
}
trap cleanup EXIT TERM INT

setsid python judge_test.py --model Qwen/Qwen3.8-27B --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 16384 --language-model-only &
PGID=$!
wait "$PGID"
status=$?
echo "Exit: $status  End: $(date)"
exit $status
