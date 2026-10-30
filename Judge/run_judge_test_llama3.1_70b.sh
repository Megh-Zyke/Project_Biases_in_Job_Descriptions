#!/bin/bash
#SBATCH --job-name=jtest_llama70b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=06:00:00
#SBATCH --nice=400
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/meghss/Project_Biases_in_Job_Descriptions/Judge
#SBATCH --output=/home/meghss/Project_Biases_in_Job_Descriptions/Judge/logs/%x-%j.out
# Two-stage judge (judge_test.py): Llama-3.1-70B-Instruct (bf16, 132 GB) — tensor-parallel over 4 A6000s on one node.
# Submit:  sbatch run_judge_test_llama3.1_70b.sh

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

setsid python judge_test.py --model meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 16384 &
PGID=$!
wait "$PGID"
status=$?
echo "Exit: $status  End: $(date)"
exit $status
