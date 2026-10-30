#!/bin/bash
#SBATCH --job-name=kv_cache
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nice=300
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/meghss/Project_Biases_in_Job_Descriptions/Judge
#SBATCH --output=/home/meghss/Project_Biases_in_Job_Descriptions/Judge/logs/%x-%j.out
# Submit with a GPU type, e.g.:  sbatch --gres=gpu:A6000:1 run_kv_cache.sh
# Runs the HF experiment, then the vLLM one (pass "hf" or "vllm" to run only one).
# Pass "judge" to run the two-stage Nemotron judge (judge_test.py) instead.

set -uo pipefail
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"

busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$busy" ]]; then
    echo "FATAL: GPU already in use on $(hostname) — node is dirty:"; echo "$busy"; exit 1
fi

cleanup() { pkill -TERM -P $$ 2>/dev/null; sleep 5; pkill -KILL -P $$ 2>/dev/null; }
trap cleanup EXIT TERM INT

WHICH=${1:-both}
if [[ "$WHICH" == "both" || "$WHICH" == "hf" ]]; then
    echo "===== HF transformers ====="
    python kv_cache.py
fi
if [[ "$WHICH" == "both" || "$WHICH" == "vllm" ]]; then
    echo "===== vLLM ====="
    python kv_cache_vllm.py
fi
if [[ "$WHICH" == "judge" ]]; then
    echo "===== Two-stage judge (vLLM) ====="
    python judge_test.py
fi
