#!/bin/bash
#SBATCH --job-name=judge_smoke_tp2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=01:30:00
#SBATCH --nice=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/meghss/Project_Biases_in_Job_Descriptions/Judge
#SBATCH --output=/home/meghss/Project_Biases_in_Job_Descriptions/Judge/logs/%x-%j.out
# Smoke test of the 2-GPU (tensor-parallel) path with small models, 5 personas per batch.
# Submit:  sbatch run_smoke_tp2.sh

set -uo pipefail
export HF_HUB_OFFLINE=1 NCCL_DEBUG=INFO
OUT=/home/meghss/Project_Biases_in_Job_Descriptions/Output_Files/smoke
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  GPUs: ${CUDA_VISIBLE_DEVICES:-none}  Start: $(date)"

busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$busy" ]]; then
    echo "FATAL: GPU already in use on $(hostname) — node is dirty:"; echo "$busy"; exit 1
fi

cleanup() {
    [[ -n "${PGID:-}" ]] && { kill -TERM -"$PGID" 2>/dev/null; sleep 10; kill -KILL -"$PGID" 2>/dev/null; }
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [[ "$(ps -o user= -p "$pid" 2>/dev/null)" == "$USER" ]] && kill -KILL "$pid" 2>/dev/null
    done
    PGID=""
}
trap cleanup EXIT TERM INT

declare -a RESULTS
run() {  # name, then the command
    local name=$1; shift
    echo; echo "########## TEST $name: $* ##########"
    setsid "$@" &
    PGID=$!
    wait "$PGID"; local status=$?
    cleanup; sleep 5
    echo "########## TEST $name exit=$status ##########"
    RESULTS+=("$name: $([[ $status == 0 ]] && echo PASS || echo "FAIL (exit $status)")")
    return $status
}

COMMON="--tensor-parallel-size 2 --max-model-len 16384 --batch-size 5"
run A  python kv_cache_vllm.py --model Qwen/Qwen3-4B $COMMON --results-path $OUT/A_kv_qwen3-4b_tp2.csv \
  || run A-mp1 env VLLM_ENABLE_V1_MULTIPROCESSING=1 python kv_cache_vllm.py --model Qwen/Qwen3-4B $COMMON --results-path $OUT/A-mp1_kv_qwen3-4b_tp2.csv
run B  python kv_cache_vllm.py --model Qwen/Qwen3.5-4B --dtype bfloat16 --language-model-only $COMMON --results-path $OUT/B_kv_qwen3.5-4b_tp2.csv
run C  python judge_test.py $COMMON --results-path $OUT/C_judge_nemotron8b_tp2.csv
run D  python judge_test.py --model Qwen/Qwen3.5-4B --language-model-only $COMMON --results-path $OUT/D_judge_qwen3.5-4b_tp2.csv

echo; echo "========== SMOKE SUMMARY =========="; printf '%s\n' "${RESULTS[@]}"; echo "End: $(date)"
