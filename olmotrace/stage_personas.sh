#!/bin/bash
#SBATCH --job-name=olmo_stages
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nice=400
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --chdir=/shared/5/shrujaya/olmo_stages
#SBATCH --output=/shared/5/shrujaya/olmo_stages/logs/%x-%A_%a.out
# Persona generation for one OLMo-2-13B training stage per array task.
# Submit: sbatch --gres=gpu:A6000:1 --array=0-2%3 --export=ALL,PROJ_DIR=/home/shrujaya/Project_Biases_in_Job_Descriptions stage_personas.sh
# Smoke test: add END=2 N=2 OUT_SUFFIX=_smoke to --export
set -euo pipefail

STAGES=(base sft dpo)
STAGE=${STAGES[$SLURM_ARRAY_TASK_ID]}
ROOT=/shared/5/shrujaya/olmo_stages
declare -A MODEL=([base]=OLMo-2-1124-13B [sft]=OLMo-2-1124-13B-SFT [dpo]=OLMo-2-1124-13B-DPO)
OUT="$ROOT/out/personas_${STAGE}${OUT_SUFFIX:-}.csv"

echo "[$(date)] host=$(hostname) stage=$STAGE gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
python "$PROJ_DIR/olmotrace/stage_personas.py" --stage "$STAGE" \
  --model-path "$ROOT/models/${MODEL[$STAGE]}" --output "$OUT" \
  --n "${N:-10}" ${END:+--end "$END"} --budget-hours 3.2
python -c "import pandas as pd,sys; print('[done] rows:', len(pd.read_csv(sys.argv[1])))" "$OUT"
echo "[$(date)] finished"
