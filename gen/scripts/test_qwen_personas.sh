#!/bin/bash
#SBATCH --job-name=test_personas_qwen
#SBATCH --account=shrujaya
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=test_personas_qwen_%j.out

MODEL="/shared/4/models/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

# Input CSV and outputs
INPUT_FILE="CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv"
OUT_DIR="gen/out/personas_output_qwen_test"

# Number of personas per job description
NUM_PERSONAS=10

# Number of rows processed per chunk
CHUNK_SIZE=1

# Process exactly 1 row (row 0 to 1, exclusive) for testing
START_ROW=0
END_ROW=1

## ---- RUN ----
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "========================================================================"
echo "TEST JOB: Running 10 generations for 1 job description"
echo "Job started on $(hostname) at $(date)"
echo "========================================================================"
echo "Input: $INPUT_FILE"
echo "Model: $MODEL"
echo "Personas per description: $NUM_PERSONAS"
echo "Processing row range: $START_ROW to $END_ROW"
echo "========================================================================"

mkdir -p "$OUT_DIR"
mkdir -p logs

# Run Python script for the single row test
python persona_generation.py \
  --input "$INPUT_FILE" \
  --outdir "$OUT_DIR" \
  --model "$MODEL" \
  --n "$NUM_PERSONAS" \
  --start $START_ROW --end $END_ROW \
  2>&1 | tee -a "logs/test_run.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "ERROR: Test job failed! Check logs/test_run.log"
  exit 1
fi

echo ""
echo "========================================================================"
echo "Test run completed successfully!"
echo "========================================================================"
echo "Output files:"
echo "  JSONL: $OUT_DIR/personas.jsonl"
echo "Logs:   logs/test_run.log"
echo "========================================================================"
echo "Job finished at $(date)"
