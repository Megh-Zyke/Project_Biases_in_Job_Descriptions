#!/bin/bash
#SBATCH --job-name=test_personas_olmo
#SBATCH --account=shrujaya
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=test_personas_olmo_%j.out

MODEL="/shared/4/models/models--allenai--OLMo-2-1124-13B-Instruct/snapshots/3a5c85baefbb1896a54d56fe2e76c0395627ddf4"

# Input CSV and outputs
INPUT_FILE="CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv"
OUT_DIR="gen/out/personas_output_olmo_test"
OUT_FILE="$OUT_DIR/personas.csv"

# Number of personas per job description
NUM_PERSONAS=10

# Process exactly 1 row (row 0 to 1, exclusive) for testing
START_ROW=0
END_ROW=1

## ---- RUN ----
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "========================================================================"
echo "TEST JOB: Running 10 generations for 1 job description (Olmo)"
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
python persona_generation/persona_generation.py \
  --data-path "$INPUT_FILE" \
  --output "$OUT_FILE" \
  --model-path "$MODEL" \
  --n "$NUM_PERSONAS" \
  --start $START_ROW --end $END_ROW \
  2>&1 | tee -a "logs/test_olmo_run.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "ERROR: Test job failed! Check logs/test_olmo_run.log"
  exit 1
fi

echo ""
echo "========================================================================"
echo "Test run completed successfully!"
echo "========================================================================"
echo "Output files:"
echo "  CSV:  $OUT_FILE"
echo "Logs:   logs/test_olmo_run.log"
echo "========================================================================"
echo "Job finished at $(date)"
