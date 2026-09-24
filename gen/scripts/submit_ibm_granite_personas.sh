#!/bin/bash
#SBATCH --job-name=generate_personas_ibm_granite
#SBATCH --account=shrujaya
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --output=generate_personas_ibm_granite_%j.out

MODEL="/shared/4/models/models--ibm-granite--granite-4.2-8b/snapshots/7fce579a7fbbad4b1e7703b6850cefd517a4002b"

# Input CSV and outputs
INPUT_FILE="CSV_Dataset_Files/Sample_job_descriptions_balanced_300.csv"
OUT_DIR="gen/out/personas_output_ibm_granite"

# Number of personas per job description
NUM_PERSONAS=10

# Number of rows processed per chunk
# Adjust this chunk size to balance model loading overhead and check-pointing frequency
CHUNK_SIZE=50

# Optional: Process only a subset of rows (useful for parallel jobs)
# Set via SLURM --export flag or change here
START_ROW="${START:-0}"
END_ROW="${END:-300}"
USER_OUT_DIR="${OUT_DIR:-}"
APPEND_FIRST="${APPEND_FIRST:-false}"

## ---- RUN ----
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "========================================================================"
echo "Job started on $(hostname) at $(date)"
echo "========================================================================"
echo "Input: $INPUT_FILE"
echo "Model: $MODEL"
echo "Personas per description: $NUM_PERSONAS"
echo "Chunk size: $CHUNK_SIZE"
echo "Processing rows: $START_ROW to $END_ROW"
echo "========================================================================"

# Create unique output directory for this job (to allow parallel jobs)
if [ -n "$USER_OUT_DIR" ]; then
    OUT_DIR="$USER_OUT_DIR"
elif [ "$START_ROW" -eq 0 ] && [ "$END_ROW" -eq 300 ]; then
    # Single job processing all rows
    OUT_DIR="personas_output_qwen"
else
    # Parallel job - use unique directory
    OUT_DIR="personas_output_qwen_${SLURM_JOB_ID:-temp}_${START_ROW}_${END_ROW}"
fi

mkdir -p "$OUT_DIR"
mkdir -p logs

# Count rows to process
TOTAL_TO_PROCESS=$((END_ROW - START_ROW))
echo "Total rows to process: $TOTAL_TO_PROCESS"

# Process rows in chunks
# First chunk without --append (unless APPEND_FIRST is true), subsequent chunks with --append to accumulate results
CHUNK_NUM=0
CURRENT_START=$START_ROW

echo ""
echo "Starting processing..."
echo ""

while [ $CURRENT_START -lt $END_ROW ]; do
  CHUNK_NUM=$((CHUNK_NUM + 1))
  CURRENT_END=$((CURRENT_START + CHUNK_SIZE))
  if [ $CURRENT_END -gt $END_ROW ]; then
    CURRENT_END=$END_ROW
  fi

  ROWS_IN_CHUNK=$((CURRENT_END - CURRENT_START))
  echo "[$CHUNK_NUM] Processing rows $CURRENT_START .. $((CURRENT_END-1)) ($ROWS_IN_CHUNK rows)"

  # Run Python script
  if [ $CURRENT_START -eq $START_ROW ] && [ "$APPEND_FIRST" != "true" ]; then
    # First chunk: overwrite any existing output files
    python persona_generation.py \
      --input "$INPUT_FILE" \
      --outdir "$OUT_DIR" \
      --model "$MODEL" \
      --n "$NUM_PERSONAS" \
      --start $CURRENT_START --end $CURRENT_END \
      2>&1 | tee -a "logs/chunk_$CHUNK_NUM.log"
  else
    # Subsequent chunks (or first chunk if APPEND_FIRST is true): append to existing output
    python persona_generation.py \
      --input "$INPUT_FILE" \
      --outdir "$OUT_DIR" \
      --model "$MODEL" \
      --n "$NUM_PERSONAS" \
      --start $CURRENT_START --end $CURRENT_END \
      --append \
      2>&1 | tee -a "logs/chunk_$CHUNK_NUM.log"
  fi

  # In bash with set -e, the command failure would already trigger exit, but keeping check/safeguard
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Chunk $CHUNK_NUM failed! Check logs/chunk_$CHUNK_NUM.log"
    exit 1
  fi

  CURRENT_START=$CURRENT_END
done

echo ""
echo "========================================================================"
echo "All $CHUNK_NUM chunks processed successfully!"
echo "========================================================================"
echo "Output files:"
echo "  JSONL: $OUT_DIR/personas.jsonl (one JSON line per persona generation)"
echo "Logs:   logs/chunk_*.log"
echo "========================================================================"
echo "Job finished at $(date)"
