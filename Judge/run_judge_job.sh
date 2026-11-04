#!/bin/bash
#SBATCH --job-name=judge_triggers
#SBATCH --output=/shared/2/projects/Bias_Job_Descriptions/slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:A6000:2

# Load environment
export PATH=/opt/anaconda/bin:$PATH
source activate base

# Store HF models in shared storage
export HF_HOME="/shared/4/models/"
# FlashInfer sampler can mismatch the node driver (see gpu-and-serving guide)
export VLLM_USE_FLASHINFER_SAMPLER=0

# Fail loudly on a dirty GPU instead of fighting for memory
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU already in use on $(hostname); aborting" >&2
    exit 1
fi

cd /shared/2/projects/Bias_Job_Descriptions/

# Pass --sample for a quick check; resumes from the output JSONL otherwise

## run batch file as 
# sbatch run_judge_job.sh \
#     --input-csv /shared/2/projects/Bias_Job_Descriptions/Job_descriptions_nemotron.csv \
#     --output /shared/2/projects/Bias_Job_Descriptions/judge_all_personas.jsonl \
#     --output-csv /shared/2/projects/Bias_Job_Descriptions/judge_all_personas.csv


/opt/anaconda/bin/python -u /shared/2/projects/Bias_Job_Descriptions/judge_trigger_signals_vllm_final.py "$@"


