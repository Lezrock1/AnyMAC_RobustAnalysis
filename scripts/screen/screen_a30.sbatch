#!/usr/bin/env bash
#SBATCH --job-name=screen
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_screen_%A_%a.log
# One A30 per model (array index = line in models.txt). Resubmitting resumes finished episodes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
exec bash scripts/screen/run.sh run
