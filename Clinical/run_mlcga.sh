#!/bin/bash
#SBATCH --job-name=mlcga
#SBATCH --account=jib10001
#SBATCH --partition=lo-core
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=mlcga_%j.out
#SBATCH --error=mlcga_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rif17002@uconn.edu

echo "========================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh
# OR if you use anaconda:
# source ~/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate honors

# Verify R is available
echo "R version:"
R --version

# Set working directory
cd /home/rif17002/honors_thesis

# List R scripts to verify
echo "R scripts in directory:"
ls -la *.R

# Run analysis
Rscript Clinical/mlcga_opt.R

echo "========================================"
echo "Job finished: $(date)"
echo "========================================"