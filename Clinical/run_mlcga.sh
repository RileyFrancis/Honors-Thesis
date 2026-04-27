#!/bin/bash
#SBATCH --job-name=mlcga
#SBATCH --account=jib10001
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mlcga_%j.out
#SBATCH --error=mlcga_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rif17002@uconn.edu

echo "========================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Source conda
source ~/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate honors

# Verify
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "R version: $(R --version | head -n1)"

# Change to working directory
cd /home/rif17002/honors_thesis

# Run the script
echo "Running Clinical/mlcga_opt.R"
Rscript Clinical/mlcga_opt.R

echo "========================================"
echo "Job finished: $(date)"
echo "========================================"