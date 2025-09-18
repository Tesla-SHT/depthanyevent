#!/bin/bash

# -----------------------------------------------------------------------------
# Script to set up the environment for the DepthAnyEvent project.
#
# Usage:
#   bash scripts/test.sh
#
# What this script does:
#   1. Sources the conda.sh script to enable conda commands in the shell.
#   2. Activates the 'depthanyevent' conda environment.
#   3. Sets the CODEBASE_PATH variable to the root directory of the project.
#
# Notes for users:
#   - Ensure that Miniconda or Anaconda is installed on your system.
#   - Modify the MINICONDA_PROFILE_PATH, CONDA_ENV_NAME, and CODEBASE_PATH variables
#     to match your local setup.
#   - This script assumes that the conda environment and project directory already exist.
# -----------------------------------------------------------------------------

MINICONDA_PROFILE_PATH="/home/luca/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="depthanyevent"
CODEBASE_PATH="/home/luca/Scrivania/projects/depthanyevent/"

source "$MINICONDA_PROFILE_PATH"
conda activate "$CONDA_ENV_NAME"
cd "$CODEBASE_PATH"

export CUDA_VISIBLE_DEVICES=0

# Tab 1

python test.py --config configs/test/dav2/dav2_mvsec_test.json --csv_path results/tab1_dav2_mvsec.csv --loadmodel weights/dav2/synth/synth.pth
python test.py --config configs/test/recdav2/rec_dav2_mvsec_test.json --csv_path results/tab1_rec_dav2_mvsec.csv --loadmodel weights/rec_dav2/synth/synth.pth
python test.py --config configs/test/dav2/dav2_dsec_test.json --csv_path results/tab1_dav2_dsec.csv --loadmodel weights/dav2/synth/synth.pth
python test.py --config configs/test/recdav2/rec_dav2_dsec_test.json --csv_path results/tab1_rec_dav2_dsec.csv --loadmodel weights/rec_dav2/synth/synth.pth

# # Tab 2

python test.py --config configs/test/dav2/dav2_mvsec_test.json --csv_path results/tab2_dav2_finetuned_mvsec.csv --loadmodel weights/dav2/finetuned_mvsec/finetuned_mvsec.pth
python test.py --config configs/test/recdav2/rec_dav2_mvsec_test.json --csv_path results/tab2_rec_dav2_finetuned_mvsec.csv --loadmodel weights/rec_dav2/finetuned_mvsec/finetuned_mvsec.pth
python test.py --config configs/test/dav2/dav2_dsec_test.json --csv_path results/tab2_dav2_finetuned_dsec.csv --loadmodel weights/dav2/finetuned_dsec/finetuned_dsec.pth
python test.py --config configs/test/recdav2/rec_dav2_dsec_test.json --csv_path results/tab2_rec_dav2_finetuned_dsec.csv --loadmodel weights/rec_dav2/finetuned_dsec/finetuned_dsec.pth

# Tab 3

python test.py --config configs/test/dav2/dav2_mvsec_test.json --csv_path results/tab3_dav2_distillation_mvsec.csv --loadmodel weights/dav2/distillation_mvsec/distillation_mvsec.pth
python test.py --config configs/test/recdav2/rec_dav2_mvsec_test.json --csv_path results/tab3_rec_dav2_distillation_mvsec.csv --loadmodel weights/rec_dav2/distillation_mvsec/distillation_mvsec.pth
python test.py --config configs/test/dav2/dav2_dsec_test.json --csv_path results/tab3_dav2_distillation_dsec.csv --loadmodel weights/dav2/distillation_dsec/distillation_dsec.pth
python test.py --config configs/test/recdav2/rec_dav2_dsec_test.json --csv_path results/tab3_rec_dav2_distillation_dsec.csv --loadmodel weights/rec_dav2/distillation_dsec/distillation_dsec.pth

# Tab 4

python test.py --config configs/test/dav2/dav2_dsec_test.json --csv_path results/tab4_dav2_metric_dsec.csv --loadmodel weights/dav2/metric_dsec/metric_dsec.pth
python test.py --config configs/test/recdav2/rec_dav2_dsec_test.json --csv_path results/tab4_rec_dav2_metric_dsec.csv --loadmodel weights/rec_dav2/metric_dsec/metric_dsec.pth
