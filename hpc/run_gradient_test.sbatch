#!/bin/bash
#SBATCH --job-name=switi_gradient_test
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_gradient_test/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-2

DSET_LIST=(HT_LIF24_500ms HT_LIF24_5ms PaviaATN)
TILE_SIZE_LIST=("64 64" "64 64" "64 64")
OVERLAP_LIST=("32 32" "32 32" "32 32")

# DSET_LIST=(CARE3D_liver CARE3D_zebrafish HT_LIF24_500ms HT_LIF24_5ms PaviaATN)
# TILE_SIZE_LIST=("5 64 64" "5 64 64" "64 64" "64 64" "64 64")
# OVERLAP_LIST=("2 32 32" "2 32 32" "32 32" "32 32" "32 32")

DATASET=${DSET_LIST[$SLURM_ARRAY_TASK_ID]}
TILE_SIZE=${TILE_SIZE_LIST[$SLURM_ARRAY_TASK_ID]}
OVERLAP=${OVERLAP_LIST[$SLURM_ARRAY_TASK_ID]}

python scripts/run_gradient_test_on_dataset.py \
    --dataset $DATASET \
    --predictions_subdir predictions_MMSE64 \
    --methods SWITi \ # inner_tiling
    --tile_size $TILE_SIZE \
    --overlap $OVERLAP \
    --output_root /project/careamics/switi/results/$DATASET/predictions_MMSE64/