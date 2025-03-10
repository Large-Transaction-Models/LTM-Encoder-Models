#!/bin/bash

# Set lists of argument values to iterate over
DATASETS=("Aave_V2_Mainnet" "Aave_V3_Polygon", "cosmetics", "electronics", "AML_LI_Small")  # Different datasets
NUM_EPOCHS_LIST=(5 10 20)  # Different training epochs
TRAIN_BATCH_SIZES=(8 16)  # Different batch sizes
GPUS_LIST=(8)  # Number of GPUs
TIME_LIST=("12:00:00" "24:00:00")  # Different time limits

# Flags (optional: set to true/false for different runs)
INCLUDE_USER_FEATURES="--include_user_features"
CHECK_PREPROCESS_CACHED="--check_preprocess_cached"
CHECK_PRELOAD_CACHED="--check_preload_cached"

# Iterate over all combinations of parameter values
for DATASET in "${DATASETS[@]}"; do
  for NUM_EPOCHS in "${NUM_EPOCHS_LIST[@]}"; do
    for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
      for GPUS in "${GPUS_LIST[@]}"; do
        for TIME in "${TIME_LIST[@]}"; do
          
          # Generate a unique experiment name based on parameter choices
          EXP_NAME="exp_${DATASET}_epochs${NUM_EPOCHS}_bs${TRAIN_BATCH_SIZE}_gpu${GPUS}_time${TIME//:/}"

          # Submit the job with the generated experiment name
          ./submit_slurm_job.sh \
            --exp_name "$EXP_NAME" \
            --dataset "$DATASET" \
            --num_train_epochs "$NUM_EPOCHS" \
            --train_batch_size "$TRAIN_BATCH_SIZE" \
            --gpus "$GPUS" \
            --time "$TIME" \
            $INCLUDE_USER_FEATURES \
            $CHECK_PREPROCESS_CACHED \
            $CHECK_PRELOAD_CACHED

          # Optional: Add a short sleep to prevent overloading SLURM
          sleep 1

        done
      done
    done
  done
done
