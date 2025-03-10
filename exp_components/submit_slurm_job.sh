#!/bin/bash

# Default SLURM parameters
PARTITION="gpu"
GPUS=8
TIME="6:00:00"
JOB_NAME="LTM_pretraining_test"

# Default experiment parameters (from arguments.py)
EXP_NAME="slurmTest1"
DATASET="cosmetics"
SEED=42
CHECKPOINT=0
CHECKPOINT_DIR="checkpoints"
LOGGING_DIR="logs"
LOG_FILE="output.log"
NUM_EPOCHS=20
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
STRIDE=5
SEQ_LEN=10
VOCAB_DIR="vocab"
TRAIN_TEST_THRES=0.6
NUM_BINS=10
NROWS=-1
RESAMPLE_METHOD=None
RESAMPLE_RATIO=10
RESAMPLE_SEED=100
TIME_POS_TYPE="regular_position"
MLM_PROB=0.15
SAVE_STEPS=10000
EVAL_STEPS=10000
FREEZE=false

# Boolean flags
INCLUDE_USER_FEATURES=true
INCLUDE_TIME_FEATURES=true
INCLUDE_MARKET_FEATURES=false
INCLUDE_EXO_FEATURES=false
CHECK_PREPROCESS_CACHED=true
CHECK_PRELOAD_CACHED=true
CHECK_PROCESS_CACHED=true
DO_PRETRAIN=true
PAD_SEQ_FIRST=true
LONG_AND_SORT=true
FLATTEN=false
MLM=true
CLS_TASK=false
REG_TASK=false
EXPORT_TASK=false
EXPORT_LAST_ONLY=false
EXPORT_CLS_EMBEDDINGS=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition) PARTITION="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --job_name) JOB_NAME="$2"; shift ;;
        --exp_name) EXP_NAME="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift ;;
        --logging_dir) LOGGING_DIR="$2"; shift ;;
        --log_file) LOG_FILE="$2"; shift ;;
        --num_train_epochs) NUM_EPOCHS="$2"; shift ;;
        --nrows) NROWS="2"; shift ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift ;;
        --stride) STRIDE="$2"; shift ;;
        --seq_len) SEQ_LEN="$2"; shift ;;
        --vocab_dir) VOCAB_DIR="$2"; shift ;;
        --train_test_thres) TRAIN_TEST_THRES="$2"; shift ;;
        --num_bins) NUM_BINS="$2"; shift ;;
        --resample_method) RESAMPLE_METHOD="$2"; shift ;;
        --resample_ratio) RESAMPLE_RATIO="$2"; shift ;;
        --resample_seed) RESAMPLE_SEED="$2"; shift ;;
        --time_pos_type) TIME_POS_TYPE="$2"; shift ;;
        --mlm_prob) MLM_PROB="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --eval_steps) EVAL_STEPS="$2"; shift ;;
        --freeze) FREEZE=true ;;
        --include_user_features) INCLUDE_USER_FEATURES=true ;;
        --include_time_features) INCLUDE_TIME_FEATURES=true ;;
        --include_market_features) INCLUDE_MARKET_FEATURES=true ;;
        --include_exo_features) INCLUDE_EXO_FEATURES=true ;;
        --check_preprocess_cached) CHECK_PREPROCESS_CACHED=true ;;
        --check_preload_cached) CHECK_PRELOAD_CACHED=true ;;
        --check_process_cached) CHECK_PROCESS_CACHED=true ;;
        --do_pretrain) DO_PRETRAIN=true ;;
        --pad_seq_first) PAD_SEQ_FIRST=true ;;
        --long_and_sort) LONG_AND_SORT=true ;;
        --flatten) FLATTEN=true ;;
        --mlm) MLM=true ;;
        --cls_task) CLS_TASK=true ;;
        --reg_task) REG_TASK=true ;;
        --export_task) EXPORT_TASK=true ;;
        --export_last_only) EXPORT_LAST_ONLY=true ;;
        --export_cls_embeddings) EXPORT_CLS_EMBEDDINGS=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Create a SLURM batch script dynamically
SLURM_SCRIPT="${JOB_NAME}_${EXP_NAME}.sh"

cat <<EOT > $SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=$SLURM_SCRIPT
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --time=$TIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run the experiment script
python run_experiment.py \\
    --exp_name $EXP_NAME \\
    --dataset $DATASET \\
    --seed $SEED \\
    --checkpoint $CHECKPOINT \\
    --checkpoint_dir $CHECKPOINT_DIR \\
    --logging_dir $LOGGING_DIR \\
    --log_file_name $LOG_FILE \\
    --num_train_epochs $NUM_EPOCHS \\
    --train_batch_size $TRAIN_BATCH_SIZE \\
    --eval_batch_size $EVAL_BATCH_SIZE \\
    --stride $STRIDE \\
    --seq_len $SEQ_LEN \\
    --vocab_dir $VOCAB_DIR \\
    --train_test_thres $TRAIN_TEST_THRES \\
    --num_bins $NUM_BINS \\
    --nrows $NROWS \\
    --resample_method $RESAMPLE_METHOD \\
    --resample_ratio $RESAMPLE_RATIO \\
    --resample_seed $RESAMPLE_SEED \\
    --time_pos_type $TIME_POS_TYPE \\
    --mlm_prob $MLM_PROB \\
    --save_steps $SAVE_STEPS \\
    --eval_steps $EVAL_STEPS \\
    $( [[ "$FREEZE" == true ]] && echo "--freeze" ) \\
    $( [[ "$INCLUDE_USER_FEATURES" == true ]] && echo "--include_user_features" ) \\
    $( [[ "$INCLUDE_TIME_FEATURES" == true ]] && echo "--include_time_features" ) \\
    $( [[ "$INCLUDE_MARKET_FEATURES" == true ]] && echo "--include_market_features" ) \\
    $( [[ "$INCLUDE_EXO_FEATURES" == true ]] && echo "--include_exo_features" ) \\
    $( [[ "$CHECK_PREPROCESS_CACHED" == true ]] && echo "--check_preprocess_cached" ) \\
    $( [[ "$CHECK_PRELOAD_CACHED" == true ]] && echo "--check_preload_cached" ) \\
    $( [[ "$CHECK_PROCESS_CACHED" == true ]] && echo "--check_process_cached" ) \\
    $( [[ "$DO_PRETRAIN" == true ]] && echo "--do_pretrain" ) \\
    $( [[ "$PAD_SEQ_FIRST" == true ]] && echo "--pad_seq_first" ) \\
    $( [[ "$LONG_AND_SORT" == true ]] && echo "--long_and_sort" ) \\
    $( [[ "$FLATTEN" == true ]] && echo "--flatten" ) \\
    $( [[ "$MLM" == true ]] && echo "--mlm" ) \\
    $( [[ "$CLS_TASK" == true ]] && echo "--cls_task" ) \\
    $( [[ "$REG_TASK" == true ]] && echo "--reg_task" ) \\
    $( [[ "$EXPORT_TASK" == true ]] && echo "--export_task" ) \\
    $( [[ "$EXPORT_LAST_ONLY" == true ]] && echo "--export_last_only" ) \\
    $( [[ "$EXPORT_CLS_EMBEDDINGS" == true ]] && echo "--export_cls_embeddings" )
EOT

# Submit the job
bash $SLURM_SCRIPT
