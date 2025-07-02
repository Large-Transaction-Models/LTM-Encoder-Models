#!/bin/bash
#SBATCH --job-name=LTM_pretraining_test_slurmTest1.sh
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run the experiment script
python run_experiment.py \
    --exp_name slurmTest1 \
    --dataset cosmetics \
    --seed 42 \
    --checkpoint 0 \
    --checkpoint_dir checkpoints \
    --logging_dir logs \
    --log_file_name output.log \
    --num_train_epochs 5 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --stride 5 \
    --seq_len 10 \
    --vocab_dir vocab \
    --train_test_thres 0.6 \
    --num_bins 4 \
    --nrows -1 \
    --resample_method None \
    --resample_ratio 10 \
    --resample_seed 100 \
    --time_pos_type regular_position \
    --mlm_prob 0.15 \
    --save_steps 10000 \
    --eval_steps 10000 \
     \
    --include_user_features \
    --include_time_features \
     \
     \
    --check_preprocess_cached \
    --check_preload_cached \
    --check_process_cached \
    --do_pretrain \
    --pad_seq_first \
    --long_and_sort \
     \
    --mlm \
     \
     \
     \
     \
    
