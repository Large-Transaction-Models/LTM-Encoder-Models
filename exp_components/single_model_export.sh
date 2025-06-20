# Run the experiment script
python load_pretrained_model.py \
    --exp_name aave_seqLen10\
    --dataset Aave_V2_Mainnet \
    --seed 42 \
    --checkpoint -1 \
    --checkpoint_dir checkpoints \
    --logging_dir logs \
    --log_file_name export_output.log \
    --num_train_epochs 50 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --stride 1 \
    --seq_len 10 \
    --vocab_dir vocab \
    --train_test_thres 0.7 \
    --num_bins 4 \
    --nrows -1 \
    --time_pos_type regular_position \
    --mlm_prob 0.15 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --get_rids \
    --field_hs 8 \
     \
     \
     --include_time_features \
     --include_user_features \
     --include_market_features \
     \
    --check_preprocess_cached \
    --check_preload_cached \
    --check_process_cached \
    --pad_seq_first \
    --long_and_sort \
     \
    --mlm \
    
