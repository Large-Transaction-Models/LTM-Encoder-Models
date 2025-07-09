# Run the experiment script
python load_pretrained_model.py \
    --exp_name aave_testing\
    --dataset Aave_V2_Mainnet \
    --seed 42 \
    --checkpoint -1 \
    --checkpoint_dir checkpoints \
    --logging_dir logs \
    --log_file_name export_output.log \
    --num_train_epochs 5 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --stride 10 \
    --seq_len 10 \
    --vocab_dir vocab \
    --train_test_thres 0.7 \
    --num_bins 10 \
    --nrows 100000 \
    --time_pos_type time_aware_sin_cos_position \
    --mlm_prob 0.15 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --get_rids \
    --field_hs 8 \
    --num_attention_heads 12 \
    --static_features \
     \
     \
     --include_time_features \
     --include_user_features \
     --include_market_features \
     \
    --check_preprocess_cached \
    --check_preload_cached \
    --check_process_cached \
    --do_pretrain \
    --pad_seq_first \
    --long_and_sort \a
     \
    --mlm \