# Run the experiment script
python run_experiment.py \
    --exp_name test_export \
    --dataset electronics \
    --seed 42 \
    --checkpoint 0 \
    --checkpoint_dir checkpoints \
    --logging_dir logs \
    --log_file_name output.log \
    --vocab_dir vocab \
    --time_pos_type regular_position \
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
    --get_rids \
     \
     \
     \
     \
     \
    
