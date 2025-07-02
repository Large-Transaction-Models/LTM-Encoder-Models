import os
import sys
sys.path.append(os.path.abspath('../'))
from dataset.preload import PreloadDataset
import os.path as path
from dataset.vocab import *

def preload(args, data_path, feature_extension, log):
    
    # Root and file configurations
    train_fname = f"transactions{feature_extension}_train"  
    test_fname = f"transactions{feature_extension}_test"  

    if args.check_preload_cached:
        if os.path.exists(f"{data_path}preloaded_{args.num_bins}_{args.nrows}_seqLen{args.seq_len}/{train_fname}.encoded.csv") and os.path.exists(f"{data_path}preloaded_{args.num_bins}_{args.nrows}_seqLen{args.seq_len}/{test_fname}.encoded.csv"):
            log.info("Data has already been preloaded. Skipping preload.")
            return
    
    # Instantiate the dataset for preloading
    dataset = PreloadDataset(
        num_bins=args.num_bins,
        root=data_path,
        fname=train_fname,
        vocab_dir=args.vocab_dir,
        nrows=args.nrows,
        adap_thres=args.adap_thres,
        get_rids=args.get_rids,
        columns_to_select=None
    )
    
    # Print out vocab sizes for verification
    log.info(f"Vocab sizes: {len(dataset.dynamic_vocab)}, {len(dataset.time_feature_vocab)}, {len(dataset.static_vocab)}.")
    

    print(f"dataset.encoder_path: {dataset.encoder_path}")
    external_encoder_path = dataset.encoder_path

    test_dataset = PreloadDataset(
        num_bins=args.num_bins,
        encoder_cached=True,  # Force generation of a new encoder if missing
        external_encoder_path=external_encoder_path,  # Should be empty or point to existing encoder
        vocab_cached=True,
        root=data_path,
        fname=test_fname,
        vocab_dir=args.vocab_dir,
        nrows=args.nrows,
        adap_thres=args.adap_thres,
        get_rids=args.get_rids,
        columns_to_select=None
    )
    
    # Print out vocab sizes for the test dataset to ensure consistency
    log.info(f"Vocab sizes: {len(test_dataset.dynamic_vocab)}, {len(test_dataset.time_feature_vocab)}, {len(test_dataset.static_vocab)}.")
