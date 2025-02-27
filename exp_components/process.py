import os
import sys
sys.path.append(os.path.abspath('../'))

from dataset.dataset import Dataset
from dataset.basic import BasicDataset


def process(args, data_path, feature_extension, log):

    vocab_dir=f"{data_path}{args.vocab_dir}"
    train_fname = f"transactions{feature_extension}_train"  
    val_fname = "" 
    test_fname = f"transactions{feature_extension}_test"  

    if args.check_process_cached:
        if os.path.exists(f"{data_path}{train_fname}.user.pkl") and os.path.exists(f"{data_path}{test_fname}.user.pkl"):
            log.info("Data has already been processed. Skipping process.")
            return
    

    dataset = Dataset(cls_task=True,
                      seq_len=args.seq_len,
                      root=data_path,
                      fname=train_fname,
                      preload_vocab_dir=vocab_dir,
                      save_vocab_dir=vocab_dir,
                      nrows=args.nrows,
                      flatten=args.flatten,
                      stride=args.stride,
                      return_labels=True,
                      label_category='last_label',
                      pad_seq_first=args.pad_seq_first,
                      get_rids=args.get_rids,
                      long_and_sort=args.long_and_sort,
                      resample_method=args.resample_method,
                      resample_ratio=args.resample_ratio,
                      resample_seed=args.resample_seed)


    external_vocab_path=dataset.vocab_path
    vocab_cached=True
    encoder_cached=True

    test_dataset = Dataset(cls_task=True,
                          seq_len=args.seq_len,
                          root=data_path,
                          fname=test_fname,
                          vocab_cached=vocab_cached,
                          preload_vocab_dir=vocab_dir,
                          external_vocab_path=external_vocab_path,
                          save_vocab_dir=vocab_dir,
                          nrows=args.nrows,
                          flatten=args.flatten,
                          stride=args.stride,
                          return_labels=True,
                          label_category='last_label',
                          pad_seq_first=args.pad_seq_first,
                          get_rids=args.get_rids,
                          long_and_sort=args.long_and_sort,
                          resample_method=args.resample_method,
                          resample_ratio=args.resample_ratio,
                          resample_seed=args.resample_seed)