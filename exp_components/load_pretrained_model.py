import torch
from models.modules import TabFormerBertLM  # or TabFormerBertForClassification

import os
from os.path import join, basename
import sys
sys.path.append(os.path.abspath('../'))
from dataset.dataset import Dataset

import numpy as np
import torch
import random
import json
def load_pretrained_model(args, data_path, feature_extension, log):

    train_fname = f"transactions{feature_extension}_train"  
    val_fname = f"transactions{feature_extension}_val" 
    test_fname = f"transactions{feature_extension}_test"  
    output_dir=f"{data_path}output/{args.exp_name}"
    checkpoint=args.checkpoint
    vocab_dir=f"{data_path}vocab"
    
    random.seed(args.seed)  # python
    np.random.seed(args.seed)  # numpy
    torch.manual_seed(args.seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)  # torch.cuda

    # return labels when classification
    args.return_labels = args.cls_task
    
    # Path to wherever run_experiment.py wrote the final checkpoint
    dataset = Dataset(cls_task=args.cls_task or args.mlm,
                       seq_len=args.seq_len,
                       root=data_path,
                       fname=train_fname,
                       user_level_cached=True, # We should only be running export after we've created the user-level data in process()
                       vocab_cached=True, # the vocab should have been created in preload, which needs to be done before this.
                       external_vocab_path=f"{vocab_dir}/vocab_ob",
                       nrows=args.nrows,
                       flatten=args.flatten,
                       stride=args.stride,
                       return_labels=args.return_labels,
                       label_category=args.label_category,
                       pad_seq_first=args.pad_seq_first,
                       get_rids=args.get_rids,
                       long_and_sort=args.long_and_sort,
                       resample_method=args.resample_method,
                       resample_ratio=args.resample_ratio,
                       resample_seed=args.resample_seed,
                       )

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()
    log.info(f'vocab size: {len(vocab)}')
    
    net = TabFormerBertLM(custom_special_tokens,
                              vocab=vocab,
                              field_ce=args.field_ce,
                              flatten=args.flatten,
                              ncols=dataset.ncols,
                              field_hidden_size=args.field_hs,
                              time_pos_type=args.time_pos_type)
    
    # 2) Load the saved state_dict (the weights):
    final_model_path = f"{data_path}/{args.checkpoint_dir}/checkpoint-50000/"
    state_dict = torch.load(f"{final_model_path}pytorch_model.bin", map_location="cpu")

    # 3) Because TabFormerBertLM often has a self.model inside, check if you need:
    #    net.model.load_state_dict(state_dict)
    # or 
    #    net.load_state_dict(state_dict)
    
    # If your training code stored weights directly into `net.model`, do:
    net.model.load_state_dict(state_dict)
    
    # 4) Put model in eval mode and move to GPU if desired
    net.eval()
    net.cuda()  # if you have CUDA
    
    # Now `net` is ready for inference or extracting CLS embeddings

    return net
