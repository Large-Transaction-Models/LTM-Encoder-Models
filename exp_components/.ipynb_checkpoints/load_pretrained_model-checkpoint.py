import torch
import os
from os.path import join, basename
import sys
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath('../'))
from dataset.dataset import Dataset
from models.modules import TabFormerBertLM, TabFormerBertModel  # or TabFormerBertForClassification

from utils import setup_logging

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

import numpy as np
import torch
import random
import json

import logging

from pathlib import Path
from arguments import parse_arguments
from utils import get_data_path, find_latest_checkpoint
from dataset.datacollator import *
# The first step is to find the proper directory where the pretrained model lies:

args = parse_arguments()

data_path, feature_extension = get_data_path(args)

log = setup_logging(log_dir=f"{data_path}preprocessed{feature_extension}/preloaded_{args.num_bins}_{args.nrows}/{args.logging_dir}/{args.exp_name}", log_file_name = args.log_file_name) 

data_path, feature_extension = get_data_path(args)
if(args.nrows == -1):
    args.nrows = None
data_path += f"preprocessed{feature_extension}/preloaded_{args.num_bins}_{args.nrows}/"

checkpoint_root = f"{data_path}{args.checkpoint_dir}/{args.exp_name}/"
final_model_path = join(checkpoint_root, 'final-model')

if(args.checkpoint == -1):
    # 2. Otherwise, parse for the checkpoint-* with the highest step
    latest_ckpt = find_latest_checkpoint(checkpoint_root)
    if latest_ckpt is None:
        print("No final-model or checkpoint-* found. Quitting.")
        quit()
    else:
        print(f"No final-model checkpoint found. Using {latest_ckpt}")
        model_path = latest_ckpt
else:
    if args.checkpoint > 0:
        model_path = join(checkpoint_root, f'checkpoint-{args.checkpoint}')
        if not os.path.isdir(model_path):
            print(f"Checkpoint directory {model_path} does not exist. Quitting.")
            quit()
    else:
        # no checkpoint found means or sought means there is no model to load.
        print("Invalid checkpoint specified for loading pretrained model. Quitting.")
        quit()

print(f"Loading pretrained model from {model_path}")

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

net = TabFormerBertModel(custom_special_tokens,
                          vocab=vocab,
                          field_ce=args.field_ce,
                          flatten=args.flatten,
                          ncols=dataset.ncols,
                          field_hidden_size=args.field_hs,
                          time_pos_type=args.time_pos_type,
                          seq_len = dataset.seq_len,
                          num_labels = 2,
                        output_hidden_states = True)

# 2) Load the saved state_dict (the weights):
print(f"Trying to load state dictionary from {model_path}/pytorch_model.bin")
state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")

# If your training code stored weights directly into `net.model`, do:
net.model.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.model.to(device)
# 4) Put model in eval mode and move to GPU if desired
net.model.eval()
net.model.cuda()  # if you have CUDA



# get the CLS embeddings:
# Count total parameters
total_params = sum(p.numel() for p in net.model.parameters())

# Count trainable parameters only
trainable_params = sum(p.numel() for p in net.model.parameters() if p.requires_grad)

log.info(f"Total parameters: {total_params}")
log.info(f"Trainable parameters: {trainable_params}")
collator_cls = "TransDataCollatorForExtraction"


log.info(f"collator class: {collator_cls}")
data_collator = eval(collator_cls)(
    tokenizer=net.tokenizer,
    mlm=False
)

# Dataloader
loader = DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=data_collator)

cls_embeddings = []
row_identifiers = []

with torch.no_grad():
    for idx, batch in enumerate(loader):
        batch.pop('labels', None)
        batch.pop('masked_lm_labels', None)

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = net.model(**batch)

        cls_embed = outputs['logits'][:, 0, :]
        cls_embeddings.append(cls_embed.cpu().numpy())

        # Correctly retrieve the IDs
        ids_batch = dataset.data_sids[idx * args.eval_batch_size : idx * args.eval_batch_size + len(cls_embed)]
        row_ids = [rid.split('_')[-1] for rid in ids_batch]

        # Ensure you're extending a list (not numpy array!)
        row_identifiers.extend(row_ids)

# Finally, convert lists to NumPy arrays
cls_embeddings = np.vstack(cls_embeddings)
row_identifiers = np.array(row_identifiers)
np.save(f"{model_path}/cls_embeddings.npy", cls_embeddings)
np.save(f"{model_path}/cls_embeddings_row_ids.npy", row_identifiers)
print(f"Extracted CLS embeddings shape: {cls_embeddings.shape}")
