# Import necessary libraries
import pyreadr
import os
from os.path import join, basename
import sys
sys.path.append(os.path.abspath('../'))
import pandas as pd
import numpy as np
import math

import logging

from pathlib import Path
from arguments import parse_arguments
from utils import get_data_path
from utils import setup_logging
from preprocess import preprocess
from preload import preload
from process import process
from pretrain import pretrain

args = parse_arguments()
np.random.seed(args.seed) 
args.checkpoint_dir += f"/{args.exp_name}"
if(args.nrows == -1):
    args.nrows = None

data_path, feature_extension = get_data_path(args)


log = setup_logging(log_dir=f"{data_path}preprocessed{feature_extension}/preloaded_{args.num_bins}_{args.nrows}_seqLen{args.seq_len}/{args.logging_dir}/{args.exp_name}", log_file_name = args.log_file_name) 

log.info("Running preprocess")
preprocess(args, data_path, feature_extension, log)
log.info("Preprocess complete")

data_path += f"preprocessed{feature_extension}/"

log.info("Running preload")
preload(args, data_path, feature_extension, log)
log.info("preload complete")

data_path += f"preloaded_{args.num_bins}_{args.nrows}/"

log.info("Running process")
process(args, data_path, feature_extension, log)
log.info("process complete")


if args.do_pretrain:
    log.info("Pretraining...")
    pretrain(args, data_path, feature_extension, log)
    log.info("Completed pretraining.")
else:
    log.info("skipping pretraining")

if args.export_task:
    log.info("Loading pretrained model for exporting embeddings...")
    model = load_pretrained_model(args, data_path, feature_extension, log)
    log.info("Model loaded. Generating embeddings...")



