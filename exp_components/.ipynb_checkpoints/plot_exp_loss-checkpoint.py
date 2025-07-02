#!/usr/bin/env python3

import json
import argparse
import matplotlib.pyplot as plt
# Import necessary libraries
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
from utils import get_data_path, find_latest_checkpoint

    



def main():

    args = parse_arguments()
    

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
            latest_ckpt = join(checkpoint_root, f'checkpoint-{args.checkpoint}')
            if not os.path.isdir(latest_ckpt):
                print(f"Checkpoint directory {model_path} does not exist. Quitting.")
                quit()
            model_path = latest_ckpt
        else:
            # checkpoint == 0 or some other special case => train from scratch
            quit()
        
    
    with open(join(model_path, "trainer_state.json"), "r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    # The 'log_history' key holds the individual training/eval logs
    logs = trainer_state.get("log_history", [])
    if not logs:
        raise ValueError("No 'log_history' found in trainer_state.json.")

    steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    # Each entry in log_history may contain 'loss', 'eval_loss', 'step', 'epoch', etc.
    for entry in logs:
        # Training loss
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            train_losses.append(entry["loss"])
        # Evaluation loss
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    # --- Plotting ---
    plt.figure()  # Ensure we use a single distinct figure
    # Plot training loss if we have it
    if train_losses:
        plt.plot(steps, train_losses, label="Training Loss")
    # Plot evaluation loss if we have it
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Evaluation Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"{args.dataset} {args.exp_name} {args.num_bins} bins")
    plt.legend()
    plot_file = latest_ckpt + "/train_loss_plot.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to: {plot_file}")

if __name__ == "__main__":
    main()
