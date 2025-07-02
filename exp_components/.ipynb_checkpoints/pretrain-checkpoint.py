import os
import sys
sys.path.append(os.path.abspath('../'))
from os import makedirs
from os.path import join, basename
import numpy as np
import torch
import random
import json

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from dataset.dataset import Dataset
from dataset.dataset_time_static import DatasetWithTimePosAndStaticSplit
from models.modules import TabFormerBertLM, TabFormerBertForClassification, TabFormerBertModel, TabStaticFormerBert, TabStaticFormerBertLM, TabStaticFormerBertClassification
from misc.utils import ordered_split_dataset, compute_cls_metrics, random_split_dataset
from dataset.datacollator import *


def pretrain(args, data_path, feature_extension, log):
    
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

    if args.static_features:
        dataset_class = 'DatasetWithTimePosAndStaticSplit'
    else:
        dataset_class = 'Dataset'
    
    dataset = eval(dataset_class)(cls_task=args.cls_task or args.mlm,
                       seq_len=args.seq_len,
                       root=data_path,
                       fname=train_fname,
                       user_level_cached=True, # We should only be running pretrain after we've created the user-level data in process()
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
    log.info(f'vocab size: {len(vocab)}')
    log.info(f'dataset size: {len(dataset)}')
    custom_special_tokens = vocab.get_special_tokens()

    
    
    valtrainN = len(dataset)
    trainN = int(0.84 * valtrainN)
    valN = valtrainN - trainN
    lengths = [trainN, valN]
    train_dataset, eval_dataset = random_split_dataset(dataset, lengths)

    test_dataset = eval(dataset_class)(cls_task=args.cls_task or args.mlm,
                           seq_len=args.seq_len,
                           root=data_path,
                           fname=test_fname,
                           user_level_cached=True,
                           vocab_cached=True,
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
                           resample_seed=args.resample_seed)
    log.info(f"test dataset size: {len(test_dataset)}")
    trainN = len(train_dataset)
    valN = len(eval_dataset)
    testN = len(test_dataset)
    totalN = trainN + valN + testN

    log.info(
        f"# Using external test dataset, lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    num_labels = 2
    
    if args.static_features:
        model_class = 'TabStaticFormerBertLM'
        tab_net = eval(model_class)(custom_special_tokens,
                              vocab=vocab,
                              field_ce=args.field_ce,
                              flatten=args.flatten,
                              ncols=dataset.ncols,
                              field_hidden_size=args.field_hs,
                              static_ncols=dataset.static_ncols,
                              time_pos_type=args.time_pos_type,
                              num_attention_heads = args.num_attention_heads,
                              attn_implementation = "eager"
                              )
    else:
        model_class = 'TabFormerBertLM'
        tab_net = eval(model_class)(custom_special_tokens,
                              vocab=vocab,
                              field_ce=args.field_ce,
                              flatten=args.flatten,
                              ncols=dataset.ncols,
                              field_hidden_size=args.field_hs,
                              time_pos_type=args.time_pos_type,
                              num_attention_heads = args.num_attention_heads,
                              attn_implementation = "eager"
                              )

    
    log.info(f"model initiated: {tab_net.model.__class__}")
    tab_net.model.eval()  # Set model to evaluation mode

    # Count total parameters
    total_params = sum(p.numel() for p in tab_net.model.parameters())
    
    # Count trainable parameters only
    trainable_params = sum(p.numel() for p in tab_net.model.parameters() if p.requires_grad)
    
    log.info(f"Total parameters: {total_params}")
    log.info(f"Trainable parameters: {trainable_params}")
    if args.static_features:
        collator_cls = "TransWithStaticAndTimePosDataCollatorForLanguageModeling"
    else:
        collator_cls = "TransDataCollatorForLanguageModeling"

    data_collator = eval(collator_cls)(
        tokenizer=tab_net.tokenizer, mlm=args.mlm, mlm_probability=args.mlm_prob
    )
    if torch.cuda.device_count() > 1:
        per_device_train_batch_size = args.train_batch_size // torch.cuda.device_count()
        per_device_eval_batch_size = args.eval_batch_size // torch.cuda.device_count()
    else:
        per_device_train_batch_size = args.train_batch_size
        per_device_eval_batch_size = args.eval_batch_size

    if args.cls_task:
        label_names = ["labels"]
    else:
         label_names = ["masked_lm_labels"]
        
    if args.cls_task:
        metric_for_best_model = 'eval_auc_score'
    else:
        metric_for_best_model = 'eval_loss'
        
    training_args = TrainingArguments(
        output_dir=f"{data_path}/{args.checkpoint_dir}/",  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        logging_dir=f"{data_path}/{args.logging_dir}/{args.exp_name}",  # directory for storing logs
        save_steps=args.save_steps,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy="steps",
        prediction_loss_only=False if args.cls_task else True,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        eval_steps=args.eval_steps,
        label_names=label_names,
        save_safetensors=False, 
        #use_cpu=True, # Remove after debugging
        report_to = "none", # Replaces the WANDB_Reporting environment variable
    )

    if args.freeze:
        for name, param in tab_net.model.named_parameters():
            if name.startswith('tb_model.classifier'):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        trainer = Trainer(
            model=tab_net.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    if args.checkpoint:
        model_path = join(f"{data_path}/{args.checkpoint_dir}/", f'checkpoint-{args.checkpoint}')
        trainer.train(model_path)
    else:
        trainer.train(False)

    test_results = trainer.predict(test_dataset=test_dataset)

    args.main_file = basename(__file__)
    performance_dict = vars(args)

    print_str = 'Test Summary: '
    for key, value in test_results.metrics.items():
        performance_dict['test_' + key] = value
        print_str += '{}: {:8f} | '.format(key, value)
    log.info(print_str)

    for key, value in performance_dict.items():
        if type(value) is np.ndarray:
            performance_dict[key] = value.tolist()

    with open(f"{args.exp_name}_{args.record_file}", 'a+') as outfile:
        outfile.write(json.dumps(performance_dict) + '\n')

    final_model_path = join(f"{data_path}/{args.checkpoint_dir}/", 'final-model')
    trainer.save_model(final_model_path)
    final_prediction_path = join(f"{data_path}/{args.checkpoint_dir}/", 'prediction_results')
    np.savez_compressed(final_prediction_path,
                        predictions=test_results.predictions, label_ids=test_results.label_ids)