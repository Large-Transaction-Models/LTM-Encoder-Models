import argparse


def parse_arguments():
    # Adapting the argument parsing so it works more easily with scripts as opposed to notebooks:
    parser = argparse.ArgumentParser()

    # Meta
    parser.add_argument("--jid", type=int,
                        default=1,
                        help="job id: 1[default] used for job queue")
    parser.add_argument('--exp_name', type=str, 
                        default='default',
                       help="experiment name for keeping records")
    parser.add_argument('--record_file', type=str,
                        default='record.txt',
                        help="record file for end of experiment")
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument("--checkpoint_dir", type=str, 
                        default='checkpoints', help="path to model dump")
    parser.add_argument("--logging_dir", type=str,
                        default='logs', help="path to save logs")
    parser.add_argument("--log_file_name", type=str,
                        default='output.log', help="log file name")
    # data locating:
    parser.add_argument('--dataset', type=str, default='cosmetics', 
                                    choices=['Aave_V2_Mainnet', 'Aave_V2_Polygon', 'Aave_V2_Avalanche',
                                             'Aave_V3_Ethereum', 'Aave_V3_Polygon', 'Aave_V3_Avalanche', 
                                             'Aave_V3_Optimism', 'Aave_V3_Harmony', 'Aave_V3_Fantom', 'Aave_V3_Arbitrum',
                                             'AML_LI_Small', 'AML_LI_Medium', 'AML_LI_Large',
                                             'AML_HI_Small', 'AML_HI_Medium', 'AML_HI_Large',
                                             'electronics', 'cosmetics',
                                             'Uni_V2', 'Uni_V3'])
    parser.add_argument('--include_user_features', action='store_true',
                        help='whether to include user-level features to be used as static features.')
    parser.add_argument('--include_time_features', action='store_true',
                        help='whether to include features derived from the timestamp, such as day of week, day of month, etc.')
    parser.add_argument('--include_market_features', action='store_true',
                        help='whether to include features derived from the overall market, i.e. historical features based on the whole transaction dataset')
    parser.add_argument('--include_exo_features', action='store_true',
                        help='whether to include any exogenous features. The features to include will come from the exo_prefix and exo_suffix args')
    parser.add_argument('--exo_prefixes', type=str, default='None')
    parser.add_argument('--exo_suffixes', type=str, default='None')
    parser.add_argument('--data_format', type=str, default='csv', 
                                    choices=['rds', 'csv', 'h5'])
    parser.add_argument('--data_loading', type=str, default='default', 
                                    choices=['default', 'lazy'])

    
    # Data pre-processing:
    parser.add_argument("--check_preprocess_cached", action="store_true",
                        help="based on the dataset and choice of features, check if this data has already been preprocessed and use the preprocessed version if so")
    parser.add_argument('--train_test_thres', type=float, default=0.6,
                        help="percent of data to use in training data")
    parser.add_argument('--seq_len', type=int, default=10,
                        help="how many transactions to use in one transaction sequence")

    # Data preloading:
    parser.add_argument("--check_preload_cached", action="store_true",
                        help="based on the dataset and choice of features, check if this data has already been preloaded and use the preloaded version if so.")
    parser.add_argument("--num_bins", type=int,
                        default=10, help="the number of bins for quantizing data")
    parser.add_argument("--get_rids", action='store_true',
                        help="if transaction rid will be stored")
    parser.add_argument("--adap_thres", type=float,
                        default=10**8, help="adaptive threshold for vocab size")
    parser.add_argument("--vocab_dir", type=str,
                        default="/vocab",
                        help="path to load vocab file")

    # Data processing:
    parser.add_argument("--check_process_cached", action="store_true",
                        help="based on the dataset and choice of features, check if this data has already been processed and use the processed version if so.")
    parser.add_argument("--pad_seq_first", action='store_true',
                        help="if each user first few transactions will be padded to build sequences (default not pad)")
    parser.add_argument("--long_and_sort", action='store_true',
                        help="if transaction data is very and sorted by id, if so, will not use .loc function to process data")
    parser.add_argument("--flatten", action='store_true',
                        help="enable flattened input, no hierarchical")
    

    # Task selection:
    parser.add_argument("--do_pretrain", action='store_true',
                        help="enable if you want to pretrain the model from scratch. The reason you might not include this is if you want to load a pretrained model instead in order to generate embeddings")
    parser.add_argument("--mlm", action='store_true',
                        help="masked lm loss; pass it for BERT")
    parser.add_argument("--cls_task", action='store_true',
                        help="classification loss; pass it for BERT")
    parser.add_argument("--reg_task", action='store_true',
                        help="regression loss; pass it for BERT")
    parser.add_argument("--export_task", action='store_true',
                        help='if export embeddings or not')

    # MLM Hyperparams:
    parser.add_argument("--mlm_prob", type=float,
                        default=0.15,
                        help="mask mlm_probability")
    parser.add_argument("--checkpoint", type=int,
                        default=0,
                        help='set to continue training from checkpoint')
    parser.add_argument("--num_train_epochs", type=int,
                        default=20,
                        help="number of training epochs")
    parser.add_argument("--save_steps", type=int,
                        default=10000,
                        help="set checkpointing")
    parser.add_argument("--eval_steps", type=int,
                        default=10000,
                        help="Number of update steps between two evaluations")
    parser.add_argument("--train_batch_size", type=int,
                        default=8,
                        help="training batch size")
    parser.add_argument("--eval_batch_size", type=int,
                        default=8,
                        help="eval batch size")
    parser.add_argument("--stride", type=int,
                        default=5,
                        help="stride for transaction sliding window")
    parser.add_argument("--field_hs", type=int,
                        default=32,
                        help="field hidden size for transaction transformer")
    parser.add_argument("--hidden_size", type=int,
                        default=768,
                        help="hidden size for transaction transformer")
    parser.add_argument("--nrows", type=int,
                        default=None,
                        help="no of transactions to use")
    parser.add_argument('--freeze', action='store_true',
                        help='If set, freezes all layer parameters except for the output layer')
    parser.add_argument("--label_category", type=str,
                        default="last_label", choices=['last_label', 'window_label', 'sequence_label'],
                        help='type of target label used for the classifier')

    # Resampling params:
    parser.add_argument("--resample_method", default=None,
                        help='a method for data resampling. If None, then not resampling, if "upsample",'
                             'then the class with less samples will be upsampled, if "downsample", '
                             'then the class with more samples will be downsampled, if a int is given, '
                             'then each class will be sampled by the given number')
    parser.add_argument("--resample_ratio", type=float, default=10,
                        help='ratio for resample data, resample_ratio = # of negative samples / # of positive samples')
    parser.add_argument("--resample_seed", type=int, default=100,
                        help='random seed for data resample')

    # Other hyper-params:
    parser.add_argument("--time_pos_type", type=str,
                        default="time_aware_sin_cos_position",
                        choices=['time_aware_sin_cos_position', 'sin_cos_position', 'regular_position'],
                        help='position embedding type')
    parser.add_argument("--field_ce", action='store_true',
                        help="enable field wise CE")
    parser.add_argument("--static_features", action='store_true',
                        help="enable static field transformer")

    # Export parameters:
    parser.add_argument("--nbatches", type=int,
                        default=10,
                        help="no of batches to use in export task")
    parser.add_argument("--export_last_only", action='store_true',
                        help='when export_task is True, if only export last embedding of each sequence or not')
    parser.add_argument("--export_cls_embeddings", action='store_true',
                        help='when export_task is True, if this is true, extracts the cls embeddings for each sequences only.')

    return parser.parse_args()


