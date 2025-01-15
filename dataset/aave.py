from os import path
import pandas as pd
import tqdm
import pickle
import logging
import torch

from dataset.vocab import merge_vocab
from misc.utils import divide_chunks
from dataset.aave_basic import AaveBasicDataset

logger = logging.getLogger(__name__)
log = logger


class AaveDataset(AaveBasicDataset):
    def __init__(self,
                 cls_task,
                 user_ids=None,
                 seq_len=10,
                 root="",
                 fname="transactions_train",
                 user_level_cached=False,
                 vocab_cached=False,
                 external_vocab_path="",
                 preload_vocab_dir="",
                 save_vocab_dir="",
                 preload_fextension="",
                 fextension="",
                 nrows=None,
                 flatten=False,
                 stride=5,
                 return_labels=True,
                 label_category='last_label',
                 pad_seq_first=False,
                 get_rids=True,
                 long_and_sort=True,
                 resample_method=None,
                 resample_ratio=2,
                 resample_seed=100
                 ):

        super().__init__(cls_task,
                         user_ids,
                         seq_len,
                         root,
                         fname,
                         user_level_cached,
                         vocab_cached,
                         external_vocab_path,
                         preload_vocab_dir,
                         save_vocab_dir,
                         preload_fextension,
                         fextension,
                         nrows,
                         flatten,
                         stride,
                         return_labels,
                         label_category,
                         pad_seq_first,
                         get_rids,
                         long_and_sort,
                         resample_method,
                         resample_ratio,
                         resample_seed
                         )

    def __getitem__(self, index):
        if self.flatten:
            return_data_fea = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data_fea = torch.tensor(
                self.data[index], dtype=torch.long).reshape(self.seq_len, -1)
        return_data = return_data_fea
        if self.return_labels:
            # only consider sequence_label and last_label
            return_data_label = torch.tensor(
                self.labels[index], dtype=torch.long).reshape(self.seq_len, -1)
            if self.label_category == "sequence_label":
                return_data = (return_data, return_data_label)
            else:
                return_data = (return_data, return_data_label[-1, :])

        return return_data

    def get_final_vocab(self):
        vocab = merge_vocab(self.dynamic_vocab, self.time_feature_vocab)
        self.vocab = merge_vocab(vocab, self.static_vocab)

    def user_level_split_data(self):
        fname = path.join(
            self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels = [], []
        if self.get_rids:
            trans_rids = []

        if self.user_level_cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]
            bks_exist = False
            if 'RIDs' in cached_data.keys():
                trans_rids = cached_data['RIDs']
                bks_exist = True

        
            # Reorder columns to match vocab
            ignore_columns = ["rowNumber", "user", "timestamp", "id"]
            vocab_fields = self.vocab.get_field_keys(remove_target=True, ignore_special=True)

            # Ensure columns_names matches vocab fields and ignore_columns
            if not set(columns_names).issuperset(set(vocab_fields)):
                raise ValueError("Mismatch between vocab fields and columns in cached data.")

            # Reorder columns
            reordered_columns = ignore_columns + [col for col in vocab_fields if col in columns_names]
            columns_names = [i for i in reordered_columns if i not in ignore_columns]

        else:
            columns_names = list(self.trans_table.columns)
            other_columns = ['rowNumber', 'user', 'timestamp', 'id']
            bks_exist = pd.Series(
                [i in columns_names for i in other_columns]).all()
            columns_names = [i for i in columns_names if i not in other_columns]
            start_idx_list = self.trans_table.index[
                self.trans_table['user'].ne(self.trans_table['user'].shift())]
            end_idx_list = start_idx_list[1:] - 1
            end_idx_list = end_idx_list.append(self.trans_table.index[-1:])
            for ix in tqdm.tqdm(range(len(start_idx_list))):
                start_ix, end_ix = start_idx_list[ix], end_idx_list[ix]
                user_data = self.trans_table.iloc[start_ix:end_ix + 1]
                user_trans, user_label = [], []
             
                if bks_exist:
                    skip_idx = len(other_columns)
                    if self.get_rids:
                        user_rids = []
                else:
                    skip_idx = 0

                for idx, row in user_data.iterrows():
                    row = list(row)
                    user_trans.extend(row[skip_idx:])
                    user_label.append(row[0])
                    if self.get_rids and bks_exist:
                        user_rids.append(row[0])
                trans_data.append(user_trans)
                trans_labels.append(user_label)
                if self.get_rids and bks_exist:
                    trans_rids.append(user_rids)

            with open(fname, 'wb') as cache_file:
                if self.get_rids and bks_exist:
                    pickle.dump({"trans": trans_data, "labels": trans_labels,
                                 "RIDs": trans_rids, "columns": columns_names}, cache_file)
                else:
                    pickle.dump({"trans": trans_data, "labels": trans_labels,
                                 "columns": columns_names}, cache_file)

        # convert to str
        if self.get_rids and bks_exist:
            return trans_data, trans_labels, trans_rids, columns_names
        else:
            return trans_data, trans_labels, columns_names

    def format_trans(self, trans_lst, column_names):
        """
        Formats a list of transactions into a list of tokenized representations 
        suitable for further processing, like feeding into a model.
    
        Args:
            trans_lst (list): A flattened list of transaction fields.
            column_names (list): A list of column names corresponding to the transaction fields.
    
        Returns:
            list: A list of lists, where each inner list contains tokenized representations 
                  of a single transaction, potentially ending with a [SEP] token.
        """

        # Break the flat list of transactions into chunks of size equal to the number of columns
        trans_lst = list(divide_chunks(trans_lst, len(column_names)))
        
        # Initialize a list to store the tokenized representations of all transactions
        pan_vocab_ids = []
    
        # Retrieve the ID for the [SEP] token from the vocabulary, marking it as a special token
        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)
    
        # Iterate over each transaction in the chunked transaction list
        for trans in trans_lst:
            vocab_ids = []  # Temporary list to store token IDs for the current transaction
            # Map each field in the transaction to its corresponding vocabulary ID
            for jdx, field in enumerate(trans):
                # Use the vocabulary to get the token ID, specifying the column context
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)
                
            # Optionally add the [SEP] token ID to the end of the tokenized transaction
            # if a classification task is active (e.g., for BERT-like models)
            if self.cls_task:
                vocab_ids.append(sep_id)
    
            # Append the processed transaction to the list of all tokenized transactions
            pan_vocab_ids.append(vocab_ids)
    
        # Return the list of tokenized transactions
        return pan_vocab_ids

    def prepare_samples(self):
        """
        Prepares user-level data samples for training or evaluation by tokenizing transactions, 
        adding padding, and appending special tokens like [BOS] and [EOS] where necessary.
    
        This function handles sequence formatting, padding, and labeling for tasks such as 
        classification (CLS tasks) or sequence modeling (e.g., GPT-like models).
    
        Raises:
            AssertionError: If the lengths of data, labels, and related lists are inconsistent.
        """
        # Log the start of user-level data preparation
        log.info("preparing user level data...")
    
        # Split the data into transactions, labels, and (optionally) unique row IDs
        if self.get_rids:
            trans_data, trans_labels, trans_rids, columns_names = self.user_level_split_data()
        else:
            trans_data, trans_labels, columns_names = self.user_level_split_data()
    
        log.info("creating transaction samples with vocab")
    
        # Iterate through each user's data
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            # Extract user-level transaction data and convert to token IDs
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)
    
            # Extract labels and (optionally) row IDs for the user
            user_labels = trans_labels[user_idx]
            if self.get_rids:
                user_rids = trans_rids[user_idx]
    
            # Get special tokens from the vocabulary
            bos_token = self.vocab.get_id(self.vocab.bos_token, special_token=True)  # Begin-of-sequence token
            eos_token = self.vocab.get_id(self.vocab.eos_token, special_token=True)  # End-of-sequence token
            pad_token = self.vocab.get_id(self.vocab.pad_token, special_token=True)  # Padding token
    
            # Handle padding for sequences shorter than `self.seq_len`, padding at the beginning
            if self.pad_seq_first:
                for jdx in range(0 - self.seq_len + 1, min(0, len(user_row_ids) - self.seq_len + 1), self.trans_stride):
                    ids_tail = user_row_ids[0:(jdx + self.seq_len)]
                    ncols = len(ids_tail[0])
                    # Flatten the IDs and add padding for sequences shorter than `self.seq_len`
                    ids_tail = [idx for ids_lst in ids_tail for idx in ids_lst]
                    ids = [pad_token for _ in range(ncols) for _ in range(jdx, 0)]
                    ids.extend(ids_tail)
                    # Add [BOS] and [EOS] for sequence models
                    if not self.cls_task and self.flatten:
                        ids = [bos_token] + ids + [eos_token]
                    self.data.append(ids)
    
                    # Handle labels and add padding for sequences shorter than `self.seq_len`
                    ids_tail = user_labels[0:(jdx + self.seq_len)]
                    ids = [0 for _ in range(jdx, 0)]
                    ids.extend(ids_tail)
                    self.labels.append(ids)
                    self.data_seq_last_labels.append(ids[-1])
    
                    # Handle unique row IDs (rids) if `self.get_rids` is True
                    if self.get_rids:
                        rids_tail = user_rids[0:(jdx + self.seq_len)]
                        rids = [-1 for _ in range(jdx, 0)]
                        rids.extend(rids_tail)
                        self.data_sids.append('_'.join([str(int(_)) for _ in rids]))
                        self.data_seq_last_rids.append(rids[-1])
    
            # Handle padding for shorter sequences, padding at the end
            elif not self.pad_seq_first and len(user_row_ids) < self.seq_len:
                pad_len = self.seq_len - len(user_row_ids)
                ids_tail = user_row_ids[0:]
                ncols = len(ids_tail[0])
                # Flatten the IDs and add padding
                ids_tail = [idx for ids_lst in ids_tail for idx in ids_lst]
                ids = [pad_token for _ in range(ncols) for _ in range(pad_len)]
                ids.extend(ids_tail)
                # Add [BOS] and [EOS] for sequence models
                if not self.cls_task and self.flatten:
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)
    
                # Handle labels and padding
                ids_tail = user_labels[0:]
                ids = [0 for _ in range(pad_len)]
                ids.extend(ids_tail)
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])
    
                # Handle row IDs (rids) if `self.get_rids` is True
                if self.get_rids:
                    rids_tail = user_rids[0:]
                    rids = [-1 for _ in range(pad_len)]
                    rids.extend(rids_tail)
                    self.data_sids.append('_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])
    
            # Handle standard stride-based sequence creation
            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # Flatten the sequence
                # Add [BOS] and [EOS] for sequence models
                if not self.cls_task and self.flatten:
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)
    
                # Handle labels for each stride
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])
    
                # Handle row IDs for each stride if `self.get_rids` is True
                if self.get_rids:
                    rids = user_rids[jdx:(jdx + self.seq_len)]
                    self.data_sids.append('_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])
    
        # Assert that all lists have the same length to ensure consistency
        assert len(self.data) == len(self.labels) == len(self.data_sids) \
               == len(self.data_seq_last_labels) == len(self.data_seq_last_rids), \
            f"len data: {len(self.data)}, len labels: {len(self.labels)}, len data sids: {len(self.data_sids)}, " \
            f"len data seq_last_rids: {len(self.data_seq_last_rids)}, len data seq_last_labels: {len(self.data_seq_last_labels)}"
    
        # Calculate the number of columns (ncols) based on the vocabulary and task type
        self.ncols = len(self.vocab.field_keys) - 1 + (1 if self.cls_task else 0)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")

