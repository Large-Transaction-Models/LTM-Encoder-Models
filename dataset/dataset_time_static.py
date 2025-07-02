from os import path
import pandas as pd
import tqdm
import pickle
import logging
import torch

from dataset.vocab import merge_vocab
from misc.utils import divide_chunks
from dataset.basic import BasicDataset

logger = logging.getLogger(__name__)
log = logger


class DatasetWithTimePosAndStaticSplit(BasicDataset):
    def __init__(self,
                 cls_task,
                 user_ids=None,
                 seq_len=10,
                 root="",
                 fname="",
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
                         seq_len,
                         root,
                         fname,
                         user_level_cached,
                         vocab_cached,
                         external_vocab_path,
                         preload_vocab_dir,
                         save_vocab_dir,
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
            return_dynamic_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_dynamic_data = torch.tensor(
                self.data[index], dtype=torch.long).reshape(self.seq_len, -1)
            
        return_static_data = torch.tensor(self.static_data[index], dtype=torch.long)
        return_time = torch.tensor(self.times[index], dtype=torch.long)
        return_pos_ids = torch.tensor(self.pos_ids[index], dtype=torch.long)
        return_type_ids = torch.cat((torch.zeros(1, dtype=torch.long), torch.ones(self.seq_len, dtype=torch.long)),
                                    dim=0)
        return_data = (return_dynamic_data, return_static_data, return_time, return_pos_ids, return_type_ids)
        
        if self.return_labels:
            # only consider sequence_label and last_label
            return_data_label = torch.tensor(
                self.labels[index], dtype=torch.long).reshape(self.seq_len, -1)
            if self.label_category == "sequence_label":
                return_data = return_data + (return_data_label,)
            else:
                return_data = return_data + (return_data_label[-1, :],)

        return return_data

    def get_final_vocab(self):
        # Don't need to include the time_feature_vocab if we're using time_aware_sin_cos_position
        self.vocab = merge_vocab(self.dynamic_vocab, self.static_vocab)

    def user_level_split_data(self):
        fname = path.join(
            self.root, f"{self.fname}.user.pkl")
        trans_data, trans_labels, trans_time = [], [], []
        if self.get_rids:
            trans_rids = []
        
        if self.user_level_cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            trans_time = cached_data["time"]
            columns_names = cached_data["columns"]
            static_columns = cached_data["static_columns"]
            self.static_ncols = len(static_columns) + 1
            bks_exist = False
            if 'RIDs' in cached_data.keys():
                trans_rids = cached_data['RIDs']
                bks_exist = True

             # Reorder columns to match vocab
            ignore_columns = ["rowNumber", "user", "timestamp", "id"]
            vocab_fields = self.vocab.get_field_keys(remove_target=True, ignore_special=True)

            # Ensure columns_names matches vocab fields and ignore_columns
            combined_columns = set(columns_names) | set(static_columns)
            if not combined_columns.issuperset(set(vocab_fields)):
                missing = set(vocab_fields) - combined_columns
                log.warning(f"❌ Missing columns from cached data: {missing}")
                log.info(f"✅ Available columns in cached data: {list(combined_columns)}")
                log.info(f"🎯 Expected vocab fields: {vocab_fields}")
                raise ValueError("Mismatch between vocab fields and columns in cached data.")


            # Reorder columns
            reordered_columns = ignore_columns + [col for col in vocab_fields if col in columns_names]
            columns_names = [i for i in reordered_columns if i not in ignore_columns]

        else:
            columns_names = list(self.trans_table.columns)
            other_columns = ['rowNumber', 'user', 'timestamp', 'id']
            not_use_columns = ['timeFeature']
            # our static columns are any columns that are prefixed with 'static':
            static_columns = [col for col in self.trans_table.columns if col.startswith('static')]
            #*****************************************************
            # Why do we add 1 to the length of the static columns?
            self.static_ncols = len(static_columns) + 1
            #*****************************************************
            bks_exist = pd.Series(
                [i in columns_names for i in other_columns]).all()
            columns_names = [i for i in columns_names if i not in other_columns and i not in static_columns and i not in not_use_columns]
            start_idx_list = self.trans_table.index[
                self.trans_table['user'].ne(self.trans_table['user'].shift())]
            end_idx_list = start_idx_list[1:] - 1
            end_idx_list = end_idx_list.append(self.trans_table.index[-1:])
            
            for ix in tqdm.tqdm(range(len(start_idx_list))):
                start_ix, end_ix = start_idx_list[ix], end_idx_list[ix]
                user_data = self.trans_table.iloc[start_ix:end_ix + 1]
                user_trans_static, user_trans, user_label, user_time = [], [], [], []
                # new assumption: 'rownumber', 'user', 'timestamp', 'id' are the 0th-3rd columns
                # 'Timestamp' is the 3rd column, we will keep it in user_time
                # 'avg_dollar_amt', 'std_dollar_amt', 'top_mcc', 'top_chip' are the -4 - -1th columns
                # 'timeFeature' is the -5th column, we will not use it
                if bks_exist:
                    skip_idx = 3
                    if self.get_rids:
                        user_rids = []
                else:
                    skip_idx = 0
                    
                # get static feature from the first row in the sequence
                start_static = self.trans_table.iloc[start_ix][static_columns].tolist()
                user_trans_static.extend(start_static)
                for idx, row in user_data.iterrows():
                    row = list(row)
                    row_dict = dict(zip(self.trans_table.columns, row))
                    user_trans.extend([row_dict[col] for col in columns_names])  # dynamic
                    user_label.append(row_dict['user'])  # update this to your actual label col
                    user_time.append(row_dict['timestamp'])      # or whatever you're using for time
                    if self.get_rids and bks_exist:
                        user_rids.append(row[0])
                trans_data.append((user_trans_static, user_trans))
                trans_labels.append(user_label)
                trans_time.append(user_time)
                if self.get_rids and bks_exist:
                    trans_rids.append(user_rids)

            with open(fname, 'wb') as cache_file:
                if self.get_rids and bks_exist:
                    pickle.dump({"trans": trans_data, "labels": trans_labels,
                                 "RIDs": trans_rids, "time": trans_time,
                                 "columns": columns_names, "static_columns": static_columns}, cache_file)
                else:
                    pickle.dump({"trans": trans_data, "labels": trans_labels, 
                                 "time": trans_time,
                                 "columns": columns_names, "static_columns": static_columns}, cache_file)

        self.vocab.set_static_field_keys(static_columns)
        self.vocab.set_dynamic_field_keys(columns_names)
        # convert to str
        if self.get_rids and bks_exist:
            return trans_data, trans_labels, trans_rids, trans_time, columns_names, static_columns
        else:
            return trans_data, trans_labels, trans_time, columns_names, static_columns
            
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
                try:
                    # Use the vocabulary to get the token ID, specifying the column context
                    vocab_id = self.vocab.get_id(field, column_names[jdx])
                    vocab_ids.append(vocab_id)
                except Exception as e:
                    print("🛑 Error on field:", field)
                    print("🧩 Column name:", column_names[jdx])
                    print("📦 Valid tokens in this field:", list(self.vocab.token2id.get(column_names[jdx], {}).keys()))
                    raise e
                
            # Optionally add the [SEP] token ID to the end of the tokenized transaction
            # if a classification task is active (e.g., for BERT-like models)
            if self.cls_task:
                vocab_ids.append(sep_id)
    
            # Append the processed transaction to the list of all tokenized transactions
            pan_vocab_ids.append(vocab_ids)
    
        # Return the list of tokenized transactions
        return pan_vocab_ids

    def format_static_trans(self, static_row, static_columns):
        user_static_ids = []
        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)
        for jdx, field in enumerate(static_row):
            static_vocab_id = self.vocab.get_id(field, static_columns[jdx])
            user_static_ids.append(static_vocab_id)

        # TODO : need to handle ncols when sep is not added
        # and self.flatten:  # only add [SEP] for BERT + flatten scenario
        if self.cls_task:
            user_static_ids.append(sep_id)

        return user_static_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        if self.get_rids:
            trans_data, trans_labels, trans_rids, trans_time, columns_names, static_columns = self.user_level_split_data()
        else:
            trans_data, trans_labels, trans_time, columns_names, static_columns = self.user_level_split_data()
        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_static_row, user_row = trans_data[user_idx]
            user_static_ids = self.format_static_trans(user_static_row, static_columns)
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]
            
            user_time = trans_time[user_idx]
            if self.get_rids:
                user_rids = trans_rids[user_idx]

            bos_token = self.vocab.get_id(
                self.vocab.bos_token, special_token=True)  # will be used for GPT2
            eos_token = self.vocab.get_id(
                self.vocab.eos_token, special_token=True)  # will be used for GPT2
            # will be used for padding sequence
            pad_token = self.vocab.get_id(
                self.vocab.pad_token, special_token=True)

            # Padding tokens for first few transaction sequences shorter than self.seq_len
            if self.pad_seq_first:
                for jdx in range(0 - self.seq_len + 1, min(0, len(user_row_ids) - self.seq_len + 1), self.trans_stride):
                    ids_tail = user_row_ids[0:(jdx + self.seq_len)]
                    ncols = len(ids_tail[0])
                    # flattening
                    ids_tail = [idx for ids_lst in ids_tail for idx in ids_lst]
                    ids = [pad_token for _ in range(
                        ncols) for _ in range(jdx, 0)]
                    ids.extend(ids_tail)
                    # for GPT2, need to add [BOS] and [EOS] tokens
                    if not self.cls_task and self.flatten:
                        ids = [bos_token] + user_static_ids + ids + [eos_token]
                    self.data.append(ids)

                    self.static_data.append(user_static_ids)

                    time_tail = user_time[0:(jdx + self.seq_len)]
                    time_tail = [i - time_tail[0] for i in time_tail]
                    time = [0 for _ in range(jdx, 0)]
                    time.extend(time_tail)
                    # add a time position for static features
                    time = [0] + time
                    self.times.append(time)

                    tail_len = len(time_tail)
                    pos_tail = list(range(0, tail_len))
                    # add a position for static features
                    head_len = self.seq_len - tail_len + 1
                    pos_head = [0 for _ in range(head_len)]
                    pos_ids = pos_head + pos_tail
                    self.pos_ids.append(pos_ids)

                    ids_tail = user_labels[0:(jdx + self.seq_len)]
                    ids = [0 for _ in range(jdx, 0)]
                    ids.extend(ids_tail)
                    self.labels.append(ids)
                    self.data_seq_last_labels.append(ids[-1])

                    if self.get_rids:
                        rids_tail = user_rids[0:(jdx + self.seq_len)]
                        rids = [-1 for _ in range(jdx, 0)]
                        rids.extend(rids_tail)
                        self.data_sids.append(
                            '_'.join([str(int(_)) for _ in rids]))
                        self.data_seq_last_rids.append(rids[-1])
            elif not self.pad_seq_first and len(user_row_ids) < self.seq_len:
                pad_len = self.seq_len - len(user_row_ids)
                ids_tail = user_row_ids[0:]
                ncols = len(ids_tail[0])
                # flattening
                ids_tail = [idx for ids_lst in ids_tail for idx in ids_lst]
                ids = [pad_token for _ in range(
                    ncols) for _ in range(pad_len)]
                ids.extend(ids_tail)
                # for GPT2, need to add [BOS] and [EOS] tokens
                if not self.cls_task and self.flatten:
                    ids = [bos_token] + user_static_ids + ids + [eos_token]
                self.data.append(ids)

                self.static_data.append(user_static_ids)

                time_tail = user_time[0:]
                time_tail = [i - time_tail[0] for i in time_tail]
                time = [0 for _ in range(pad_len)]
                time.extend(time_tail)
                # add a time position for static features
                time = [0] + time
                self.times.append(time)

                tail_len = len(time_tail)
                pos_tail = list(range(0, tail_len))
                # add a position for static features
                head_len = self.seq_len - tail_len + 1
                pos_head = [0 for _ in range(head_len)]
                pos_ids = pos_head + pos_tail
                self.pos_ids.append(pos_ids)

                ids_tail = user_labels[0:]
                ids = [0 for _ in range(pad_len)]
                ids.extend(ids_tail)
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])

                if self.get_rids:
                    rids_tail = user_rids[0:]
                    rids = [-1 for _ in range(pad_len)]
                    rids.extend(rids_tail)
                    self.data_sids.append(
                        '_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])

            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                # for GPT2, need to add [BOS] and [EOS] tokens
                if not self.cls_task and self.flatten:
                    ids = [bos_token] + user_static_ids + ids + [eos_token]
                self.data.append(ids)
                self.static_data.append(user_static_ids)
                time = user_time[jdx:(jdx + self.seq_len)]
                time = [_ - time[0] for _ in time]
                time = [0] + time
                self.times.append(time)
                pos_ids = [0] + list(range(0, self.seq_len))
                self.pos_ids.append(pos_ids)
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)
                self.data_seq_last_labels.append(ids[-1])

                if self.get_rids:
                    rids = user_rids[jdx:(jdx + self.seq_len)]
                    self.data_sids.append(
                        '_'.join([str(int(_)) for _ in rids]))
                    self.data_seq_last_rids.append(rids[-1])

        assert len(self.data) == len(self.static_data) == len(self.times) == len(self.labels) \
               == len(self.data_sids) == len(self.data_seq_last_labels) == len(self.data_seq_last_rids), \
            f"len data: {len(self.data)}, len static data: {len(self.static_data)}, len times data: {len(self.times)},"\
            f"len labels: {len(self.labels)}, len data sids: {len(self.data_sids)}," \
            f"len data seq_last_rids: {len(self.data_seq_last_rids)}" \
            f"len data seq_last_labels: {len(self.data_seq_last_labels)}"

        '''
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 2 + \
                     (1 if self.cls_task else 0)
        self.ncols = self.ncols - (self.static_ncols - 1)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")
