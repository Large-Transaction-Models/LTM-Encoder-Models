import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging

from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from torch.utils.data.dataset import Dataset
from dataset.vocab import Vocabulary

logger = logging.getLogger(__name__)
log = logger


class AavePreloadDataset(Dataset):

    def __init__(self,
                 user_ids=None,
                 num_bins=10,
                 cached= False,
                 encoder_cached=False,
                 external_encoder_path="",
                 vocab_cached=False,
                 root="",
                 fname="transactions_train",
                 vocab_dir="",
                 fextension="",
                 nrows=None,
                 adap_thres=10 ** 8,
                 get_rids=True,
                 columns_to_select=None):
        """
        :param user_ids: user ids will be used, None means all are needed
        :param num_bins: number of bins will be used to discretize continuous fields
        :param cached: if cached encoded data will be used, if so, encoded data will be loaded
        :param encoder_cached: if cached encoder will be used, if so, encoder will be loaded from external_encoder_path
        :param external_encoder_path: path to the external encoder file
        :param vocab_cached: if cached preloaded vocab files will be used,
        if so, preloaded vocab files under vocab_dir will be loaded
        :param root: directory stores input data
        :param fname: file name of input data
        :param vocab_dir: directory stores external preloaded vocab files (dynamic, time_feature, static)
        or generated preloaded vocab files will be stored under this directory
        :param fextension: file name extension used by preloaded vocab files and encoded data
        :param nrows: number of rows that will be read from encoded data, None means all are needed
        :param adap_thres: threshold for setting adaptive softmax
        :param get_rids: if row ids will be kept in the dataset
        :param columns_to_select: columns that will be kept in the encoded data, None means all are needed
        """
        self.user_ids = user_ids
        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.encoder_cached = encoder_cached
        self.external_encoder_path = external_encoder_path
        self.vocab_cached = vocab_cached
        self.get_rids = get_rids
        self.columns_to_select = columns_to_select

        self.dynamic_vocab = Vocabulary(adap_thres, target_column_name='')
        self.time_feature_vocab = Vocabulary(adap_thres, target_column_name='')
        self.static_vocab = Vocabulary(adap_thres, target_column_name='')
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.static_data = []
        self.times = []
        self.pos_ids = []
        self.labels = []
        self.data_sids = []
        self.data_seq_last_rids = []
        self.data_seq_last_labels = []

        self.ncols = None
        self.static_ncols = None
        self.num_bins = num_bins
        self.encoder_path = None

        self.encode_data()
        if self.vocab_cached:
            self.load_vocab(f"{root}/{vocab_dir}")
        else:
            self.init_vocab()
            self.save_vocab(f"{root}/{vocab_dir}")

    def init_vocab(self):
        # all column names:
        column_names = list(self.trans_table.columns)
        log.info(f"starting with {len(column_names)} columns")
        # specific columns that we want to drop:
        drop_columns = ['rowNumber', 'user', 'timestamp','id']
        
        # we are using the user-level features as static columns
        static_columns = [col for col in self.trans_table.columns if col.startswith('user') and col != 'user']

        # this is the single column that will be used for the time-aware aspect
        time_feature_columns = ['timeFeature']
        
        # all other columns are to be used as dynamic columns
        dynamic_columns = [col for col in self.trans_table.columns if col not in drop_columns + static_columns + time_feature_columns]
        
        self.dynamic_vocab.set_field_keys(dynamic_columns)
        self.time_feature_vocab.set_field_keys(time_feature_columns)
        self.static_vocab.set_field_keys(static_columns)

        for column in dynamic_columns:
            unique_values = self.trans_table[column].value_counts(
                sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.dynamic_vocab.set_id(val, column)

        for column in time_feature_columns:
            unique_values = self.trans_table[column].value_counts(
                sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.time_feature_vocab.set_id(val, column)

        for column in static_columns:
            unique_values = self.trans_table[column].value_counts(
                sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.static_vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}, "
                 f"total dynamic columns: {dynamic_columns}, "
                 f"total time feature columns: {time_feature_columns},"
                 f"total static columns: {static_columns}")
        log.info(f"total dynamic vocabulary size: {len(self.dynamic_vocab.id2token)}, "
                 f"total time feature vocabulary size: {len(self.time_feature_vocab.id2token)}, "
                 f"total static vocabulary size: {len(self.static_vocab.id2token)}")

        for vocab in [self.dynamic_vocab, self.time_feature_vocab, self.static_vocab]:
            for column in vocab.field_keys:
                vocab_size = len(vocab.token2id[column])
                log.info(f"column : {column}, vocab size : {vocab_size}")

                if vocab_size > vocab.adap_thres:
                    log.info(f"\tsetting {column} for adaptive softmax")
                    vocab.adap_sm_cols.add(column)

    def save_vocab(self, vocab_dir):
        """
        Saves the vocabularies (dynamic, time feature, and static) to specified directory.
        Each vocabulary is saved in two formats: a regular serialized format and a pickled object.
    
        Args:
            vocab_dir (str): Directory where the vocabulary files will be saved.
        """
        # Iterate over the vocabularies and their corresponding prefixes
        for vocab, vocab_prefix in zip([self.dynamic_vocab, self.time_feature_vocab, self.static_vocab],
                                       ['dynamic', 'time_feature', 'static']):
            # Construct the file name for the serialized vocabulary format
            file_name = path.join(vocab_dir, f'{vocab_prefix}_vocab{self.fextension}.nb')
            
            # Log the saving process for the serialized format
            log.info(f"saving vocab at {file_name}")
            
            # Ensure the directory exists
            os.makedirs(vocab_dir, exist_ok=True)
            
            # Save the vocabulary in a serialized format (assumes vocab has a `save_vocab` method)
            vocab.save_vocab(file_name)
            
            # Construct the file name for the pickled object format
            file_name = path.join(vocab_dir, f'{vocab_prefix}_vocab_ob{self.fextension}')
            
            # Save the vocabulary as a pickled object
            with open(file_name, 'wb') as vocab_file:
                pickle.dump(vocab, vocab_file)
            
            # Log the saving process for the pickled format
            log.info(f"saving vocab object at {file_name}")
    
    def load_vocab(self, vocab_dir):
        """
        Loads the vocabularies (dynamic, time feature, and static) from specified directory.
        The vocabularies are loaded from pickled objects.
    
        Args:
            vocab_dir (str): Directory from where the vocabulary files will be loaded.
    
        Raises:
            FileNotFoundError: If any expected vocabulary file is missing.
        """
        # Iterate over the vocabulary prefixes to locate and load the corresponding files
        for vocab_prefix in ['dynamic', 'time_feature', 'static']:
            # Construct the file name for the pickled object format
            file_name = path.join(vocab_dir, f'{vocab_prefix}_vocab_ob{self.fextension}')
            
            # Check if the file exists
            if not path.exists(file_name):
                # Raise an error if the vocabulary file is missing
                raise FileNotFoundError(f"external {vocab_prefix} file is not found")
            else:
                # Load the pickled vocabulary object and assign it to the corresponding attribute
                if vocab_prefix == 'dynamic':
                    self.dynamic_vocab = pickle.load(open(file_name, 'rb'))
                elif vocab_prefix == 'time_feature':
                    self.time_feature_vocab = pickle.load(open(file_name, 'rb'))
                else:
                    self.static_vocab = pickle.load(open(file_name, 'rb'))


    @staticmethod
    def label_fit_transform(column, enc_type="label", unk_value=None):
        column = np.asarray(column).reshape(-1, 1)
        if enc_type == "label":
            mfit = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=unk_value)
        else:
            mfit = RobustScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)


    def _quantization_binning(self, data):
        """
        Computes quantile-based bin edges, bin centers, and bin widths,
        ignoring NaN values.

        Parameters:
        - data (np.ndarray): Input data of shape (num_samples, num_features).

        Returns:
        - bin_edges (np.ndarray): Array of bin edges, shape (num_bins + 1, num_features).
        - bin_centers (np.ndarray): Array of bin centers, shape (num_bins, num_features).
        - bin_widths (np.ndarray): Array of bin widths, shape (num_bins, num_features).
        """
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.apply_along_axis(
            lambda col: np.quantile(col[~np.isnan(col)], qtls),
            axis=0,
            arr=data,
        )  # Shape: (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)  # Shape: (num_bins, num_features)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # Shape: (num_bins, num_features)
        
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        """
        Quantizes the input data based on bin edges, ignoring NaN values.
        NaN values are replaced with 0.

        Parameters:
        - inputs (np.ndarray): Input data of shape (num_samples, num_features).
        - bin_edges (np.ndarray): Bin edges from `_quantization_binning`.

        Returns:
        - quant_inputs (np.ndarray): Quantized data, shape (num_samples, num_features).
        """
        # Initialize output with zeros for NaN handling
        quant_inputs = np.zeros_like(inputs, dtype=int)

        # Mask for non-NaN values
        non_nan_mask = ~np.isnan(inputs)

        # Quantize only the non-NaN values
        quantized_non_nan = np.digitize(inputs[non_nan_mask], bin_edges, right=False)
        quantized_non_nan = np.clip(quantized_non_nan, 1, self.num_bins) - 1

        # Assign quantized values to the appropriate locations
        quant_inputs[non_nan_mask] = quantized_non_nan
        
        # Set NaN values to 0
        quant_inputs[~non_nan_mask] = 0

        return quant_inputs

    def prepare_samples(self):
        pass

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        # A column is considered mixed-type if it has more than one unique data type
        mixed_type_columns = [
            col for col in data.columns if data[col].apply(type).nunique() > 1
        ]
        
        # Convert mixed-type columns to categorical
        for col in mixed_type_columns:
            data[col] = data[col].astype('category')
            
        if self.user_ids:
            log.info(f'Filtering data by user ids list: {self.user_ids}...')
            self.user_ids = map(int, self.user_ids)
            data = data[data['user'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")
    
        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(
                dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return
        elif self.cached and not path.isfile(path.join(dirname, fname)):
            raise FileNotFoundError("cached encoded data is not found")
    
        if self.encoder_cached and path.isfile(self.external_encoder_path):
            log.info(
                f"cached encoder is read from {self.external_encoder_path}")
            self.encoder_fit = pickle.load(
                open(self.external_encoder_path, "rb"))
        elif self.encoder_cached and not path.isfile(self.external_encoder_path):
            raise FileNotFoundError("cached encoder is not found")
    
        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")


        
        log.info("nan resolution.")
        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['category']).columns
        obj_cols = data.select_dtypes(include = ['object']).columns
        time_feature_cols = ['timeFeature']
        # Fill NA values
        # data[numeric_cols] = data[numeric_cols].fillna(0)
        
        # Add 'None' category and fill NA values
        for col in categorical_cols:
            data[col] = data[col].cat.add_categories("None")  # Add the new category
            data[col] = data[col].fillna("None")             # Fill NA with the new category
            data[col] = data[col].astype('str')

        for col in obj_cols:
            data[col] = data[col].fillna("None")
            data[col] = data[col].astype('str')

        bool_cols = data.select_dtypes(include=[bool]).columns
        for col in bool_cols:
            data[col] = data[col].astype('str')
            data[col] = data[col].fillna("None")
        
        log.info("timestamp fit transform")
        #data['timeFeature'] = data['total_minutes']

    
        
    
        log.info("label-fit-transform.")
        unk_value = -1
        for col_name in tqdm.tqdm(list(categorical_cols) + list(obj_cols) + list(bool_cols)):
            col_data = data[col_name]
            encoder_name = col_name
            if self.encoder_fit.get(encoder_name) is not None:
                col_fit = self.encoder_fit.get(encoder_name)
                if isinstance(col_fit, OrdinalEncoder):
                    col_data = np.asarray(col_data).reshape(-1, 1)
                col_data = col_fit.transform(col_data)
            else:
                col_fit, col_data = self.label_fit_transform(
                    col_data, unk_value=unk_value)
                self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data
    
        log.info("amount quant transform")
        for col_name in tqdm.tqdm(list(numeric_cols)):
            coldata = np.array(data[col_name])
            if self.encoder_fit.get(col_name) is not None:
                bin_edges, bin_centers, bin_widths = self.encoder_fit.get(col_name)
                data[col_name] = self._quantize(coldata, bin_edges)
            else:
                bin_edges, bin_centers, bin_widths = self._quantization_binning(
                    coldata)
                data[col_name] = self._quantize(coldata, bin_edges)
                self.encoder_fit[col_name] = [bin_edges, bin_centers, bin_widths]
            
        if self.columns_to_select:
            columns_to_select = self.columns_to_select
        else:
            # Select all relevant columns including time features
            columns_to_select = list(set(list(numeric_cols) + list(categorical_cols) + list(obj_cols) + list(bool_cols) + list(time_feature_cols)))

        if not self.get_rids:
            columns_to_select = [col for col in columns_to_select if col != 'rowNumber'] # Drop rowNumber if not needed

        # List of columns to move to the front
        first_cols = ['rowNumber', 'user', 'timestamp', 'id']
        # we are using the user-level features as static columns
        static_columns = [col for col in columns_to_select if col.startswith('user') and col != 'user']

        # this is the single column that will be used for the time-aware aspect
        time_feature_columns = ['timeFeature']
        
        # all other columns are to be used as dynamic columns
        dynamic_columns = [col for col in columns_to_select if col not in first_cols + static_columns + time_feature_columns]
        
        rearranged_columns = first_cols + dynamic_columns + time_feature_columns + static_columns
        
        self.trans_table = data[rearranged_columns]
    
        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))
    
        self.encoder_path = path.join(
            dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {self.encoder_path}")
        pickle.dump(self.encoder_fit, open(self.encoder_path, "wb"))
