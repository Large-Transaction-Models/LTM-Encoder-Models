from collections import OrderedDict
import numpy as np
import copy


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        # Initialize the dictionary using the parent class's constructor
        super(AttrDict, self).__init__(*args, **kwargs)
        # Set the object's __dict__ attribute to reference itself
        # This enables attribute-style access to dictionary keys
        self.__dict__ = self

class Vocabulary:
    def __init__(self, adap_thres=10000, target_column_name=""):
        # Define special tokens for various purposes, such as unknown, padding, and control tokens
        self.unk_token = "[UNK]"  # Token for unknown words
        self.sep_token = "[SEP]"  # Token for separating sequences
        self.pad_token = "[PAD]"  # Token for padding sequences to uniform length
        self.cls_token = "[CLS]"  # Token representing an entire sequence
        self.mask_token = "[MASK]"  # Token used for masked positions in sequences
        self.bos_token = "[BOS]"  # Token indicating the beginning of a sequence
        self.eos_token = "[EOS]"  # Token indicating the end of a sequence

        # Threshold for adaptive tokenization, e.g., switching between dynamic and static vocabulary
        self.adap_thres = adap_thres
        # Set of columns for adaptive tokenization, potentially for dynamic feature handling
        self.adap_sm_cols = set()

        # Name of the target column for classification or other token-related tasks
        self.target_column_name = target_column_name
        # Tag for grouping all special tokens under a single "field" in the vocabulary
        self.special_field_tag = "SPECIAL"

        # Collect all special tokens in a list for easier handling
        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]

        # Maps tokens to their corresponding IDs and fields
        # Example: {field_name: {token: id}}
        self.token2id = OrderedDict()
        # Maps global IDs to their corresponding tokens and fields
        # Example: {id: [token, field_name, local_id]}
        self.id2token = OrderedDict()

        # Dictionaries for tracking field keys and dividing them into categories
        self.field_keys = OrderedDict()  # All field keys in the vocabulary
        self.static_field_keys = OrderedDict()  # Field keys for static tokens
        self.dynamic_field_keys = OrderedDict()  # Field keys for dynamically generated tokens

        # Initialize a special field for the special tokens
        self.token2id[self.special_field_tag] = OrderedDict()

        # Placeholder for the vocabulary's filename, which will be set when saved
        self.filename = ''

        # Add all special tokens to the vocabulary, assigning unique IDs
        for token in self.special_tokens:
            global_id = len(self.id2token)  # Global unique ID for the token
            local_id = len(self.token2id[self.special_field_tag])  # Local ID within the special field

            # Map the token to its global and local IDs in the special field
            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            # Map the global ID back to the token and its associated metadata
            self.id2token[global_id] = [token, self.special_field_tag, local_id]


    def set_id(self, token, field_name, return_local=False, unk_token=-1):
        """
        Assigns a unique global and local ID to a token in a specified field.
        If the token does not exist in the vocabulary for the given field, it adds the token to the vocabulary.
    
        Args:
            token (str): The token to be assigned an ID.
            field_name (str): The field in which the token belongs (e.g., 'dynamic', 'static').
            return_local (bool): If True, returns the local ID (field-specific); otherwise, returns the global ID.
            unk_token (int or str): The token to be used as a placeholder for unknown tokens in the field.
                                    Defaults to -1 (indicating no specific unknown token).
    
        Returns:
            int: The global ID (or local ID if `return_local` is True) assigned to the token.
        """
    
        # Initialize placeholders for global and local IDs
        global_id, local_id = None, None
    
        # Step 1: Ensure the unknown token is added to the vocabulary for this field
        if unk_token not in self.token2id[field_name]:
            # Assign a new global ID and local ID for the unknown token
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])
            # Add the unknown token to the field-specific vocabulary and global ID mapping
            self.token2id[field_name][unk_token] = [global_id, local_id]
            self.id2token[global_id] = [unk_token, field_name, local_id]
    
        # Step 2: Add the token to the vocabulary if it doesn't already exist
        if token not in self.token2id[field_name]:
            # Assign a new global ID and local ID for the token
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])
            # Add the token to the field-specific vocabulary and global ID mapping
            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            # Retrieve the existing global and local IDs if the token is already present
            global_id, local_id = self.token2id[field_name][token]
    
        # Step 3: Return the appropriate ID based on the `return_local` flag
        if return_local:
            # Return the field-specific local ID
            return local_id
    
        # Return the global ID
        return global_id


    def get_id(self, token, field_name="", special_token=False, return_local=False):
        """
        Retrieves the global or local ID of a token in a specific field.
    
        Args:
            token (str): The token whose ID needs to be retrieved.
            field_name (str): The field where the token is stored. Defaults to an empty string.
            special_token (bool): If True, treats the token as a special token and uses the special field tag.
            return_local (bool): If True, returns the local ID (field-specific); otherwise, returns the global ID.
    
        Returns:
            int: The global or local ID of the token.
    
        Raises:
            Exception: If the token is not found in the specified field.
        """
        global_id, local_id = None, None
    
        # Use the special field tag if the token is marked as a special token
        if special_token:
            field_name = self.special_field_tag
    
        # Retrieve the token's ID from the specified field if it exists
        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]
        else:
            # Raise an exception if the token is not found
            raise Exception(f"token {token} not found in field: {field_name}")
    
        # Return the local ID if requested; otherwise, return the global ID
        if return_local:
            return local_id
    
        return global_id
    
    def set_field_keys(self, keys):
        """
        Initializes field keys and their associated token mappings in the vocabulary.
    
        Args:
            keys (list): A list of field keys to be initialized.
        """
        # Initialize a token mapping for each key and add it to the field keys
        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None
    
        # Retain the special field tag in the field keys for consistency
        self.field_keys[self.special_field_tag] = None
    
    def set_static_field_keys(self, keys):
        """
        Sets up static field keys, which are fields with predefined or non-changing tokens.
    
        Args:
            keys (list): A list of static field keys to be initialized.
        """
        # Initialize static field keys and set them to None for now
        for key in keys:
            self.static_field_keys[key] = None
    
        # Retain the special field tag in the static field keys
        self.static_field_keys[self.special_field_tag] = None
    
    def set_dynamic_field_keys(self, keys):
        """
        Sets up dynamic field keys, which are fields with tokens that can change or evolve.
    
        Args:
            keys (list): A list of dynamic field keys to be initialized.
        """
        # Initialize dynamic field keys and set them to None for now
        for key in keys:
            self.dynamic_field_keys[key] = None
    
        # Retain the special field tag in the dynamic field keys
        self.dynamic_field_keys[self.special_field_tag] = None
    
    def get_field_ids(self, field_name, return_local=False):
        """
        Retrieves all IDs (global or local) for tokens in a specific field.
    
        Args:
            field_name (str): The name of the field whose token IDs are to be retrieved.
            return_local (bool): If True, returns local IDs (field-specific); otherwise, returns global IDs.
    
        Returns:
            list: A list of global or local IDs for all tokens in the field.
    
        Raises:
            Exception: If the specified field name is invalid.
        """
        # Check if the field exists in the token-to-ID mapping
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            # Raise an exception if the field name is invalid
            raise Exception(f"field name {field_name} is invalid.")
    
        # Select the index for local IDs if requested, otherwise use the global IDs
        selected_idx = 0
        if return_local:
            selected_idx = 1
    
        # Return the list of IDs (global or local) for the tokens in the field
        return [ids[idx][selected_idx] for idx in ids]


    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)

    def get_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_static_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.static_field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_dynamic_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.dynamic_field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        special_tokens_map = {}
        # TODO : remove the dependency of re-initializing here. retrieve from field_key = SPECIAL
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_


def merge_vocab(vocab: Vocabulary, other_vocab: Vocabulary) -> Vocabulary:
    final_vocab = copy.deepcopy(vocab)
    new_field_list = [field for field in final_vocab.field_keys.keys() if field != final_vocab.special_field_tag]
    extend_field_list = [field for field in other_vocab.field_keys.keys()
                         if field not in final_vocab.field_keys.keys()]
    new_field_list.extend(extend_field_list)

    final_vocab.field_keys = OrderedDict()
    for field in new_field_list:
        final_vocab.field_keys[field] = None
    final_vocab.field_keys[final_vocab.special_field_tag] = None  # retain the order of columns

    for field in extend_field_list:
        final_vocab.token2id[field] = OrderedDict()
        for token in other_vocab.token2id[field].keys():
            final_vocab.set_id(token, field)

    return final_vocab


def delete_field(vocab: Vocabulary, field_to_delete: str) -> Vocabulary:
    if field_to_delete not in vocab.field_keys.keys():
        raise ValueError(f'{field_to_delete} is not in the given vocab')
    new_vocab = Vocabulary(adap_thres=vocab.adap_thres, target_column_name=vocab.target_column_name)
    new_field_list = [i for i in vocab.field_keys.keys() if i != field_to_delete
                      and i != vocab.special_field_tag]
    new_dynamic_field_list = [i for i in vocab.dynamic_field_keys.keys()
                              if i != field_to_delete and i != vocab.special_field_tag]
    new_static_field_list = [i for i in vocab.static_field_keys.keys()
                             if i != field_to_delete and i != vocab.special_field_tag]
    new_vocab.set_field_keys(new_field_list)
    new_vocab.set_dynamic_field_keys(new_dynamic_field_list)
    new_vocab.set_static_field_keys(new_static_field_list)
    for field in new_field_list:
        for token in vocab.token2id[field].keys():
            new_vocab.set_id(token, field)
    return new_vocab


def add_field_and_ids(vocab: Vocabulary, field_to_add: str, ordered_token_list) -> Vocabulary:
    if field_to_add in vocab.field_keys.keys():
        raise ValueError(f'{field_to_add} is already in the given vocab')
    new_vocab = copy.deepcopy(vocab)
    new_field_list = [field for field in new_vocab.field_keys.keys() if field != new_vocab.special_field_tag]
    new_field_list.append(field_to_add)

    new_vocab.field_keys = OrderedDict()
    for field in new_field_list:
        new_vocab.field_keys[field] = None
    new_vocab.field_keys[new_vocab.special_field_tag] = None  # retain the order of columns

    new_vocab.token2id[field_to_add] = OrderedDict()
    for token in ordered_token_list:
        new_vocab.set_id(token, field_to_add)
    return new_vocab
    