import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pyreadr
import os
from os.path import join, basename
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath('../'))

from arguments import parse_arguments
from exp_components.utils import get_data_path

def get_previous_transactions_per_transaction(transactions, transaction_id_column='id', user_column='user', timestamp_column='timestamp', max_previous=9):
    """
    For each transaction ID, obtain the current transaction and up to max_previous transactions
    from the same user that occurred before it.
    
    Parameters:
    - transactions: DataFrame containing transaction data
    - transaction_id_column: Name of the transaction ID column (default 'id')
    - user_column: Name of the user ID column (default 'user')
    - timestamp_column: Name of the timestamp column (default 'timestamp')
    - max_previous: Maximum number of previous transactions to retrieve (default 9)
    
    Returns:
    - A list of tuples, each containing (current transaction Series, DataFrame of previous transactions)
    """
    # Sort by user and timestamp
    transactions = transactions.sort_values(by=[user_column, timestamp_column]).reset_index(drop=True)
    
    # Assign an index for each transaction within the user group
    transactions['transaction_index'] = transactions.groupby(user_column).cumcount()
    
    # For each transaction ID, get the previous transactions for that user
    previous_transactions = []
    for idx, row in transactions.iterrows():
        current_transaction = row
        user = row[user_column]
        current_timestamp = row[timestamp_column]
        # Get transactions for the same user with a timestamp earlier than the current one
        previous = transactions[
            (transactions[user_column] == user) & 
            (transactions[timestamp_column] < current_timestamp)
        ].sort_values(by=timestamp_column, ascending=False).head(max_previous)
        previous_transactions.append((current_transaction, previous))
    
    return previous_transactions

def flatten_previous_transactions(previous_transactions, transaction_id_column='id', user_column='user', max_previous=9):
    """
    Flatten each transaction ID and its preceding transactions into a single row,
    placing the user in the first column.
    
    Parameters:
    - previous_transactions: List returned from get_previous_transactions_per_transaction
    - transaction_id_column: Name of the transaction ID column (default 'id')
    - user_column: Name of the user ID column (default 'user')
    - max_previous: Maximum number of previous transactions to retrieve (default 9)
    
    Returns:
    - A flattened DataFrame with each row corresponding to a transaction ID.
    """
    flattened_data = []
    for current, previous in previous_transactions:
        row = {user_column: current[user_column], transaction_id_column: current[transaction_id_column]}
        # Add features from the current transaction
        for col in current.index:
            if col not in [transaction_id_column, user_column, 'timestamp']:
                row[f"{col}_current"] = current[col]
        # Add features from the previous transactions
        for i in range(max_previous):
            if i < len(previous):
                prev_transaction = previous.iloc[i]
                for col in prev_transaction.index:
                    if col not in [transaction_id_column, user_column, 'timestamp']:
                        row[f"{col}_previous_{i+1}"] = prev_transaction[col]
            else:
                # If there are fewer than max_previous transactions, fill with NaN
                for col in current.index:
                    if col not in [transaction_id_column, user_column, 'timestamp']:
                        row[f"{col}_previous_{i+1}"] = np.nan
        flattened_data.append(row)
    
    flattened_df = pd.DataFrame(flattened_data)
    
    # Reorder columns to place the user column first
    cols = [user_column, transaction_id_column] + [col for col in flattened_df.columns if col not in [user_column, transaction_id_column]]
    flattened_df = flattened_df[cols]
    
    return flattened_df

def preprocess_flattened_data_s(flattened_data, train_data=None, max_previous=9):
    """
    Preprocess the flattened data.
    
    Parameters:
    - flattened_data: Flattened DataFrame
    - train_data: (Optional) training data used for aligning categorical features
    - max_previous: Maximum number of previous transactions to retrieve (default 9)
    
    Returns:
    - A preprocessed DataFrame.
    """
    # Drop unnecessary columns (e.g., historical 'timestamp' columns)
    features_to_drop = ['timestamp']
    transaction_columns_to_drop = [
        f"{feature}_previous_{i}" for feature in features_to_drop for i in range(1, max_previous + 1)
    ] + [f"{feature}_current" for feature in features_to_drop]
    
    flattened_data = flattened_data.drop(
        columns=[col for col in transaction_columns_to_drop if col in flattened_data.columns]
    )
    
    # Convert object type columns to category type
    for col in flattened_data.select_dtypes(include=['object']).columns:
        flattened_data[col] = flattened_data[col].astype('category')
    
    # If training data is provided, align the categorical features
    if train_data is not None:
        categorical_features = flattened_data.select_dtypes(include=['category']).columns
        for feature in categorical_features:
            flattened_data[feature] = pd.Categorical(
                flattened_data[feature], categories=train_data[feature].cat.categories
            )
    
    # Fill all missing values with -1.
    # For categorical columns, add -1 to categories before filling if necessary.
    for col in flattened_data.columns:
        if flattened_data[col].dtype.name == 'category':
            if -1 not in flattened_data[col].cat.categories:
                flattened_data[col] = flattened_data[col].cat.add_categories([-1])
            flattened_data[col] = flattened_data[col].fillna(-1)
        else:
            flattened_data[col] = flattened_data[col].fillna(-1)
    
    return flattened_data

# Main program
args = parse_arguments()
data_path, feature_extension = get_data_path(args)

# Load transaction data (only the first 100000 rows)
transactions = pyreadr.read_r(f"{data_path}transactions{feature_extension}.rds")[None].head(100000)
print(f"Loaded transaction data from {data_path}transactions{feature_extension}.rds (first 100000 rows)")

# Filter data: drop columns containing 'exo' and remove transactions with type 'collateral'
transactions = transactions.loc[:, ~transactions.columns.str.contains('exo')]
transactions = transactions[transactions['type'] != 'collateral']

# Convert the timestamp to datetime format
transactions['timestamp'] = pd.to_datetime(transactions['timestamp'].astype(float), unit='s', origin='1970-01-01')

# Get previous transactions for each transaction ID
previous_transactions = get_previous_transactions_per_transaction(transactions)

# Flatten the data and place the user column as the first column
flattened_df = flatten_previous_transactions(previous_transactions)

# Preprocess the flattened data (all missing values are now replaced with -1)
processed_data = preprocess_flattened_data_s(flattened_df)

# Split into training and testing sets
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Save the data
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")
train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)
print(f"Training data saved to: {train_path}")
print(f"Test data saved to: {test_path}")

print("Training data preview:")
print(train_data.head())
print("Test data preview:")
print(test_data.head())
