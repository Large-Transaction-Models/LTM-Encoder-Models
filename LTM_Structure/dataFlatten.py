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

def get_recent_transactions_per_user(data, transactions, user_column='user', timestamp_column='timestamp'):
    recent_transactions = transactions[transactions[user_column].isin(data[user_column])].copy()
    
    recent_transactions = recent_transactions.sort_values(by=[user_column, timestamp_column], ascending=[True, False])
    
    recent_transactions = recent_transactions.groupby(user_column).head(10).reset_index(drop=True)
    
    numeric_cols = recent_transactions.select_dtypes(include=['int64', 'float64']).columns
    recent_transactions[numeric_cols] = recent_transactions[numeric_cols].fillna(-999)
    
    return recent_transactions

def get_user_history_features(data, user_column='user'):
    history_features = [col for col in data.columns if user_column in col and col != user_column]
    return history_features

def flatten_transactions(data, user_column='user', timestamp_column='timestamp', max_transactions=10, user_history_features=None):
    if user_history_features is None:
        user_history_features = []
    
    data = data.sort_values(by=[user_column, timestamp_column]).copy()
    data['transaction_index'] = data.groupby(user_column).cumcount() + 1
    data = data[data['transaction_index'] <= max_transactions].reset_index(drop=True)
    
    transaction_counts = data.groupby(user_column).size().reset_index(name='transaction_count')
    data = data.merge(transaction_counts, on=user_column, how='left')
    
    for feature in user_history_features:
        data.loc[(data['transaction_count'] < max_transactions) & (data['transaction_index'] == 1), feature] = np.nan
    
    data = data.drop(columns=['transaction_count'])
    
    for feature in user_history_features:
        data.loc[data['transaction_index'] > 1, feature] = np.nan
    
    pivot_cols = [col for col in data.columns if col not in [user_column, 'transaction_index']]
    flattened_df = data.pivot(index=user_column, columns='transaction_index', values=pivot_cols)
    
    flattened_df.columns = [f"{col[0]}_transaction_{col[1]}" for col in flattened_df.columns]
    flattened_df = flattened_df.reset_index()
    
    columns_to_drop = [f"{feature}_transaction_{i}" for feature in user_history_features for i in range(2, max_transactions + 1)]
    flattened_df = flattened_df.drop(columns=[col for col in columns_to_drop if col in flattened_df.columns])
    
    numeric_cols = flattened_df.select_dtypes(include=['int64', 'float64']).columns
    flattened_df[numeric_cols] = flattened_df[numeric_cols].fillna(-999)
    
    return flattened_df

def preprocess_flattened_data_s(flattened_data, train_data=None, max_transactions=10):

    features_to_drop = [ 'timestamp', 'user', 'quarter']
    transaction_columns_to_drop = [f"{feature}_transaction_{i}" for feature in features_to_drop for i in range(1, max_transactions + 1)]
    
    flattened_data = flattened_data.drop(columns=[col for col in transaction_columns_to_drop if col in flattened_data.columns])
    if 'user' in flattened_data.columns:
        flattened_data = flattened_data.drop(columns=['user'])
    
    for col in flattened_data.select_dtypes(include=['object']).columns:
        flattened_data[col] = flattened_data[col].astype('category')
    
    if train_data is not None:
        categorical_features = flattened_data.select_dtypes(include=['category']).columns
        for feature in categorical_features:
            flattened_data[feature] = pd.Categorical(flattened_data[feature], categories=train_data[feature].cat.categories)
    
    return flattened_data

args = parse_arguments()
data_path, feature_extension = get_data_path(args)

transactions = pyreadr.read_r(f"{data_path}transactions{feature_extension}.rds")[None]
print(f"Loaded transaction data from {data_path}transactions{feature_extension}.rds")

transactions = transactions.loc[:, ~transactions.columns.str.contains('exo')]
transactions = transactions[transactions['type'] != 'collateral']

recent_transactions = get_recent_transactions_per_user(transactions, transactions)

recent_transactions['timestamp'] = pd.to_datetime(recent_transactions['timestamp'].astype(float), unit='s', origin='1970-01-01')

user_history_features = get_user_history_features(recent_transactions, user_column='user')

flattened_df = flatten_transactions(recent_transactions, user_history_features=user_history_features)

processed_data = preprocess_flattened_data_s(flattened_df)

train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

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
