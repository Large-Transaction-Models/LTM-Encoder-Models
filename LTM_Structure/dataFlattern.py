import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Function to get the 10 most recent transactions per user (equivalent to get_recent_transactions_per_user)
def get_recent_transactions_per_user(data, transactions, user_column='user', timestamp_column='timestamp'):
    # Filter transactions to only include users in the input dataset
    recent_transactions = transactions[transactions[user_column].isin(data[user_column])].copy()
    
    # Sort by user and timestamp (descending)
    recent_transactions = recent_transactions.sort_values(by=[user_column, timestamp_column], ascending=[True, False])
    
    # Group by user and take the 10 most recent transactions
    recent_transactions = recent_transactions.groupby(user_column).head(10).reset_index(drop=True)
    
    # Merge survival data (timeDiff, status) with transactions
    unique_data = data.drop_duplicates(subset=[user_column])[[user_column, 'timeDiff', 'status']]
    recent_transactions = recent_transactions.merge(unique_data, on=user_column, how='left')
    
    # Convert timeDiff from seconds to days
    recent_transactions['timeDiff'] = recent_transactions['timeDiff'] / 86400
    
    # Fill NA values with -999
    recent_transactions = recent_transactions.fillna(-999)
    
    # Reorder columns to have timeDiff and status first
    cols = ['timeDiff', 'status'] + [col for col in recent_transactions.columns if col not in ['timeDiff', 'status']]
    recent_transactions = recent_transactions[cols]
    
    return recent_transactions

# Function to extract user history features (equivalent to get_user_history_features)
def get_user_history_features(data, user_column='user'):
    # Identify columns containing 'user' in their names, excluding the user_column itself
    history_features = [col for col in data.columns if user_column in col and col != user_column]
    return history_features

# Function to flatten transactions (equivalent to flatten_transactions)
def flatten_transactions(data, user_column='user', timestamp_column='timestamp', max_transactions=10, user_history_features=None):
    if user_history_features is None:
        user_history_features = []
    
    # Step 1: Arrange and filter data by user and timestamp
    data = data.sort_values(by=[user_column, timestamp_column]).copy()
    data['transaction_index'] = data.groupby(user_column).cumcount() + 1
    data = data[data['transaction_index'] <= max_transactions].reset_index(drop=True)
    
    # Step 2: Handle historical features for users with fewer than max_transactions
    transaction_counts = data.groupby(user_column).size().reset_index(name='transaction_count')
    data = data.merge(transaction_counts, on=user_column, how='left')
    
    for feature in user_history_features:
        data.loc[(data['transaction_count'] < max_transactions) & (data['transaction_index'] == 1), feature] = np.nan
    
    data = data.drop(columns=['transaction_count'])
    
    # Step 3: Set historical features and timeDiff to NA for transactions beyond the first
    for feature in user_history_features:
        data.loc[data['transaction_index'] > 1, feature] = np.nan
    data.loc[data['transaction_index'] > 1, 'timeDiff'] = np.nan
    
    # Step 4: Flatten the data into one row per user
    pivot_cols = [col for col in data.columns if col not in [user_column, 'transaction_index']]
    flattened_df = data.pivot(index=user_column, columns='transaction_index', values=pivot_cols)
    
    # Flatten multi-level column names
    flattened_df.columns = [f"{col[0]}_transaction_{col[1]}" for col in flattened_df.columns]
    flattened_df = flattened_df.reset_index()
    
    # Step 5: Remove redundant columns (e.g., historical features for transactions 2+)
    columns_to_drop = []
    for feature in user_history_features:
        columns_to_drop.extend([f"{feature}_transaction_{i}" for i in range(2, max_transactions + 1)])
    timeDiff_columns = [f"timeDiff_transaction_{i}" for i in range(2, max_transactions + 1)]
    status_columns = [f"status_transaction_{i}" for i in range(2, max_transactions + 1)]
    columns_to_drop.extend(timeDiff_columns + status_columns)
    
    flattened_df = flattened_df.drop(columns=[col for col in columns_to_drop if col in flattened_df.columns])
    
    # Step 6: Replace all NA values with -999
    flattened_df = flattened_df.fillna(-999)
    
    return flattened_df

# Function to preprocess flattened data (equivalent to preprocess_flattened_data_s)
def preprocess_flattened_data_s(flattened_data, train_data=None, max_transactions=10, set_timeDiff=25):
    # Create an 'event' column based on the 'timeDiff_transaction_1' threshold
    flattened_data['event'] = np.where(flattened_data['timeDiff_transaction_1'] > set_timeDiff, 1, 0)
    
    # Drop unnecessary columns
    features_to_drop = ['timestamp', 'id', 'timeDiff', 'deployment', 'version', 'user', 'status', 'quarter']
    transaction_columns_to_drop = [f"{feature}_transaction_{i}" for feature in features_to_drop for i in range(1, max_transactions + 1)]
    
    flattened_data = flattened_data.drop(columns=[col for col in transaction_columns_to_drop if col in flattened_data.columns])
    if 'user' in flattened_data.columns:
        flattened_data = flattened_data.drop(columns=['user'])
    
    # Convert categorical columns to category type (equivalent to R factors)
    for col in flattened_data.select_dtypes(include=['object']).columns:
        flattened_data[col] = flattened_data[col].astype('category')
    
    # Align categorical levels with training data if provided
    if train_data is not None:
        categorical_features = flattened_data.select_dtypes(include=['category']).columns
        for feature in categorical_features:
            flattened_data[feature] = pd.Categorical(flattened_data[feature], categories=train_data[feature].cat.categories)
    
    return flattened_data

### Data Preparation Workflow

# Load transaction data (assuming the file path is adjusted for your environment)
transactions = pd.read_pickle("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/transactions_user_market_time_exoLagged.rds")

# Remove columns containing 'exo' in their names
transactions = transactions.loc[:, ~transactions.columns.str.contains('exo')]

# Load train and test splits (assuming get_train_test_data is available in Python)
indexEvent = "borrow"
outcomeEvent = "withdraw"
train, test = get_train_test_data(indexEvent, outcomeEvent)  # Placeholder; implement or import this function

# Get the 10 most recent transactions per user
train_recent_transactions = get_recent_transactions_per_user(train, transactions)
test_recent_transactions = get_recent_transactions_per_user(test, transactions)

# Convert Unix timestamp to readable DateTime format
train_recent_transactions['timestamp'] = pd.to_datetime(train_recent_transactions['timestamp'].astype(float), unit='s', origin='1970-01-01')
test_recent_transactions['timestamp'] = pd.to_datetime(test_recent_transactions['timestamp'].astype(float), unit='s', origin='1970-01-01')

# Extract user history features
user_history_features = get_user_history_features(train_recent_transactions, user_column='user')

# Flatten the data
flattened_train_df = flatten_transactions(train_recent_transactions, user_history_features=user_history_features)
flattened_test_df = flatten_transactions(test_recent_transactions, user_history_features=user_history_features)

# Preprocess for classification
classification_cutoff = get_classification_cutoff(indexEvent, outcomeEvent)  # Placeholder; implement or import this function
train_data = preprocess_flattened_data_s(flattened_train_df)
test_data = preprocess_flattened_data_s(flattened_test_df, train_data=train_data)

# Convert to DataFrame (if not already)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# At this point, train_data and test_data are ready for classification
print("Train data prepared:")
print(train_data.head())
print("Test data prepared:")
print(test_data.head())