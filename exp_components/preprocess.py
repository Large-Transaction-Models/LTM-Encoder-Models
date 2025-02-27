# Import necessary libraries
import pyreadr
import os
import sys
sys.path.append(os.path.abspath('../'))
import pandas as pd
import numpy as np
import math

from pathlib import Path

def preprocess(args, data_path, feature_extension, log):
    
    file_path = f"{data_path}transactions{feature_extension}.rds"
    train_path = f"{data_path}preprocessed{feature_extension}/transactions{feature_extension}_train.csv"
    test_path = f"{data_path}preprocessed{feature_extension}/transactions{feature_extension}_test.csv"
    
    if args.check_preprocess_cached:
        if os.path.exists(train_path) and os.path.exists(test_path):
            log.info("Train and test data have already been preprocessed. Skipping preprocess.")
            return
    
    fullData = pyreadr.read_r(file_path)[None]  
    
    
    # Drop the unnecessary columns:
    # First, we can definitely drop any columns for which ALL values are NA:
    all_na_columns = fullData.columns[fullData.isna().all()].tolist()
    
    # We build the list of user-level features here, in case we want to drop them easily for some experiments:
    user_features = [col for col in fullData.columns if col.startswith('user') and col != "user"]
    
    # We build the list of time features in case we want to drop them easily for some experiments:
    time_features = ["timeOfDay", "dayOfWeek", "dayOfMonth", 
        "dayOfYear", "quarter", "dayOfQuarter",
        "sinTimeOfDay", "cosTimeOfDay", "sinDayOfWeek",
        "cosDayOfWeek", "sinDayOfMonth", "cosDayOfMonth",
        "sinDayOfQuarter", "cosDayOfQuarter", "sinDayOfYear",
        "cosDayOfYear", "sinQuarter", "cosQuarter", "isWeekend"]
    
    # We are going to drop the market features for now, because they are similar to exogenous
    # features and we will likely handle them differently later on.
    market_features = [col for col in fullData.columns if col.startswith('market')]
    
    # We are going to drop the exogenous features for now and handle them separately once we have 
    # the rest of the model working.
    exo_features = [col for col in fullData.columns if col.startswith('exo')]
    
    # There might be additional, hand-selected columns we want to drop just because they are not useful or because
    # they are going to be re-created in a more general way:
    other_columns_to_drop = ['logAmountUSD', 'logAmount', 'logAmountETH']
    
    # Concatenate these lists and then drop the columns:
    columns_to_drop = all_na_columns + other_columns_to_drop
    if not args.include_user_features:
        columns_to_drop += user_features
    if not args.include_market_features:
        columns_to_drop += market_features
    if not args.include_time_features:
        columns_to_drop += time_features
    if not args.include_exo_features:
        columns_to_drop += exo_features
    
    data = fullData.drop(columns=columns_to_drop, errors="ignore")
    
    data['timeFeature'] = ((data['timestamp'] - data['timestamp'].min())//60).astype(int)
    data['Year'] = pd.to_datetime(data['timestamp'], unit='s').dt.year
    
    # Let's make sure certain columns are cast properly to categorical columns so they 
    # are handled appropriately when building the vocabulary down the line.
    if "Aave" in args.dataset:
        categorical_columns = ['fromState', 'toState', 'borrowRateMode', 'reserve', 'type', 
                           'collateralReserve', 'borrowRateModeTo', 'borrowRateModeFrom',
                           'coinType', 'userReserveMode', 'userCoinTypeMode', 'userIsNew']
    elif "cosmetics" in args.dataset:
        # Categorical columns for cosmetics:
        categorical_columns = ['type', 'product_id', 'product_brand',
                            'category_1', 'category_2', 'category_3',
                            'newSession']
    elif "AML" in args.dataset:
        # AML categoricals
        categorical_columns = ['type', 'userBank', 'recipientBank', 'reserve']
    
    if args.include_time_features:
        categorical_columns += ["dayOfWeek", "dayOfMonth", "dayOfYear", "quarter", "dayOfQuarter","sinDayOfWeek",
                            "cosDayOfWeek", "sinDayOfMonth", "cosDayOfMonth",
                            "sinDayOfQuarter", "cosDayOfQuarter", "sinDayOfYear",
                            "cosDayOfYear", "sinQuarter", "cosQuarter", 'isWeekend', 'Year']
    
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    
    
    sort_columns =  ['user', 'timeFeature']
    new_data = data.sort_values(by=sort_columns)
    
    
    # columns we specifically don't want to log-transform:
    columns_not_to_log = ['timeFeature', 'id']
    """
    # List of keywords indicating a column should be log-transformed
    keywords_to_log = ['amount', 'sum', 'rate', 'count', 'activedays', 'sin', 'cos']
    
    # Identify columns to log-transform
    columns_to_log = [col for col in new_data.columns if any(keyword in col.lower() for keyword in keywords_to_log)]
    columns_to_log = list(set(columns_to_log))  # Remove duplicates
    columns_to_log = [x for x in columns_to_log if isinstance(x, (int, float))]
    """
    
    # We want to log-transform all numeric columns:
    columns_to_log = new_data.select_dtypes(include=['number']).columns.tolist()
    
    # Apply log transformation to identified columns
    for col in columns_to_log:
        if col in columns_not_to_log:
            continue
        new_data[col] = new_data[col].apply(lambda x: np.log(x) if x > 0 else np.nan)  # Use 1 for non-positive values
        
    new_data['rowNumber'] = np.arange(len(new_data))
    
    # List of columns to move to the front
    first_cols = ['rowNumber', 'user', 'timestamp', 'id']
    # Rearrange the columns
    rearranged_columns = first_cols + [col for col in new_data.columns if col not in first_cols]
    new_data = new_data[rearranged_columns]
    
    
    # Compute the min and max timestamps
    min_timestamp = new_data['timestamp'].min()
    max_timestamp = new_data['timestamp'].max()
    
    # Compute the two-thirds timestamp
    train_test_thres_timestamp = min_timestamp + (max_timestamp - min_timestamp) * args.train_test_thres
    
    # For cosmetics:
    train_data = new_data.loc[new_data['timestamp'] < train_test_thres_timestamp]
    basic_test_data = new_data.loc[new_data['timestamp'] >= train_test_thres_timestamp]
    
    train_user = set(train_data['user'].unique())
    test_user = set(basic_test_data['user'].unique())
    train_test_user = train_user.intersection(test_user)
    test_only_user = test_user.difference(train_user)
    groupby_columns = ['user']
    
    def get_index(x, seq_len):
        return x.index[-(seq_len-1):]
    
    
    test_extra_index = train_data.loc[train_data['user'].isin(train_test_user)].groupby(groupby_columns).apply(get_index, args.seq_len)
    test_extra_index = test_extra_index.explode()
    
    test_data = pd.concat([new_data.loc[test_extra_index], basic_test_data])
    
    
    test_data.sort_values(by=sort_columns, inplace=True)
    
    train_dir = Path(train_path)
    train_dir.parent.mkdir(exist_ok=True, parents=True)
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
