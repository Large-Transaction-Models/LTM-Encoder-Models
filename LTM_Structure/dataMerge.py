import pandas as pd
import pyreadr
import os

y_train_path = "/data/IDEA_DeFi_Research/Data/Survival_Data/Borrow/Repay/y_train.rds"
y_train = pyreadr.read_r(y_train_path)[None]

# Load train dataset from CSV
train = pd.read_csv("~/LTM-Encoder-Models/LTM_Structure/data/train.csv")
test = pd.read_csv("~/LTM-Encoder-Models/LTM_Structure/data/test.csv")

# Merge train with y_train on the 'id' column using an inner join
# This will keep only rows with matching 'id' in both datasets
new_train = pd.merge(train, y_train, on='id', how='inner')
new_test = pd.merge(test, y_train, on='id', how='inner')

# Optionally, save the new merged dataset for training purposes
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
new_train_path = os.path.join(output_dir, "final_train.csv")
new_train.to_csv(new_train_path, index=False)

new_test_path = os.path.join(output_dir, "final_test.csv")
new_test.to_csv(new_test_path, index=False)

print("New training dataset preview:")
print(new_train.head())
