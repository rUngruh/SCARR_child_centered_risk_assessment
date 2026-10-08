######################################################################

# This scipt creates train, validation and test split, including removal of invalid profiles.
# The details can be specified in the command line arguments.

######################################################################

import os
from pathlib import Path
import pandas as pd
import argparse



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


parser = argparse.ArgumentParser(description='Create train, validation and test split, including  removal of invalid profiles.')
parser.add_argument('--year', type=int, help='Year to filter the dataset (indicator as per filtering for timeframe)', default=2015)
parser.add_argument('--remove_missing_profiles', action='store_true', help='Remove users with missing items in train, validation, or test sets')
parser.add_argument('--train_split_size', type=float, help='Train split size', default=0.7)
parser.add_argument('--validation_split_size', type=float, help='Validation split size', default=0.2)
args = parser.parse_args()



remove_missing_profiles = args.remove_missing_profiles


train_size = args.train_split_size
validation_size = args.validation_split_size
test_size = 1 - train_size - validation_size

assert 0 < train_size < 1
assert 0 < validation_size < 1
assert train_size + validation_size < 1



import dotenv
from pathlib import Path
data_dir = Path('../data')
dataset_dir = Path('../data/processed/MAL')



interactions_path = dataset_dir / f'interactions_{args.year}.tsv'
    
    
save_path = data_dir / 'rec'


train_path = save_path / 'train.tsv'
validation_path = save_path / 'validation.tsv'
train_val_path = save_path / 'train_validation.tsv'
test_path = save_path / 'test.tsv'

if not os.path.exists(save_path):
    os.makedirs(save_path)


if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(validation_path):
    os.remove(validation_path)
if os.path.exists(test_path):
    os.remove(test_path)

interactions = pd.read_csv(interactions_path, sep='\t')



interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'], keep='first')



interactions = interactions[['user_id', 'item_id', 'rating']]
train_chunks = []
validation_chunks = []
test_chunks = []

for user_id, user_data in interactions.groupby('user_id'):
    user_data = user_data.sample(frac=1, random_state=42)  # Shuffle user data
    
    n = len(user_data)
    n_train = int(train_size * n)
    n_validation = int(validation_size * n)
    
    train = user_data.iloc[:n_train]
    validation = user_data.iloc[n_train:n_train + n_validation]
    test = user_data.iloc[n_train + n_validation:]

    train_chunks.append(train)
    validation_chunks.append(validation)
    test_chunks.append(test)

train_chunk = pd.concat(train_chunks)
validation_chunk = pd.concat(validation_chunks)
test_chunk = pd.concat(test_chunks)
print("Finished splitting")

del interactions

if remove_missing_profiles:
    print("Removing missing profiles")

    # Get unique user IDs from each dataset once
    train_user_ids = set(train_chunk['user_id'].unique())
    validation_user_ids = set(validation_chunk['user_id'].unique())
    test_user_ids = set(test_chunk['user_id'].unique())

    # Find valid user IDs that are present in all three datasets
    valid_user_ids = train_user_ids & validation_user_ids & test_user_ids

    print(f'Number of valid users: {len(valid_user_ids)}')

    # Filter user_info and chunks by valid user IDs
    train_chunk = train_chunk[train_chunk['user_id'].isin(valid_user_ids)]
    validation_chunk = validation_chunk[validation_chunk['user_id'].isin(valid_user_ids)]
    test_chunk = test_chunk[test_chunk['user_id'].isin(valid_user_ids)]

    
train_chunk.to_csv(train_path, sep='\t', index=False, header=False)
validation_chunk.to_csv(validation_path, sep='\t', index=False, header=False)
test_chunk.to_csv(test_path, sep='\t', index=False, header=False)
pd.concat([train_chunk, validation_chunk]).to_csv(train_val_path, sep='\t', index=False, header=False)