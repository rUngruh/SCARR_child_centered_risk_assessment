######################################################################

# Adapted from https://github.com/rUngruh/2026_UMAP_age_ratings

######################################################################

import sys
import pandas as pd
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from pathlib import Path
dataset_dir = Path('./data')

import utils.genre_processing as gp


save_dir = dataset_dir / 'processed' / 'MAL'
data_dir = dataset_dir / 'raw' / 'MyAnimeList'
ratingsSavePath = save_dir / 'interactions.tsv'
itemSavePath = save_dir / 'items.tsv'
usersSavePath = save_dir / 'users.tsv'

item_path = data_dir / 'anime_cleaned.csv'
ratings_path = data_dir / 'animelists_cleaned.csv'
users_path = data_dir / 'users_cleaned.csv'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import argparse
parser = argparse.ArgumentParser(description='Process MAL dataset.')
parser.add_argument('--min_age', type=int, default=13, help='Minimum age of users to include.')
parser.add_argument('-no_binarization', action='store_false', help='Select this option if ratings should not be binarized to implicit feedback.')
parser.add_argument('--user_k_core', type=int, default=5, help='Minimum number of interactions per user for k-core filtering.')
parser.add_argument('--experiment_year', type=int, default=2015, help='Year to filter interactions for the experiment.')
args = parser.parse_args()

min_age = args.min_age
binarize = args.no_binarization
user_k_core = args.user_k_core
experiment_year = args.experiment_year

expRatingsSavePath = save_dir / f'interactions_{experiment_year}.tsv'
expItemSavePath = save_dir / f'items_{experiment_year}.tsv'
expUsersSavePath = save_dir / f'users_{experiment_year}.tsv'

print('Loading and processing...')
delimiter = ','

items = pd.read_csv(item_path, delimiter=delimiter)
items = items[['anime_id', 'title', 'title_english', 'rating', 'genre']]
items.rename(columns={'anime_id':'item_id', 'genre' : 'genres', 'rating':'age_rating_raw'}, inplace=True)

items['genres'] = items['genres'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])



print(f'Number of items: {items.shape[0]}')    
empty_tags_count = items[items['genres'].apply(lambda x: len(x) == 0)].shape[0]
print(f"Number of rows without a genre: {empty_tags_count}")

items['age_rating_raw'] = items['age_rating_raw'].fillna('None').apply(lambda x: x.split(' - ')[0])

valid_item_ids = items['item_id'].unique()

genres = set()
for item_list in items['genres']:
    genres.update(item_list)
        
gp.save_genres(genres, 'mal')
print(items.head(10))

items['genres'] = items['genres'].apply(lambda x: ','.join(x))

print('Processing complete.')

users = pd.read_csv(users_path, delimiter=delimiter)
user_to_id = users.set_index('username')['user_id'].to_dict()
users = users[['user_id', 'gender', 'location', 'birth_date']]


ratings = pd.read_csv(ratings_path, delimiter=delimiter)
ratings['user_id'] = ratings['username'].map(user_to_id)
ratings = ratings[['user_id', 'anime_id', 'my_score', 'my_last_updated', 'my_status']]
ratings.rename(columns={'anime_id': 'item_id', 'my_score': 'rating', 'my_last_updated': 'timestamp'}, inplace=True)


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])

ratings = ratings[
    (ratings['user_id'].notna()) &
    (ratings['my_status'] == 2) &
    (ratings['timestamp'].dt.year > 2005) # Some interactions are from 1970. We consider these invalid
].copy().reset_index(drop=True)
# According to cell 19 in https://github.com/racinmat/mal-analysis/blob/master/user_analysis.ipynb, 2 equals completed

ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])
users['birth_date'] = pd.to_datetime(users['birth_date'])

ref_date = pd.Timestamp(f'{experiment_year + 1}-01-01')

before_birthday = (
    (users['birth_date'].dt.month > ref_date.month) |
    (
        (users['birth_date'].dt.month == ref_date.month) &
        (users['birth_date'].dt.day > ref_date.day)
    )
)

users['age'] = (
    ref_date.year - users['birth_date'].dt.year - before_birthday.astype(int)
)


ratings = ratings.merge(users[['user_id', 'birth_date']], on='user_id', how='left')

# Compute age at time of interaction
ratings['age_at_interaction'] = (ratings['timestamp'] - ratings['birth_date']).dt.days / 365.25

ratings = ratings[ratings['age_at_interaction']>=min_age]

ratings.drop(columns=['my_status', 'birth_date'], inplace=True)


print(f'Number of ratings: {ratings.shape[0]}')
print(f'Number of unique users: {ratings["user_id"].nunique()}')
print(f'Number of unique items: {ratings["item_id"].nunique()}')

ratings = ratings[ratings['item_id'].isin(valid_item_ids)]
print(f'Number of ratings after filtering: {ratings.shape[0]}')
print(f'Number of unique users after filtering: {ratings["user_id"].nunique()}')
print(f'Number of unique items after filtering: {ratings["item_id"].nunique()}')



valid_users = ratings['user_id'].unique()

items = items[items['item_id'].isin(ratings['item_id'])]

items.to_csv(itemSavePath, sep='\t', index=False, header=True, mode='w')

users = users[users['user_id'].isin(ratings['user_id'])]

users.to_csv(usersSavePath, sep='\t', index=False, header=True, mode='w')

ratings.to_csv(ratingsSavePath, sep='\t', index=False, header=True, mode='w')


exp_ratings = ratings[ratings['timestamp'].dt.year == experiment_year].copy().reset_index(drop=True)

print(f'Number of ratings in {experiment_year}: {exp_ratings.shape[0]}')
print(f'Number of unique users in {experiment_year}: {exp_ratings["user_id"].nunique()}')
print(f'Number of unique items in {experiment_year}: {exp_ratings["item_id"].nunique()}')

del ratings


if binarize:
    exp_ratings = exp_ratings[exp_ratings['rating'] > 6]
    print(f'Number of ratings after binarization: {exp_ratings.shape[0]}')

if user_k_core > 1:
    user_counts = exp_ratings['user_id'].value_counts()
    valid_users = user_counts[user_counts >= user_k_core].index
    exp_ratings = exp_ratings[exp_ratings['user_id'].isin(valid_users)]
    print(f'Number of ratings after {user_k_core}-core filtering: {exp_ratings.shape[0]}')
    print(f'Number of unique users after {user_k_core}-core filtering: {exp_ratings["user_id"].nunique()}')
    print(f'Number of unique items after {user_k_core}-core filtering: {exp_ratings["item_id"].nunique()}')

exp_items = items[items['item_id'].isin(exp_ratings['item_id'])]
exp_items.to_csv(expItemSavePath, sep='\t', index=False, header=True, mode='w')

exp_ratings.to_csv(expRatingsSavePath, sep='\t', index=False, header=True, mode='w')
valid_users = exp_ratings['user_id'].unique()
exp_users = users[users['user_id'].isin(valid_users)]
exp_users.to_csv(expUsersSavePath, sep='\t', index=False, header=True, mode='w')