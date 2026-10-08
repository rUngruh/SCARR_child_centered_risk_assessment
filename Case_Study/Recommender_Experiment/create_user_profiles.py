import os
import pandas as pd
import argparse
import yaml
import json


import utils.age_limit_processing as alp
import utils.genre_processing as gp

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Configure the config files after hyperparameter tuning.')
parser.add_argument('--year', type=int,  help='Profiles based on which year', default=2015)
parser.add_argument('--save_metrics', type=str, nargs='+', help='List of metrics to save', 
                    default=['ind_aud_labels', 'num_items', 'perc_genres'], 
                    choices=['ind_aud_labels', 'aud_labels_profiles', 'num_items', 'perc_genres'])
parser.add_argument('--rel_abs', type=str, help='Whether to save relative or absolute audience profiles', 
                    default='relative', choices=['relative', 'absolute', 'both'])
args = parser.parse_args()

# Use the dataset argument
dataset = 'mal'
year = args.year
save_metrics = args.save_metrics
rel_abs = args.rel_abs


from pathlib import Path
dataset_dir = Path('../data')


data_dir = dataset_dir / 'processed' / 'MAL'    
    
user_path = data_dir / f'users{"_"+str(year) if year is not None else ""}.tsv'
item_path = data_dir / f'items{"_"+str(year) if year is not None else ""}.tsv'



user_profiles_path = dataset_dir / 'rec' / 'train_validation.tsv'

results_path = dataset_dir / 'results'

if os.path.exists(results_path) is False:
    os.makedirs(results_path)
    
datset_result_path = results_path / dataset
if os.path.exists(datset_result_path) is False:
    os.makedirs(datset_result_path)
    
profile_save_path = datset_result_path / f'user_profiles.tsv'

age_limits = alp.get_age_limits(dataset)

items = pd.read_csv(item_path, sep='\t')


item_al_dict = items.set_index('item_id')['age_rating'].to_dict()
item_genre_dict = items.set_index('item_id')['genres'].to_dict()
item_genre_dict = {k: v.split(',') if pd.notna(v) else [] for k, v in item_genre_dict.items()}

def compute_absolute_profiles(items):
    age_limit_dict = {age_limit: 0 for age_limit in age_limits}
    for item_id in items:
        age_rating = item_al_dict.get(item_id)
        if age_rating in age_limit_dict:
            age_limit_dict[age_rating] += 1
    age_limit_profile = alp.transform_limit_profile(age_limit_dict, 'list', dataset)
    return age_limit_profile


def compute_absolute_genre_profiles(items):
    genre_dict = {genre: 0 for genre in gp.get_genres(dataset)}
    for item_id in items:
        item_genres = item_genre_dict.get(item_id, [])
        for genre in item_genres:
            if genre in genre_dict:
                genre_dict[genre] += 1
    return genre_dict

user_profiles = pd.read_csv(user_profiles_path, sep='\t', names=['user_id', 'item_id', 'score'])
users = pd.read_csv(user_path, sep='\t')
users = users[users['user_id'].isin(user_profiles['user_id'].unique())].copy()

users['item_consumption'] = users['user_id'].map(
    user_profiles.groupby('user_id')['item_id'].apply(list)
)

if 'num_items' in save_metrics:
    users['num_items'] = users['item_consumption'].map(lambda x: len(x) if x is not None else 0)

if 'perc_genres' in save_metrics:
    users['abs_genres'] = users['item_consumption'].map(lambda x: compute_absolute_genre_profiles(x) if x is not None else None)
    users['perc_genres'] = users.apply(
        lambda row: {
            genre: count / len(row['item_consumption'])
            for genre, count in row['abs_genres'].items()
        } if row['item_consumption'] is not None and len(row['item_consumption']) > 0 else None,
        axis=1
    )

    users['perc_comedy'] = users['perc_genres'].map(lambda x: x.get('Comedy', 0) if x is not None else 0)
    users['perc_action'] = users['perc_genres'].map(lambda x: x.get('Action', 0) if x is not None else 0)
    users['perc_drama'] = users['perc_genres'].map(lambda x: x.get('Drama', 0) if x is not None else 0)
    users['perc_romance'] = users['perc_genres'].map(lambda x: x.get('Romance', 0) if x is not None else 0)
    users['perc_kids'] = users['perc_genres'].map(lambda x: x.get('Fantasy', 0) if x is not None else 0)
    users['perc_fantasy'] = users['perc_genres'].map(lambda x: x.get('Kids', 0) if x is not None else 0)
    users['perc_horror'] = users['perc_genres'].map(lambda x: x.get('Horror', 0) if x is not None else 0)
    users['perc_hentai'] = users['perc_genres'].map(lambda x: x.get('Hentai', 0) if x is not None else 0)
    
    users.drop(columns=['abs_genres', 'perc_genres'], inplace=True)

if 'ind_aud_labels' in save_metrics or 'aud_labels_profiles' in save_metrics:
    users['aud_labels_abs'] = users['item_consumption'].map(
        lambda item_ids: compute_absolute_profiles(item_ids)
    )



    if rel_abs in ['relative', 'both']:
        users['aud_labels_rel'] = users['aud_labels_abs'].map(
            lambda x: [v / sum(x) if sum(x) > 0 else 0 for v in x]
        )




position_R = alp.label_to_numeric('R', dataset)
position_Rplus = alp.label_to_numeric('R+', dataset)
position_Rx = alp.label_to_numeric('Rx', dataset)

users.drop(columns=['item_consumption'], inplace=True)


        
if 'ind_aud_labels' in save_metrics:
    if rel_abs in ['absolute', 'both']:
        users['abs_R'] = users['aud_labels_abs'].apply(
            lambda profile: profile[position_R] if profile is not None else None
        )
        
        users['abs_Rplus'] = users['aud_labels_abs'].apply(
            lambda profile: profile[position_Rplus] if profile is not None else None
        )
        users['abs_Rx'] = users['aud_labels_abs'].apply(
            lambda profile: profile[position_Rx] if profile is not None else None
        )
        users['abs_R_Rplus_Rx'] = users['abs_R'] + users['abs_Rplus'] + users['abs_Rx']
    
    if rel_abs in ['relative', 'both']:
        
        users['perc_R'] = users['aud_labels_rel'].apply(
            lambda profile: profile[position_R] if profile is not None else None
        )
        users['perc_Rplus'] = users['aud_labels_rel'].apply(
            lambda profile: profile[position_Rplus] if profile is not None else None
        )
        
        users['perc_Rx'] = users['aud_labels_rel'].apply(
            lambda profile: profile[position_Rx] if profile is not None else None
        )

        users['perc_R_Rplus_Rx'] = users['perc_R'] + users['perc_Rplus'] + users['perc_Rx']




for col in users.columns:
    if 'profile' in col:
        users[col] = users[col].apply(lambda x: alp.transform_limit_profile(x, 'str', dataset) if x is not None else None)

if 'aud_labels_profiles' not in save_metrics:
    users = users.drop(columns=[col for col in users.columns if 'aud_labels' in col])
else:
    if rel_abs == 'relative':
        users = users.drop(columns=[col for col in users.columns if '_abs' in col])




users.to_csv(profile_save_path, sep='\t', index=False)
print(f"User profiles saved to {profile_save_path}")