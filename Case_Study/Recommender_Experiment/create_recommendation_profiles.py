import os
import pandas as pd
import argparse
import yaml
import json
import numpy as np

import utils.age_limit_processing as alp

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = argparse.ArgumentParser(description='Configure the config files after hyperparameter tuning.')
parser.add_argument('--cutoff', type=int, help='Cutoffs for evaluation', default=10)
parser.add_argument('--recommenders', type=str, nargs='+', help='List of recommenders to process', default=['RP3beta'])
parser.add_argument('--year', type=int,  help='Profiles based on which year', default=2015)
parser.add_argument('--save_metrics', type=str, nargs='+', help='List of metrics to save', default=['ind_aud_labels', 'synopsis_risks', 'image_risks'], choices=['ind_aud_labels', 'aud_labels_profiles', 'synopsis_risks', 'image_risks'])

parser.add_argument('--rel_abs', type=str, help='Whether to save relative or absolute audience label profiles', default='relative', choices=['relative', 'absolute', 'both'])
args = parser.parse_args()

# Use the dataset argument
dataset = 'mal'
cutoff = args.cutoff
year = args.year
save_metrics = args.save_metrics
rel_abs = args.rel_abs

from pathlib import Path

dataset_dir = Path('../data')



    

data_dir = dataset_dir / 'processed' / 'MAL'    
    
user_path = data_dir / f'users{"_"+str(year) if year is not None else ""}.tsv'
item_path = data_dir / f'risk_items{"_"+str(year) if year is not None else ""}.tsv'


user_profiles_path = data_dir / 'rec' / 'train_validation.tsv'

elliot_path = Path('../elliot')

recs_path = elliot_path / 'results' / f'{dataset}_final' / 'recommendations'


results_path = dataset_dir / 'results'

if os.path.exists(results_path) is False:
    os.makedirs(results_path)
    
datset_result_path = results_path / dataset
if os.path.exists(datset_result_path) is False:
    os.makedirs(datset_result_path)
    
recommender_profiles_save_path = datset_result_path / f'recommender_profiles_{cutoff}.tsv'

age_limits = alp.get_age_limits(dataset)

items = pd.read_csv(item_path, sep='\t')

item_al_dict = items.set_index('item_id')['age_rating'].to_dict()
item_syn_sexual_dict = items.set_index('item_id')['text_sexual']
item_syn_violence_dict = items.set_index('item_id')['text_violence']
item_image_nsfw_dict = items.set_index('item_id')['nsfw_score']


def compute_absolute_audience_label_profiles(items):
    if items is None or len(items) == 0:
        return None
    age_limit_dict = {age_limit: 0 for age_limit in age_limits}
    for item_id in items:
        age_rating = item_al_dict.get(item_id)
        if age_rating in age_limit_dict:
            age_limit_dict[age_rating] += 1
    age_limit_profile = alp.transform_limit_profile(age_limit_dict, 'list', dataset)
    return age_limit_profile

def compute_risk_score(items, risk):
    if items is None or len(items) == 0:
        return None
    
    if risk == 'syn_sexual':
        d = item_syn_sexual_dict
    elif risk == 'syn_violence':
        d = item_syn_violence_dict
    elif risk == 'image_nsfw':
        d = item_image_nsfw_dict
    
    scores = [d[i] for i in items]
    quad_mean_score = float(np.sqrt(np.mean([i**2 for i in scores])))
    return quad_mean_score
    

users = pd.read_csv(user_path, sep='\t')
users = users[['user_id']]

position_R = alp.label_to_numeric('R', dataset)
position_Rplus = alp.label_to_numeric('R+', dataset)
position_Rx = alp.label_to_numeric('Rx', dataset)

for file in os.listdir(recs_path):
    recommender = file.split('_')[0]
    if '.tsv' in recommender:
        recommender = recommender.replace('.tsv','')
    if '.tsv' not in file:
        continue
    if recommender not in args.recommenders:
        continue
    recs_file_path = recs_path / file
    recommendations = pd.read_csv(recs_file_path, sep='\t', names=['user_id', 'item_id', 'score'])
    
    uid_rec_dict = recommendations.groupby('user_id')['item_id'].apply(lambda x: list(x)[:cutoff]).to_dict()
    
    if 'synopsis_risks' in save_metrics:
        users[recommender + '_syn_sexual'] = users['user_id'].apply(lambda uid: compute_risk_score(uid_rec_dict.get(uid), 'syn_sexual'))
        users[recommender + '_syn_violence'] = users['user_id'].apply(lambda uid: compute_risk_score(uid_rec_dict.get(uid), 'syn_violence'))
    
    if 'image_risks' in save_metrics:
        users[recommender + '_image_nsfw'] = users['user_id'].apply(lambda uid: compute_risk_score(uid_rec_dict.get(uid), 'image_nsfw'))

    
    # Compute absolute and relative audience label profiles for each recommender
    if 'aud_labels_profiles' in save_metrics or 'ind_aud_labels' in save_metrics:
        
        users[recommender+'_aud_labels_abs'] = users['user_id'].apply(lambda uid: compute_absolute_audience_label_profiles(uid_rec_dict.get(uid)))
        
        if rel_abs in ['relative', 'both']:
            users[recommender + '_aud_labels_rel'] = users[recommender + '_aud_labels_abs'].apply(
                lambda profile: (
                    [x / sum(profile) for x in profile]
                    if profile is not None and sum(profile) > 0
                    else None
                )
            )
        
        
        if 'ind_aud_labels' in save_metrics:
            if rel_abs in ['absolute', 'both']:
                users[recommender + '_abs_R'] = users[recommender + '_aud_labels_abs'].apply(
                    lambda profile: profile[position_R] if profile is not None else None
                )
                
                users[recommender + '_abs_Rplus'] = users[recommender + '_aud_labels_abs'].apply(
                    lambda profile: profile[position_Rplus] if profile is not None else None
                )
                users[recommender + '_abs_Rx'] = users[recommender + '_aud_labels_abs'].apply(
                    lambda profile: profile[position_Rx] if profile is not None else None
                )
                users[recommender + '_abs_R_Rplus_Rx'] = users[recommender + '_abs_R'] + users[recommender + '_abs_Rplus'] + users[recommender + '_abs_Rx']
            
            if rel_abs in ['relative', 'both']:
                
                users[recommender + '_perc_R'] = users[recommender + '_aud_labels_rel'].apply(
                    lambda profile: profile[position_R] if profile is not None else None
                )
                users[recommender + '_perc_Rplus'] = users[recommender + '_aud_labels_rel'].apply(
                    lambda profile: profile[position_Rplus] if profile is not None else None
                )
                
                users[recommender + '_perc_Rx'] = users[recommender + '_aud_labels_rel'].apply(
                    lambda profile: profile[position_Rx] if profile is not None else None
                )
                users[recommender + '_perc_R_Rplus_Rx'] = users[recommender + '_perc_R'] + users[recommender + '_perc_Rplus'] + users[recommender + '_perc_Rx']

if 'aud_labels_profiles' in save_metrics or 'ind_aud_labels' in save_metrics:

    # Clean up users DataFrame by removing rows with all NaN values in audience label columns
    users = users.dropna(subset=[col for col in users.columns if 'aud_labels' in col], how='all')

    for col in users.columns:
        if 'aud_labels' in col:
            users[col] = users[col].apply(lambda x: alp.transform_limit_profile(x, 'str', dataset) if x is not None else None)

if 'aud_labels_profiles' not in save_metrics:
    users = users.drop(columns=[col for col in users.columns if 'aud_labels' in col])
else:
    if rel_abs == 'relative':
        users = users.drop(columns=[col for col in users.columns if '_abs' in col])



users.to_csv(recommender_profiles_save_path, sep='\t', index=False)
print(f'Recommender profiles saved to: {recommender_profiles_save_path}')