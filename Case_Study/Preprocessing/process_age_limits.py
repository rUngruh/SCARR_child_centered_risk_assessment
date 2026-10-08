import sys
import pandas as pd
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from pathlib import Path
dataset_dir = Path('../data')


import argparse
parser = argparse.ArgumentParser(description='Process age limits for datasets.')
year = parser.add_argument('--year', type=int, help='Year to filter the dataset (indicator as per filtering for timeframe)', default=2015)
args = parser.parse_args()

import utils.age_limit_processing as alp



    

save_dir = dataset_dir / 'processed' / 'MAL'

itemSavePath = save_dir / f'items_{args.year}.tsv'

items = pd.read_csv(itemSavePath, sep='\t', keep_default_na=False)

items['age_rating'] = items['age_rating_raw']

items['age_rating'].fillna('Not Rated', inplace=True)

rating_order = ["Not Rated", "G", "PG", "PG-13", "R", "R+", "Rx"]

alp.save_age_limits(rating_order, 'mal')
items.to_csv(itemSavePath, sep='\t', index=False, header=True)
print(f"Processed age ratings for MAL dataset and saved age limits.")
