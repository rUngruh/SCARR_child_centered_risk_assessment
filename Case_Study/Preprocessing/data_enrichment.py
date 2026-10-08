import sys
import pandas as pd
import os
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from pathlib import Path
dataset_dir = Path('../data')

import utils.genre_processing as gp


save_dir = dataset_dir / 'processed' / 'MAL'
data_dir = dataset_dir / 'raw' / 'MyAnimeList'

item_raw_path = data_dir / 'anime_cleaned.csv'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import argparse
parser = argparse.ArgumentParser(description='Process MAL dataset.')
parser.add_argument('--experiment_year', type=int, default=2015, help='Year to filter interactions for the experiment.')
args = parser.parse_args()

experiment_year = args.experiment_year

experiment_year = 2015

expItemPath = save_dir / f'items_{experiment_year}.tsv'

expItemPathEnriched = save_dir / f'items_{experiment_year}_enriched.tsv'

items = pd.read_csv(expItemPath, delimiter='\t') # Index(['item_id', 'title', 'title_english', 'age_rating_raw', 'genres'], dtype='str')


import requests

from dotenv import load_dotenv
from pathlib import Path
env_path =  'config.env'
load_dotenv(dotenv_path=env_path)

CLIENT_ID = os.getenv("CLIENT_ID")

headers = {
    "X-MAL-CLIENT-ID": CLIENT_ID
}


import re

def clean_synopsis(text):
    if not text:
        return None

    # Remove MAL attribution
    text = text.replace("[Written by MAL Rewrite]", "")

    # Replace tabs/newlines with spaces
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


from tqdm import tqdm

cover_dir = save_dir / "cover_art"
cover_dir.mkdir(parents=True, exist_ok=True)

def get_anime_info(anime_id: int, retries=5):
    for attempt in range(retries):

        try:
            response = requests.get(
                f"https://api.myanimelist.net/v2/anime/{anime_id}",
                headers=headers,
                params={
                    "fields": "title,synopsis,main_picture"
                },
                timeout=30
            )

            # retry server errors
            if response.status_code in [500, 502, 503, 504]:
                wait = 2 ** attempt
                print(
                    f"MAL server error {response.status_code} "
                    f"for {anime_id}, retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            image_url = None

            if "main_picture" in data:
                image_url = (
                    data["main_picture"].get("large")
                    or data["main_picture"].get("medium")
                )

            return {
                "title": data.get("title"),
                "synopsis": clean_synopsis(data.get("synopsis")),
                "image_url": image_url
            }


        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(
                f"Timeout for {anime_id}, retrying in {wait}s..."
            )
            time.sleep(wait)


        except requests.exceptions.RequestException as e:
            print(f"Request failed for {anime_id}: {e}")
            break


    # after all retries fail
    return {
        "title": None,
        "synopsis": None,
        "image_url": None
    }


def download_cover(image_url: str, item_id: int):
    if not image_url:
        return

    out_path = cover_dir / f"{item_id}.jpg"

    if out_path.exists():
        return

    r = requests.get(image_url, stream=True, timeout=30)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            
        
checkpoint_path = save_dir / f"items_{experiment_year}_checkpoint.tsv"


# Load checkpoint if it exists
if checkpoint_path.exists():
    print("Loading checkpoint...")
    enriched_items = pd.read_csv(
        checkpoint_path,
        sep="\t"
    )

    processed_ids = set(enriched_items["item_id"])

    remaining_items = items[
        ~items["item_id"].isin(processed_ids)
    ]

    results = enriched_items.to_dict("records")

else:
    print("Starting from scratch...")
    remaining_items = items
    results = []


print(f"Remaining items: {len(remaining_items)}")


CHECKPOINT_EVERY = 100


for idx, row in enumerate(
    tqdm(remaining_items.itertuples(index=False),
         total=len(remaining_items))
):

    item_id = int(row.item_id)

    try:
        info = get_anime_info(item_id)

        # keep original columns + add synopsis
        result = row._asdict()

        result["synopsis"] = info["synopsis"]

        results.append(result)


        # download cover
        download_cover(
            info["image_url"],
            item_id
        )


    except Exception as e:
        print(f"Failed {item_id}: {e}")

        result = row._asdict()
        result["synopsis"] = None

        results.append(result)


    # checkpoint
    if (idx + 1) % CHECKPOINT_EVERY == 0:

        checkpoint_df = pd.DataFrame(results)

        checkpoint_df.to_csv(
            checkpoint_path,
            sep="\t",
            index=False
        )

        print(
            f"Checkpoint saved ({len(results)} items)"
        )


    # avoid MAL rate limits
    time.sleep(0.5)



# Final save
final_df = pd.DataFrame(results)

final_df.to_csv(
    expItemPathEnriched,
    sep="\t",
    index=False
)


print("Finished!")
print(f"Saved {len(final_df)} items")