GR_genres = []
MIND_small_genres = []
MIND_large_genres = []
ML_genres = []
MAL_genres = []


from pathlib import Path
import os
this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = Path(os.path.dirname(this_file_path))

# if not os.path.exists(this_file_path / 'GR_genres.txt'):
#     print("Warning: GR_genres.txt not found. Please run GR_create_genre_dict.py to generate it.")
# else:
#     with open(this_file_path / 'GR_genres.txt', 'r') as f:
#         for line in f:
#             GR_genres.append(line.strip())
        
        
# with open(f'{this_file_path}/MIND_small_genres.txt', 'r') as f:
#     for line in f:
#         MIND_small_genres.append(line.strip())

# if not os.path.exists(this_file_path / 'MIND_large_genres.txt'):
#     print("Warning: MIND_large_genres.txt not found. Please run MIND_preprocess.py with subset 'train' to generate it.")
# else:
#     with open(this_file_path / 'MIND_large_genres.txt', 'r') as f:
#         for line in f:
#             MIND_large_genres.append(line.strip())
    
# if not os.path.exists(this_file_path / 'ML_genres.txt'):
#     print("Warning: ML_genres.txt not found. Please run ML_process_ratings.py to generate it.")
# else:
#     with open(this_file_path / 'ML_genres.txt', 'r') as f:
#         for line in f:
#             ML_genres.append(line.strip())

if not os.path.exists(this_file_path / 'MAL_genres.txt'):
    print("Warning: ML_genres.txt not found. Please run MAL_process_ratings.py to generate it.")
else:
    with open(this_file_path / 'MAL_genres.txt', 'r') as f:
        for line in f:
            MAL_genres.append(line.strip())


def save_genres(genres, dataset):
    # if dataset == 'gr':
    #     d = 'GR'
    #     GR_genres = list(genres)
    # elif dataset == 'mind_small':
    #     d = 'MIND_small'
    #     MIND_small_genres = list(genres)
    # elif dataset == 'mind_large':
    #     d = 'MIND_large'
    #     MIND_large_gernes = list(genres)
    if dataset == 'ml':
        global ML_genres
        d = 'ML'
        ML_genres = list(genres)
    elif dataset == 'mal':
        global MAL_genres
        d = 'MAL'
        MAL_genres = list(genres)
    else:
        raise ValueError("Invalid dataset. Choose from 'ml', 'gr' or 'mind_small'.")
    
    genre_path = this_file_path / f'{d}_genres.txt'
    with open(genre_path, 'w') as f:
        for genre in genres:
            f.write(f"{genre}\n")
    print(f"Genres saved to {genre_path}")

def get_genres(dataset):
    if dataset == 'gr':
        return GR_genres
    elif dataset == 'mind_small':
        return MIND_small_genres
    elif dataset == 'mind_large':
        return MIND_large_genres
    elif dataset == 'ml':
        return ML_genres
    elif dataset == 'mal':
        return MAL_genres
    else:
        raise ValueError("Invalid dataset. Choose from 'gr'.")
       

def transform_genre_profile(profile, to, dataset):
    genres = get_genres(dataset)
    if isinstance(profile, list):
        genre_dict = {genre: profile[i] for i, genre in enumerate(genres) if profile[i] != 0}
    elif isinstance(profile, str):
        genre_dict = {genre: float(score) for genre, score in zip(genres, profile.split(','))}
    elif isinstance(profile, dict):
        genre_dict = profile
    else:
        raise ValueError("Invalid profile format. Provide as list, str, or dict.")
    
    if to == 'list':
        return [genre_dict.get(genre, 0) for genre in genres]
    elif to == 'str':
        return ','.join([str(genre_dict.get(genre, 0)) for genre in genres])
    elif to == 'dict':
        return genre_dict
    else:
        raise ValueError("Invalid target format. Choose from list, str, or dict.")
    
