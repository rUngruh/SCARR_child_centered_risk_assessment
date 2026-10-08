ML_limits = []
MAL_limits = []


age_limit_color_coding = {
    "Approved":  "#4CAF50",  # green
    "G":         "#8BC34A",  # light green
    "PG":        "#CDDC39",  # lime
    "PG-13":     "#FFC107",  # amber
    "R":         "#FF9800",  # orange
    "NC-17":     "#F44336",  # red
    "R+":        "#F44336",  # red
    "Rx":        "#9C27B0",  # purple
    "Not Rated": "#9E9E9E",  # gray
}


from pathlib import Path
import os

from utils.genre_processing import MAL_genres, ML_genres
this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = Path(os.path.dirname(this_file_path))


if not os.path.exists(this_file_path / 'ML_age_limits.txt'):
    print("Warning: ML_age_limits.txt not found. Please run ML_process_ratings.py to generate it.")
else:
    with open(this_file_path / 'ML_age_limits.txt', 'r') as f:
        for line in f:
            ML_limits.append(line.strip())

if not os.path.exists(this_file_path / 'MAL_age_limits.txt'):
    print("Warning: MAL_age_limits.txt not found. Please run MAL_process_ratings.py to generate it.")
else:
    with open(this_file_path / 'MAL_age_limits.txt', 'r') as f:
        for line in f:
            MAL_limits.append(line.strip())

def save_age_limits(age_limits, dataset):
    if dataset == 'ml':
        global ML_limits
        d = 'ML'
        ML_limits = list(age_limits)
    elif dataset == 'mal':
        global MAL_limits
        d = 'MAL'
        MAL_limits = list(age_limits)
    else:
        raise ValueError("Invalid dataset. Choose from 'ml', 'mal'.")

    age_limit_path = this_file_path / f'{d}_age_limits.txt'
    with open(age_limit_path, 'w') as f:
        for age_limit in age_limits:
            f.write(f"{age_limit}\n")
    print(f"Age limits saved to {age_limit_path}")

def get_age_limits(dataset):
    if dataset == 'ml':
        return ML_limits
    elif dataset == 'mal':
        return MAL_limits
    else:
        raise ValueError("Invalid dataset. Choose from 'ml', 'mal'.")

def transform_limit_profile(profile, to, dataset):
    age_limits = get_age_limits(dataset)
    if isinstance(profile, list):
        age_limit_dict = {genre: profile[i] for i, genre in enumerate(age_limits) if profile[i] != 0}
    elif isinstance(profile, str):
        age_limit_dict = {genre: float(score) for genre, score in zip(age_limits, profile.split(','))}
    elif isinstance(profile, dict):
        age_limit_dict = profile
    else:
        raise ValueError("Invalid profile format. Provide as list, str, or dict.")
    
    if to == 'list':
        return [age_limit_dict.get(genre, 0) for genre in age_limits]
    elif to == 'str':
        return ','.join([str(age_limit_dict.get(genre, 0)) for genre in age_limits])
    elif to == 'dict':
        return age_limit_dict
    else:
        raise ValueError("Invalid target format. Choose from list, str, or dict.")
    
def get_unsuitable_limits(dataset, age):
    if dataset == 'ml':
        if age == 'Under 18':
            return ['R', 'NC-17', 'Rx']
        else:            
            return []
        
    elif dataset == 'mal':
        if age < 17:
            return ['R', 'R+', 'Rx']
        elif age < 18:
            return ['R+', 'Rx']
        else:
            return []
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    
def get_high_maturity_limits(dataset):
    if dataset == 'ml':
        return ['R', 'NC-17', 'Rx']
    elif dataset == 'mal':
        return ['R', 'R+', 'Rx']
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    
def label_to_numeric(label, dataset):
    age_limits = get_age_limits(dataset)
    if label in age_limits:
        return age_limits.index(label)
    else:
        raise ValueError(f"Label '{label}' not found in {dataset} age limits.")
    
def numeric_to_label(numeric, dataset):
    age_limits = get_age_limits(dataset)
    if 0 <= numeric < len(age_limits):
        return age_limits[numeric]
    else:
        raise ValueError(f"Numeric value '{numeric}' is out of range for {dataset} age limits.")