# Case Study
This repository includes the code to reproduce our experiments from Section 6.

# Reproducibility Code
## Set up environment
```
conda env create  -n 2026_case_study -f environment.yml
conda activate 2026_case_study
```

## Data gathering
Gather the publicly available dataset [MyAnimeList](https://www.kaggle.com/datasets/azathoth42/myanimelist).

The data directory should look like this:
```
├── data
│   ├── raw
│   │   ├── MyAnimeList
│   │   │   ├── AnimeList.csv
│   │   │   ├── anime_cleaned.csv
│   │   │   ├── animelists_cleaned.csv
│   │   │   ├── users_cleaned.csv
│   │   │   ├── ...
│   ├── processed
├── ...
```

### Install Elliot
```
cd Case_Study

conda create --yes --name elliot-env python=3.8
conda activate elliot-env
git clone https://github.com//sisinflab/elliot.git && cd elliot
pip install --upgrade pip
pip install -e . --verbose

# The following steps were accomplished to align the environment with our requirements.
pip install protobuf==3.20.3
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install cudnn=7.6.5
conda install numpy=1.23
```

### Prepare Dataset
Steps for dataset preprocessing.
```
cd Preprocessing
python MAL_process_ratings.py --experiment_year 2015
python process_age_limits.py
python data_enrichment.py
python split_set.py --year 2015

cd ..
python Recommender_Experiment/prepare_elliot.py
```

## Risk Assessment Meausure setup
Prepare the environment for the risk assessment.

```
conda create -n laion-nsfw python=3.13
conda activate laion-nsfw

pip install tensorflow autokeras torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install pandas
```

## Risk Measurement
Quantify the risks in anime using the risk assessment metrics.

- First run `risk_detection/risk_annotation.ipynb`
- Then run `risk_detection/risk_annotation_image.ipynb`


## Run Recommender Experiments
Run experiments using elliot.

### Run Experiments
To run hyperparameters using Elliot, simply follow elliots logic with the provided config files.

```
cd 2026_UMAP_age_ratings/elliot
conda activate elliot_env

python start_experiments.py --config mal_hyperparameter_tuning
```

### Update configuration file with tuned hyperparameters
We created two files for automatically processing the outcomes of hyperparameter tuning in order to create config files for the actual experiments.

```
cd Case_Study
conda activate 2026_case_study
python Recommender_Experiment/configure_final_config.py --dataset mal
```

### Run elliot with updated config file
```
cd elliot
conda activate elliot_env
python start_experiments.py --config mal_hyperparameter_tuning_final
```

## Analyze Results
Aggregate results into user profiles and recommendation profiles by running
```
cd Recommender_Experiment
conda activate 2026_case_study
python create_user_profiles.py
python create_recommendation_profiles.py
```

Afterward, analyze the results using `Results_Analysis/susceptibility_estimation.ipynb`.