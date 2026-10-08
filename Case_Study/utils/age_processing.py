ML_ages = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
MAL_ages = list(map(str, range(13, 30))) + ['30+']  # '13' to '30'


def get_ages(dataset):
    if dataset == 'ml':
        return ML_ages
    elif dataset == 'mal':
        return MAL_ages
    else:
        return None
    
    
def process_age(age, dataset):
    if dataset == 'ml':
        return age
    elif dataset == 'mal':
        if age >= 30:
            return '30+'
        else:
            return str(age)
    else:
        return None
    
def get_child_ages(dataset):
    if dataset == 'ml':
        return ['Under 18']
    elif dataset == 'mal':
        return [str(age) for age in range(13, 18)]
    else:
        return None