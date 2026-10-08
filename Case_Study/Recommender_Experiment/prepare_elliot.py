
import os
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



import shutil
from pathlib import Path
data_dir = Path('../data')

    
rec_path = data_dir / 'rec'


elliot_path = Path('../elliot')

save_path = elliot_path / 'data' / 'MAL'

config_save_path = elliot_path / 'config_files'

config_path = './config_files'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in os.listdir(save_path):
    file_path = os.path.join(save_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

for file in os.listdir(rec_path):
    src = os.path.join(rec_path, file)
    dst = os.path.join(save_path, file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        
for file in os.listdir(config_path):
    src = os.path.join(config_path, file)
    dst = os.path.join(config_save_path, file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        
print(f'Data and config files for MAL have been prepared in Elliot framework.')