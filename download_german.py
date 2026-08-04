import os
import urllib.request
import pandas as pd

save_dir = 'data/german'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
file_path = f'{save_dir}/german.data'
print("Downloading...")
urllib.request.urlretrieve(url, file_path)
print("Finished downloading. Converting to CSV...")

df = pd.read_csv(file_path, header=None, sep=r'\s+')
df.to_csv(file_path, header=False, index=False)
print('German dataset downloaded and converted to CSV.')
