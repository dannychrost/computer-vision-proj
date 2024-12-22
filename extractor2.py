import pandas as pd
import os
import requests
from tqdm import tqdm

df = pd.read_csv('IMDb-Face.csv')
images_location = 'downloads2'

os.makedirs(images_location, exist_ok=True)

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images"):
    actor_index = row['index']
    image_name = row['image']
    image_url = row['url']  
    actor_folder = os.path.join(images_location, actor_index)
    os.makedirs(actor_folder, exist_ok=True)
    output_path = os.path.join(actor_folder, image_name)
    if os.path.exists(output_path):
        continue

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        continue