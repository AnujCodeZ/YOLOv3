import json
import urllib.request
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

# read json file
clothing = []
with open('clothing.json') as f:
    for line in f:
        clothing.append(json.loads(line))

# split data
train_clothing, val_clothing = train_test_split(clothing, 
                                                test_size=0.1)

# categories
categories = []
for c in clothing:
    for a in c['annotation']:
        categories.extend(a['label'])
categories = list(set(categories))

# create dataset
def create_dataset(clothing, dataset_type):
    # making directories
    images_path = Path(f'../data/images/{dataset_type}')
    images_path.mkdir(parents=True, exist_ok=True)
    
    labels_path = Path(f'../data/labels/{dataset_type}')
    labels_path.mkdir(parents=True, exist_ok=True)
    
    for img_id, row in enumerate(tqdm(clothing)):
        # download images and save
        image_name = f'{img_id}.jpeg'
        img = urllib.request.urlopen(row['content'])
        img = Image.open(img)
        img = img.convert('RGB')
        img.save(str(images_path / image_name), 'JPEG')
        
        # making label text file
        label_name = f'{img_id}.txt'
        with (labels_path / label_name).open('w') as label_file:
            for a in row['annotation']:
                for label in a['label']:
                    # getting category index
                    category_idx = categories.index(label)
                    # bounding box points
                    points = a['points']
                    p1, p2 = points
                    x1, x2 = p1['x'], p2['x']
                    y1, y2 = p1['y'], p2['y']
                    w = x2 - x1
                    h = y2 - y1
                    x = x1 + w / 2
                    y = y1 + h / 2
                    # writing in text file
                    label_file.write(
                        f'{category_idx} {x} {y} {w} {h}\n'
                    )

create_dataset(train_clothing, 'train')
create_dataset(val_clothing, 'val')                    