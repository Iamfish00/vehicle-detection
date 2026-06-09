import json
import os
p = r'D:\html\iot\archive (1)\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json'
print('exists', os.path.exists(p))
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
print('len', len(data))
print('keys', list(data[0].keys()))
print('name', data[0]['name'])
print('labels sample', data[0]['labels'][:3])
