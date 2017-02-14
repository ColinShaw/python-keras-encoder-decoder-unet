import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def load_data():
    f1 = pd.read_csv('training-data/object-detection-crowdai/labels.csv', header=0)
    d1 = f1[(f1['Label'] == 'Car') | (f1['Label'] == 'Truck')]
    d1['File_Path'] = 'training-data/object-detection-crowdai/' + d1['Frame']
    f2 = pd.read_csv('training-data/object-dataset/labels.csv', header=0)
    d2 = f2[(f2['Label'] == 'car') | (f2['Label'] == 'truck')]
    d2['File_Path'] = 'training-data/object-dataset/' + d2['Frame']
    return pd.concat([d1,d2], axis=0)

def fix_range(a, b):
    return (a,b) if a<b else (b,a)

def load_image(file_path):
    feature = cv2.imread(file_path)
    label = np.zeros_like(feature[:,:,0], np.uint8)
    return feature, label

def save_image(feature, label, index, output_size):
    feature = cv2.resize(feature, output_size)
    label = cv2.resize(label, output_size)
    cv2.imwrite('feature-images/' + str(index) + '.jpg', feature)
    cv2.imwrite('label-images/' + str(index) + '.jpg', label)

def process_images(output_size):
    data = load_data()
    unique = data.File_Path.unique()
    for index, file_path in tqdm(enumerate(unique), total=len(unique)):
        feature, label = load_image(file_path)
        objects = data[data['File_Path'] == file_path].reset_index()
        for i in range(len(objects.File_Path)):
            xmin, xmax = fix_range(objects['xmin'][i], objects['xmax'][i])
            ymin, ymax = fix_range(objects['ymin'][i], objects['ymax'][i])
            label[ymin:ymax,xmin:xmax] = 255
        save_image(feature, label, index, output_size)

process_images((240,160))
