from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import cv2
from tqdm import tqdm
import models

def load_data(images):
    features = np.zeros((images, 160, 240, 3), dtype=np.uint8)
    labels = np.zeros((images, 160, 240, 1), dtype=np.float32)
    for i in tqdm(range(images), total=images):
        feature = cv2.imread('feature-images/' + str(i) + '.jpg')
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB)
        features[i] = feature
        label = cv2.imread('label-images/' + str(i) + '.jpg')
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = np.reshape(label, (np.shape(label)[0], np.shape(label)[1], 1))
        label = label / 127.5 - 1.0
        labels[i] = label
    return features, labels

def adjust_luminance(image):
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
    image[:,:,1] = image[:,:,1] * np.random.uniform(0.25, 1.0)
    return cv2.cvtColor(image, cv2.COLOR_HLS2RGB)

def translate_image(image, pixels, image_size):
    x = pixels * np.random.uniform() - (pixels / 2)
    y = pixels * np.random.uniform() - (pixels / 2)
    translated = cv2.warpAffine(image, np.float32([[1,0,x],[0,1,y]]), image_size)
    return np.resize(translated, (image.shape[0], image.shape[1], image.shape[2]))

def stretch_image(image, pixels, image_size):
    x1, y1 = np.random.randint(pixels), np.random.randint(pixels)
    x2, y2 = np.random.randint(pixels), np.random.randint(pixels)
    source = np.float32([[x1,y1],[image.shape[1]-x2,y1],[image.shape[1]-x2,image.shape[0]-y2],[x1,image.shape[0]-y2]])
    dest = np.float32([[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]])
    transform = cv2.getPerspectiveTransform(source, dest)
    stretched = cv2.warpPerspective(image, transform, image_size)
    return np.resize(stretched, (image.shape[0], image.shape[1], image.shape[2]))

def augmentation_pipeline(feature, label, image_size):
    feature = adjust_luminance(feature)
    feature = translate_image(feature, 30, image_size)
    feature = stretch_image(feature, 20, image_size)
    label = translate_image(label, 30, image_size)
    label = stretch_image(label, 20, image_size)
    return feature, label

def pipeline_generator(input_features, input_labels, batch_size, image_size):
    data_length = len(input_features)
    while True:
        i = 0
        features = np.zeros((batch_size,image_size[1],image_size[0],3), dtype=np.uint8)
        labels = np.zeros((batch_size,image_size[1],image_size[0],1), dtype=np.float32)
        while i < batch_size:
            index = np.random.randint(data_length)
            feature = input_features[index]
            label = input_labels[index]
            feature, label = augmentation_pipeline(feature, label, image_size)    
            features[i] = feature
            labels[i] = label
            i += 1
        yield features, labels

def iou_better(actual, predicted):
    actual = K.abs(K.flatten(actual))
    predicted = K.abs(K.flatten(predicted))
    intersection = K.sum(actual * predicted)
    union = K.sum(actual) + K.sum(predicted) - intersection
    return intersection / union

def iou_simple(actual, predicted):
    actual = K.flatten(actual)
    predicted = K.flatten(predicted)
    return K.sum(actual * predicted) / (1.0 + K.sum(actual) + K.sum(predicted))

def val_loss(actual, predicted):
    return -iou_simple(actual, predicted)

features, labels = load_data(22065) 
model = models.encoder_decoder((160,240,3))
generator = pipeline_generator(features, labels, 1, (240,160))
checkpoint = ModelCheckpoint(filepath='model_weights.h5', verbose=True, save_best_only=True)
optimizer = Adam(lr=1.0e-5)

model.compile(optimizer=optimizer, loss=val_loss, metrics=[val_loss])
model.fit_generator(generator, samples_per_epoch=20000, nb_epoch=10, callbacks=[checkpoint])

