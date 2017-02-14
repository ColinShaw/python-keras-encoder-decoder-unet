import numpy as np
import cv2
import models

def load_image(file_path, image_size):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, image_size)

def save_image(image, segmented, file_path):
    segmented = np.array(segmented)
    segmented = np.array((segmented + 1.0) * 127.5, np.uint8)
    segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
    image = cv2.addWeighted(image, 1.0, segmented, 0.4, 1.0)
    cv2.imwrite(file_path, image)

def apply_segmentation(file_name):
    image = load_image(file_name + '.jpg', (240,160))
    image_ext = np.resize(image, (1,160,240,3))
    model = models.encoder_decoder((160,240,3))
    model.load_weights('model_weights_encoder_decoder.h5')
    segmented = model.predict(image_ext)
    save_image(image, segmented[0], file_name + '_out.jpg')

apply_segmentation('test-images/1');
