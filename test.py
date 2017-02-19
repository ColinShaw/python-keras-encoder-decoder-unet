import numpy as np
import cv2
import models

model = models.encoder_decoder((160,240,3))
model.load_weights('model_weights.h5')

image = cv2.imread('test-images/1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (240,160))
image_ext = np.resize(image, (1,160,240,3))

segmented = model.predict(image_ext)
segmented = np.array(segmented[0])
segmented = np.array((segmented + 1.0) * 127.5, np.uint8)

_, segmented = cv2.threshold(segmented, 200, 255, cv2.THRESH_BINARY)
segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
segmented[:,:,0:1] = 0
image = cv2.addWeighted(segmented, 0.5, image, 0.5, 0.0)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imwrite('test-images/1_out.jpg', image)

