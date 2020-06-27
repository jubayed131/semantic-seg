import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os
import time
import contextlib
import numpy as np
import cv2
from utils import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from models import preprocess_input, dice
from config import imshape, model_name, n_classes
from utils import add_masks, crf
with contextlib.redirect_stdout(None):
    import pygame

RUN = True
MODE = 'softmax'
CALC_CRF = False
BACKGROUND = False

frame_shape = (640, 480)
target_shape = imshape[:2]
d_width = target_shape[0] // 2
d_height = target_shape[1] // 2
x0 = 0
y0 = 0
x1 = imshape[0]
y1 = imshape[1]


model = load_model(os.path.join('models', model_name+'.model'),
                   custom_objects={'dice': dice})
frame = cv2.imread('test_30_22.png',cv2.IMREAD_COLOR) ### Image read vs need to be changed
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow("OpenCV Image Reading", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(frame)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
im = frame.copy()

roi = im[x0:x1, y0:y1]
tmp = np.expand_dims(roi, axis=0)
roi_pred = model.predict(tmp)

roi_mask = add_masks(roi_pred.squeeze()*255.0)
frame[x0:x1, y0:y1] = roi_mask
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow("Predicted", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
