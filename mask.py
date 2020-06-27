import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

from IPython.display import display

def plot_pair(images, gray=False):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    i=0

    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i+=1

    plt.show()

def get_poly(ann_path):

    with open(ann_path) as handle:
        data = json.load(handle)

    shape_dicts = data['shapes']

    return shape_dicts
def create_binary_masks(im, shape_dicts):

    blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)

    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(blank, [points], 255)

    return blank


image_list = sorted(os.listdir('images'), key=lambda x: int(x.split('.')[0]))
annot_list = sorted(os.listdir('annotated'), key=lambda x: int(x.split('.')[0]))

for im_fn, ann_fn in zip(image_list, annot_list):

    im = cv2.imread(os.path.join('images', im_fn), 0)

    ann_path = os.path.join('annotated', ann_fn)
    shape_dicts = get_poly(ann_path)
    im_binary = create_binary_masks(im, shape_dicts)

    plot_pair([im, im_binary], gray=True)
    plt.show()
    break


hues = {'road': 30,
        'sidewalk': 0,
        'sidewalk_red': 90,
        'barrier': 60}

labels = sorted(hues.keys())
print(labels)



def create_multi_masks(im, shape_dicts):

    channels = []
    cls = [x['label'] for x in shape_dicts]
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
    label2poly = dict(zip(cls, poly))
    background = np.zeros(shape=im.shape, dtype=np.float32)

    for i, label in enumerate(labels):

        blank = np.zeros(shape=im.shape, dtype=np.float32)

        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], 255)
            cv2.fillPoly(background, [label2poly[label]], 255)

        channels.append(blank)
    _, thresh = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
    channels.append(thresh)

    Y = np.stack(channels, axis=2)
    return Y

for im_fn, ann_fn in zip(image_list, annot_list):

    im = cv2.imread(os.path.join('images', im_fn), 0)
    ann_path = os.path.join('annotated', ann_fn)
    shape_dicts = get_poly(ann_path)
    Y = create_multi_masks(im, shape_dicts)
    break

for i in range(5):
    mask = Y[:,:,i]
    plt.imshow(mask, cmap='gray')
    plt.show()

def draw_multi_masks(im, shape_dicts):

    blank = np.zeros(shape=im.shape, dtype=np.uint8)

    channels = []
    cls = [x['label'] for x in shape_dicts]
    poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
    label2poly = dict(zip(cls, poly))

    for i, label in enumerate(labels):

        if label in cls:
            cv2.fillPoly(blank, [label2poly[label]], (hues[label], 255, 255))

    return cv2.cvtColor(blank, cv2.COLOR_HSV2RGB)

for i, (im_fn, ann_fn) in enumerate(zip(image_list, annot_list)):

    im = cv2.imread(os.path.join('images', im_fn), 1)
    ann_path = os.path.join('annotated', ann_fn)
    shape_dicts = get_poly(ann_path)
    im_color = draw_multi_masks(im, shape_dicts)
    display(Image.fromarray(im_color))
    if i == 4:
        break
