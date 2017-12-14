import os
import random

import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator

result_dimention = (28, 75, 1)
def toone(a):
    if a > 0:
        return 1.0
    else:
        return 0.0

def load_captchas(recaptcha_folder):
    train_images = []
    train_labels = []
    recaptcha_folder = os.path.join(recaptcha_folder, "train")
    for label_index, label in enumerate(os.listdir(recaptcha_folder)):
        for filename in os.listdir(os.path.join(recaptcha_folder, label)):
            if 'upper' in filename:
                img = scipy.misc.imread(os.path.join(recaptcha_folder, label, filename))
                img = 255 - img
                img = img /255.0
                vfunc = np.vectorize(toone)
                cog = ndimage.measurements.center_of_mass(vfunc(img))
                hl = 40
                img = img[int(cog[0])-hl:int(cog[0])+hl, int(cog[1])-hl:int(cog[1]) + hl]
                img = scipy.misc.imresize(img, [28,28])
                img = np.reshape(img, [28, 28, 1])
                train_images.append(img.astype('float32') / 255.0)
                train_labels.append(ord(label) - ord('A'))
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    y_labels = np.zeros((train_labels.shape[0], 26))
    y_labels[np.arange(train_labels.shape[0]), train_labels] = 1
    return train_images, y_labels

def captcha_generator(recaptcha_folder, number_letters, batch_size, is_test, **datagen_args):
    images, labels = load_captchas(recaptcha_folder)
    datagen = ImageDataGenerator(**datagen_args).flow(images, labels, batch_size=(number_letters * batch_size), shuffle=True, seed=0)
    while 1:
        x_batch, y_batch = next(datagen)
        batch_x = []
        images_per_batch = x_batch.reshape((-1, number_letters) + images[0].shape)
        for i in range(images_per_batch.shape[0]):
            img = np.zeros(result_dimention)
            for my_img in images_per_batch[i]:
                x, y = np.random.randint(0, img.shape[1] - my_img.shape[1] + 1), np.random.randint(0, img.shape[0] - my_img.shape[0] + 1)
                img[y:(y+my_img.shape[1]), x:(x+my_img.shape[0])] += my_img
            batch_x.append(np.clip(img, 0, 1))
        batch_x = np.array(batch_x)
        y_batch = y_batch.reshape((-1, number_letters, 26)).sum(axis=1)
        if is_test:
            yield [batch_x, y_batch], [y_batch, batch_x]
        else:
            yield batch_x, y_batch

def train_generator(recaptcha_folder, batch_size=100, training_with_two_letter=False, is_test=False, character_distance=-1):
    while 1:
        [x_batch, y_batch], [_,_2] = load_recaptcha_test(recaptcha_folder, training_size=batch_size, test_size=0, training_with_two_letter=training_with_two_letter, character_distance=character_distance)
        if is_test:
            yield [x_batch, y_batch], [y_batch, x_batch]
        else:
            yield x_batch, y_batch

if __name__ == "__main__":
    x, y = next(captcha_generator(os.path.join('..', 'recaptcha_capsnet_keras','recaptcha'), 2, 1, False))
    print(x.shape)
    plt.imsave("pic.jpg", x[0].reshape(28,75))
    print(y)
