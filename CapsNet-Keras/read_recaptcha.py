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

def read_images_from_disk(recaptcha_folder):

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

    return np.array(train_images), np.array(train_labels)

def load_recaptcha_test(base_images, base_labels, training_size=10000, test_size=1000, training_with_two_letter=False, character_distance=-1):

    train_images = base_images
    train_labels = base_labels
    
    datagenArgs = {
        'shear_range': 0.5,
        'rotation_range': 15,
    }
    datagen = ImageDataGenerator( **datagenArgs)
    
    times = 0
    image_arg = []
    label_arg = []
    for x_batch, y_batch in datagen.flow(train_images, train_labels,  batch_size=len(train_labels),  shuffle=True, seed=0):
        for x in x_batch:
            image_arg.append(x/255.0)
        for y in y_batch:
            label_arg.append(y)
        times += 1
        if times > training_size/50:
            break

    
    # TRAINING
    train_set = []
    train_label = []
    if not training_with_two_letter:
        for i, img in enumerate(image_arg):

            if i >= training_size:
                continue
            final_image = np.zeros(result_dimention)
            start_index = random.randint(0, result_dimention[1] - img.shape[1])
            width = img.shape[1]
            final_image[:,start_index:start_index + width,0] = img[:,:,0]
            train_set.append(final_image)

            label = np.zeros(26)
            label[label_arg[i]] = 1
            train_label.append(label)
            #print(label)
            #plt.imshow(final_image[:,:,0])
            #plt.show()

    # TEST 
    current_size = 0
    test_set = []
    test_label = []
    while current_size < test_size :
        i1 = random.randint(0,len(image_arg)-1)
        i2 = random.randint(0,len(image_arg)-1)
        
        if label_arg[i1] == label_arg[i2]:
          continue
        
        image1 = image_arg[i1]
        image2 = image_arg[i2]
        vfunc = np.vectorize(toone)
        cof1 = ndimage.measurements.center_of_mass(vfunc(image1[:,:,0]))
        cof2 = ndimage.measurements.center_of_mass(vfunc(image2[:,:,0]))
        
        if character_distance < 0:
            distance = random.randint(20,35)
        else:
            distance = character_distance

        #print(cof1)

        new_img = np.concatenate((image1, np.zeros((image1.shape[0], int(result_dimention[1] - image1.shape[1]), result_dimention[2]))), axis=1)
        image_start = int(cof1[1] + distance - cof2[1])
        width = image2.shape[1]
        new_img[:,image_start:image_start + width,0] = np.maximum(new_img[:,image_start:image_start + width,0],image2[:,:,0])


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

class train_generator(object):
    def __init__(self, recaptcha_folder, batch_size=100, training_with_two_letter=False, is_test=False, character_distance=-1):
        self.batch_size = batch_size
        self.training_with_two_letter= training_with_two_letter
        self.is_test=is_test
        self.character_distance = character_distance
        self.base_images, self.base_labels = read_images_from_disk(recaptcha_folder)
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def next(self):
        [x_batch, y_batch], [_,_2] = load_recaptcha_test(self.base_images, self.base_labels, training_size=self.batch_size, test_size=0, training_with_two_letter=self.training_with_two_letter, character_distance=self.character_distance)
        if self.is_test:
            return [x_batch, y_batch], [y_batch, x_batch]
        else:
            return x_batch, y_batch



if __name__ == "__main__":
    """
    x, y = next(captcha_generator(os.path.join('..', 'recaptcha_capsnet_keras','recaptcha'), 2, 1, False))
    print(x.shape)
    plt.imsave("pic.jpg", x[0].reshape(28,75))
    print(y)
    """

    path = os.path.join('..', 'recaptcha_capsnet_keras','recaptcha')

    generator = train_generator(path, training_with_two_letter=True, character_distance=30, is_test=True)
    for [x,y],[y1,x1] in generator:
        for i,image in enumerate(x):
            print(y[i])
            plt.imshow(image[:,:,0])
            plt.show()
        break

