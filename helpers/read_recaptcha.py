import os
import random

import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator

def toone(a):
    if a > 0:
        return 1.0
    else:
        return 0.0

def load_recaptcha_test(recaptcha_folder):
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
                #plt.imshow(img)
                #plt.show()
                #print(noexist)
                #print(img.shape)
                img = scipy.misc.imresize(img, [28,28])
                #plt.imshow(img)
                #plt.show()
                #print(noexist)
                img = np.reshape(img, [28, 28, 1])
                train_images.append(img.astype('float32'))
                train_labels.append(ord(label) - ord('A'))

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
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
        if times > 30:
            break

    test_size = 10 
    
    current_size = 0
    test_set = []
    test_label = []
    while current_size < test_size:
        i1 = random.randint(0,len(image_arg)-1)
        i2 = random.randint(0,len(image_arg)-1)
        
        if label_arg[i1] == label_arg[i2]:
          continue
        
        image1 = image_arg[i1]
        image2 = image_arg[i2]
        vfunc = np.vectorize(toone)
        cof1 = ndimage.measurements.center_of_mass(vfunc(image1[:,:,0]))
        cof2 = ndimage.measurements.center_of_mass(vfunc(image2[:,:,0]))
        
        distance = random.randint(20,35)
        #print(cof1)

        new_img = np.concatenate((image1, np.zeros((image1.shape[0], int(image1.shape[1]*1.4), image1.shape[2]))), axis=1)
        image_start = int(cof1[1] + distance - cof2[1])
        width = image2.shape[1]
        new_img[:,image_start:image_start + width,0] = np.maximum(new_img[:,image_start:image_start + width,0],image2[:,:,0])

        #print(noexist)
        test_set.append(new_img)
        
        label = np.zeros(26)
        label[label_arg[i1]] = 1
        label[label_arg[i2]] = 1
        test_label.append(label)
        #print(label)
        #plt.imshow(new_img[:,:,0])
        #plt.show()
    return test_set, test_label

if __name__ == "__main__":
    load_recaptcha_test(os.path.join('..', 'recaptcha_capsnet_keras','recaptcha'))
