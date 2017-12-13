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

def load_recaptcha_test(recaptcha_folder, training_size=10000, test_size=1000, training_with_two_letter=False):
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
        
        distance = random.randint(20,35)
        #print(cof1)

        new_img = np.concatenate((image1, np.zeros((image1.shape[0], int(result_dimention[1] - image1.shape[1]), result_dimention[2]))), axis=1)
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
        current_size += 1

    #TRAINING SET
    current_size = 0
    while training_with_two_letter and current_size < training_size:
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

        new_img = np.concatenate((image1, np.zeros((image1.shape[0], int(result_dimention[1] - image1.shape[1]), result_dimention[2]))), axis=1)
        image_start = int(cof1[1] + distance - cof2[1])
        width = image2.shape[1]
        new_img[:,image_start:image_start + width,0] = np.maximum(new_img[:,image_start:image_start + width,0],image2[:,:,0])

        #print(noexist)
        train_set.append(new_img)
        
        label = np.zeros(26)
        label[label_arg[i1]] = 1
        label[label_arg[i2]] = 1
        train_label.append(label)
        #print(label)
        #plt.imshow(new_img[:,:,0])
        #plt.show()
        current_size += 1
    return (np.array(train_set), np.array(train_label)), (np.array(test_set), np.array(test_label))


def train_generator(recaptcha_folder, batch_size=100, training_with_two_letter=False, is_test=False):
    while 1:
        [x_batch, y_batch], [_,_2] = load_recaptcha_test(recaptcha_folder, training_size=batch_size, test_size=0, training_with_two_letter=training_with_two_letter)
        yield ([x_batch, y_batch], [y_batch, x_batch])

if __name__ == "__main__":

    path = os.path.join('..', 'recaptcha_capsnet_keras','recaptcha')

    generator = train_generator(path, training_with_two_letter=True)
    for ([x,y],[y1,x1]) in generator:
        for i,image in enumerate(x):
            print(y[i])
            plt.imshow(image[:,:,0])
            plt.show()
        break
