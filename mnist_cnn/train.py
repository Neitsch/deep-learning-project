'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
augParams = {
        'rotation_range': [30,34,60,75,90],
        'width_shift_range' : [0.2, 0.5],
        'zoom_range' : [0.5,1.0],
        'shear_range' : [0.5,1.0],
        }
with open('output.csv', 'w', newline='') as csvfile:
    import csv
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['epoch', 'aug_type', 'aug_value', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])
    for i in range(12):
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  validation_data=(x_test, y_test))
        train_score = model.evaluate(x_train, y_train)
        score = model.evaluate(x_test, y_test, verbose=0)

        row = [i, '', '', train_score[0], train_score[1], score[0], score[1]]
        csvwriter.writerow(row)

        for augType, augValues in augParams.items():
            for augValue in augValues:
                datagenArgs = {
                    augType: augValue,
                }
                print('augumenting data using following args {}'.format(datagenArgs))
                datagen = ImageDataGenerator(**datagenArgs)
                aug_x_test = []
                aug_y_test = []
                for x_batch, y_batch in datagen.flow(x_test, y_test, batch_size=len(y_test), shuffle=False, seed=0):
                    aug_x_test = x_batch
                    aug_y_test = y_batch
                    break
                score = model.evaluate(aug_x_test, aug_y_test, verbose=0)
                csvwriter.writerow([i, augType, augValue, train_score[0], train_score[1], score[0], score[1]])

        print('')
        print('Train loss:', train_score[0])
        print('Train accuracy' , train_score[1])
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
