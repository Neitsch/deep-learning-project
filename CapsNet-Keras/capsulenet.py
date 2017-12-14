"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 50
       python CapsNet.py --epochs 50 --num_routing 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras import metrics
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def new_top_two_fun(y_true, y_pred, num_correct=2):
    top_pred = np.argsort(y_pred, axis=1)[:,-num_correct:]
    top_true = np.argsort(y_true, axis=1)[:,-num_correct:]
    zipped = zip(top_pred, top_true)
    intersect_val = [np.intersect1d(x, y) for (x, y) in zipped]
    intersect_sizes = [x.size for x in intersect_val]
    intersect_sum = sum(intersect_sizes)
    per_batch = intersect_sum / y_pred.shape[0]
    return per_batch

def top_two_fun(y_true, y_pred):
    pred_exp = np.exp(y_pred)
    acc = np.sum(np.multiply(y_true, pred_exp / np.sum(pred_exp, axis=0))) / y_true.shape[0]
    return acc

def train(model, data, eval_model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    (_x_train, _y_train), (x_test2, y_test2), ordering = load_mnist(0)

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    #in_top_two = lambda x, y: metrics.top_k_categorical_acuracy(x, y, k=2)
    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy', 'twocategory':top_two_fun})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, ordering.shape)
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    test_callback = TestCallback(eval_model, (x_test2, y_test2, ordering))
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[test_callback, log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model

class TestCallback(Callback):
    def __init__(self, model, test_data):
        self.my_model = model
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x_test, y_test, ordering = self.test_data
        #metrics.evaluate(x_test, y_test)
        #print(metrics) 
        scan = scan_accuracy(self.my_model, (x_test, y_test), ordering)
        #y_pred, x_recon = self.my_model.predict(x_test)
        #loss = new_top_two_fun(y_test, y_pred, num_correct=2)
        #print('\nTesting loss: {}, acc: skipped000\n'.format(loss))
        logs.update(scan)
        #logs['our_acc'] = np.float64(loss)
        return logs

def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    print('Test loss:', margin_loss(y_test, y_pred))
    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    #plt.imshow(plt.imread("real_and_recon.png", ))
    #plt.show()


def load_mnist(dataset=0):
    if dataset == 0:
        import multi_mnist_setup
        #create_single_mnist
        #create_rand_single_mnist
        #create_rand_multi_mnist
        x_train, y_train = multi_mnist_setup.create_rand_single_mnist(samples=1000)
        x_test, y_test, orders = multi_mnist_setup.create_rand_multi_mnist(samples=1000, dataset="testing")
        x_train = x_train.reshape(-1, 28, 112, 1).astype('float32') / 255.
        x_test = x_test.reshape(-1, 28, 112, 1).astype('float32') / 255.
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        return (x_train, y_train), (x_test, y_test), orders
    elif dataset ==1:
        # the data, shuffled and split between train and test sets
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
        y_train = to_categorical(y_train.astype('float32'))
        y_test = to_categorical(y_test.astype('float32'))
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        return (x_train, y_train), (x_test, y_test)
    else:
        
        from fashion import load_fashion_mnist
        x_train, y_train, _, x_test, y_test, _ = load_fashion_mnist(100, True)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    
        return (x_train, y_train), (x_test, y_test)

def scan_accuracy(model, data, true_orders, base_size=28, num_chars = 2):
    (test_x, test_y) = data
    #print(test_x.shape)
    #print(test_y.shape)
    #print(test_y)
    all_preds = []
    for i in range(0, test_x.shape[2]-base_size, 5):
        sub_imgs = test_x[:,:,i:i+28,:]
        #print(sub_imgs.shape)
        y_pred, x_recon = model.predict(sub_imgs)
        all_preds.append(y_pred)
        
        #max_pos = np.max(y_pred,0)
        #print(max_pos)
        #if y_pred[:, max_pos] > 0.8:
        #    print(max_pos, end='')
    all_preds = np.array(all_preds)
    #print(all_preds.shape)
    maxes = np.amax(all_preds, axis=0)
    sorts = np.argsort(maxes, axis=1)[:,-num_chars:]
    #print(sorts)
    ordered = [sorted(b_n, key=lambda v: np.argmax(all_preds[:,i,v])) for i, b_n in enumerate(sorts)]
    #print("Character Accuracy: ", np.sum(ordered == true_orders) / true_orders.size)
    #print("CAPTCHA Accuracy: ", np.sum(np.all(ordered == true_orders, axis=1)) / true_orders.shape[0])
    #print(ordered)
    #print(true_orders)
    #print(ordered.shape)
    return {
        'character_acc': np.float64(np.sum(ordered == true_orders) / true_orders.size),
        'captcha_acc': np.float64(np.sum(np.all(ordered == true_orders, axis=1)) / true_orders.shape[0])
    }


if __name__ == "__main__":
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.0, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--scan_imgs', default=None, type=int)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist(1)

    # define model
    model, eval_model = CapsNet(input_shape=(28, 28, 1),
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                num_routing=args.num_routing)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), eval_model=eval_model, args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        else:
            if args.scan_imgs is None:
                test(model=eval_model, data=(x_test, y_test))
            else:
                (multi_x_train, multi_y_train), (multi_x_test, multi_y_test), orders = load_mnist(0)
                scan_accuracy(eval_model, data=(multi_x_test, multi_y_test), true_orders=orders)
