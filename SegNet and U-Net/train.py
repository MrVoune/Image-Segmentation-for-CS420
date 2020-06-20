from keras.callbacks import ModelCheckpoint, TensorBoard

import LoadBatches
from Models import SegNet, UNet
from keras import optimizers
import math

#############################################################################
train_images_path = "dataset/new_train_set/train_img/"
train_segs_path = "dataset/new_train_set/train_label/"
train_batch_size = 5
n_classes = 2

epochs = 10

input_height = 512
input_width = 512


val_images_path = "dataset/new_test_set/test_img/"
val_segs_path = "dataset/new_test_set/test_label/"
val_batch_size = 5

key = "segnet"


##################################

method = {'segnet': SegNet.SegNet,}

m = method[key](n_classes, input_height=input_height, input_width=input_width)
m.compile(
    loss='categorical_crossentropy',
    optimizer="adadelta",
    metrics=['acc'])

G = LoadBatches.imageSegmentationGenerator(train_images_path,
                                           train_segs_path, train_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

G_test = LoadBatches.imageSegmentationGenerator(val_images_path,
                                                val_segs_path, val_batch_size, n_classes=n_classes, input_height=input_height, input_width=input_width)

checkpoint = ModelCheckpoint(
    filepath="output/%s_model.h5" %
    key,
    monitor='acc',
    mode='auto',
    save_best_only='True')
tensorboard = TensorBoard(log_dir='output/log_%s_model' % key)

m.fit_generator(generator=G,
                steps_per_epoch=math.ceil(367. / train_batch_size),
                epochs=epochs, callbacks=[checkpoint, tensorboard],
                verbose=2,
                validation_data=G_test,
                validation_steps=8,
                shuffle=True)
