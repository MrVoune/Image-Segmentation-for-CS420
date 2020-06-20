import tensorflow as tf
import pathlib
import numpy as np
import pandas as pd
import segmentation_models as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from tensorflow.keras import layers

def compute_acc(img, label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if(img[i][j] == label[i][j]):
                if(label[i][j] == 0):
                    TN = TN + 1
                else:
                    TP = TP + 1
            else:
                if(label[i][j] == 0):
                    FP = FP + 1
                else:
                    FN = FN + 1
    return TN, TP, FP, FN

reshape = 192

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_root = pathlib.Path('.\\dataset\\new_train_set\\train_img')
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]

train_data = np.empty((25, reshape, reshape, 1))
for i in range(25):
    img_raw = tf.io.read_file(all_image_paths[i])
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [reshape, reshape])
    img_final = img_final/255.0
    train_data[i] = img_final

label_root = pathlib.Path('.\\dataset\\new_train_set\\train_label')
label_path = list(label_root.glob('*'))
label_path = [str(path) for path in label_path]

train_label = np.empty((25, reshape, reshape, 1))
for i in range(25):
    img_raw = tf.io.read_file(label_path[i])
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [reshape, reshape])
    img_final = img_final/255.0
    train_label[i] = img_final

test_img_root = pathlib.Path('.\\dataset\\new_test_set\\test_img')
test_img_path = list(test_img_root.glob('*'))
test_img_path = [str(path) for path in test_img_path]
test_img = np.empty((5, reshape, reshape, 1))
for i in range(5):
    img_raw = tf.io.read_file(test_img_path[i])
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [reshape, reshape])
    img_final = img_final/255.0
    test_img[i] = img_final

test_label_root = pathlib.Path('.\\dataset\\new_test_set\\test_label')
test_label_path = list(test_label_root.glob('*'))
test_label_path = [str(path) for path in test_label_path]
test_label = np.empty((5, reshape, reshape, 1))
for i in range(5):
    img_raw = tf.io.read_file(test_label_path[i])
    img_tensor = tf.image.decode_image(img_raw)
    img_final = tf.image.resize(img_tensor, [reshape, reshape])
    img_final = img_final/255.0
    test_label[i] = img_final


BACKBONE = 'resnet50'

# define model
model = sm.PSPNet(BACKBONE, encoder_weights=None, input_shape=(reshape, reshape, 1), classes=1)
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# fit model
model.fit(
   x=train_data[:20],
   y=train_label[:20],
   batch_size=16,
   epochs=8,
   validation_data=(train_data[20:], train_label[20:])
)

result = model.predict(test_img)
# print(result.shape)
# print(result[1].shape)
# img_tensor = np.squeeze(result[1])
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(img_tensor)
# plt.axis('off')
# plt.show()

acc = 0
all_TN = 0
all_TP = 0
all_FP = 0
all_FN = 0
for i in range(5):
    TN, TP, FP, FN = compute_acc(np.squeeze(result[i]), np.squeeze(test_label[i]))
    all_TN = all_TN + TN
    all_TP = all_TP + TP
    all_FP = all_FP + FP
    all_FN = all_FN + FN
acc = (all_TP + all_TN)/ (all_TP + all_FN + all_FP + all_TN)

print(acc)