from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, GlobalAveragePooling3D, Concatenate, Reshape, multiply
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow.keras.optimizers as KOP
from numpy import array
import mrcfile
from tensorflow.keras import layers

root = '../input/train0005/train_005'
folders = os.listdir(root)
X = []
Y = []
names = {}
ptr = 0

for folder in folders:
    #print(folder)
    names[ptr] = folder
    # print(names)
    files = os.listdir(os.path.join(root, folder))
    for file in files:
        image_path = os.path.join(os.path.join(root, folder, file))
        with mrcfile.open(image_path) as mrc:
            img = mrc.data
        # X.append(img.reshape(img.shape[0], -1))
        X.append(img)
        Y.append(ptr)
    ptr += 1


for i in X:
    if i.shape == (32, 30, 32) or i.shape == (31, 32, 32):
        b = X.index(i)
        Y.pop(b)
        X.remove(i)

Y = to_categorical(Y)

X = np.array(X)
print(X.shape)
Y = np.array(Y)
print(Y.shape)
# X = X.reshape(X.shape[0], -1)
print(X[0].shape)

# X = X.reshape(X.shape[0], -1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, shuffle=True, test_size=0.1, random_state=123)

X_train = X_train.reshape((-1, 32, 32, 32, 1))
X_test = X_test.reshape((-1, 32, 32, 32, 1))

print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)
sample_shape = (32, 32, 32, 1)

inputs = keras.Input(shape=sample_shape)


def SE(i):

    R = i.shape[4]
    se_shape = (1, 1, 1, R)
    print(R)
    r = R//12
    x = GlobalAveragePooling3D()(i)
    print(x.shape)
    x = Reshape(se_shape)(x)
    x = layers.Conv3D(filters=r, kernel_size=(1, 1, 1), activation="relu")(x)
    print(x.shape)
    x = layers.Conv3D(filters=R, kernel_size=(1, 1, 1), activation="sigmoid")(x)
    print(x.shape)
    output = multiply([i, x])
    print(output.shape)
    return output


x1 = layers.Conv3D(filters=16, kernel_size=(3, 3, 3), activation="relu")(inputs)
#x1 = layers.MaxPool3D(pool_size=2)(x1)
x1 = layers.BatchNormalization()(x1)
x1 = SE(x1)
x1 = SE(x1)
x1 = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu")(inputs)
x1 = layers.MaxPool3D(pool_size=2)(x1)
x1 = layers.BatchNormalization()(x1)
x1 = SE(x1)
x1 = SE(x1)
x1 = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x1)
#x1 = layers.MaxPool3D(pool_size=2)(x1)
x1 = layers.BatchNormalization()(x1)
x1 = SE(x1)
x1 = SE(x1)
x1 = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x1)
x1 = layers.MaxPool3D(pool_size=2)(x1)
x1 = layers.BatchNormalization()(x1)
x1 = SE(x1)
x1 = SE(x1)
x1 = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x1)
# x = layers.MaxPool3D(pool_size=2)(x)
x1 = layers.BatchNormalization()(x1)
x1 = SE(x1)
x1 = SE(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(units=1024, activation="relu")(x1)
x1 = layers.Dropout(0.3)(x1)
x1 = layers.Dense(units=1024, activation="relu")(x1)
x1 = layers.Dropout(0.3)(x1)
output = layers.Dense(units=10, activation='softmax')(x1)
model = tf.keras.Model(inputs=inputs, outputs=output)
print(model.summary())

kop = KOP.SGD(lr=0.004, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(optimizer=kop, loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(X_train, Y_train, batch_size=50, epochs=22)


(eval_loss, eval_accuracy) = model.evaluate(X_test, Y_test, batch_size=50, verbose=1)
print(eval_accuracy * 100)

