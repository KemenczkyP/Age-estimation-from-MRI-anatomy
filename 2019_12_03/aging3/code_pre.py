from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
#%%
fashion_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0

test_images = test_images / 255.0

'''
train_images = np.zeros((10000,28,28))
test_images = np.zeros((10000,28,28))
train_labels = []
test_labels = []

maxnum = 28*28
for idx in range(10000):
    noise = int(np.random.normal(0,3))
    if(noise<0):
        zer = -noise
    else:
        zer = 0
    numofpixels1 = np.random.randint(zer, maxnum)
    numofpixels2 = np.random.randint(zer, maxnum)
    x1 = np.random.randint(0,28, size = (numofpixels1 + noise, 2))
    train_labels.append(numofpixels1)
    test_labels.append(numofpixels2)
    for jdx in range(x1.shape[0]):
        train_images[idx,x1[jdx,0],x1[jdx,1]] = 1
    x2 = np.random.randint(0,28, size = (numofpixels2 + noise, 2))
    for jdx in range(x2.shape[0]):
        test_images[idx,x2[jdx,0],x2[jdx,1]] = 1
del x1, x2, idx,jdx, maxnum, numofpixels1,numofpixels2
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
'''
        
#%%


model = keras.Sequential([
        keras.layers.Reshape((28,28,1)),
        keras.layers.Conv2D(16,
                            kernel_size = 3,
                            padding = 'same',
                            activation = tf.nn.leaky_relu
                            ),
        keras.layers.MaxPool2D(padding = 'same'),
        keras.layers.Conv2D(32,
                            kernel_size = 3,
                            padding = 'same',
                            activation = tf.nn.leaky_relu
                            ),
        keras.layers.MaxPool2D(padding = 'same'),
        keras.layers.Conv2D(64,
                            kernel_size = 3,
                            padding = 'same',
                            activation = tf.nn.leaky_relu
                            ),
    keras.layers.Flatten(input_shape=(7, 7)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])
        
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

#%%
a = model.predict(test_images[:1,:,:])
plt.imshow(test_images[0,:,:])

t = model.trainable_variables

for idx in range(len(t)):
    np.save("model_w_{}".format(idx), t[idx].numpy())
