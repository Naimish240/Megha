# --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# --------------------------------------------
random.seed(69)  # Because reproducibility matters
# --------------------------------------------
DATASET = "/home/cached/theCode/JupyterNotebooks/Clouds/Clouds/Dataset"
CATEGORIES = ["Ci", "Cs", "Cc", "Ac", "As", "Cu", "Cb", "Ns", "Sc", "St", "Ct"]
# Ci Cs Cc Ac As Cu Cb Ns Sc St Ct
# --------------------------------------------
def cropImg(x, y, arr):
    # Initialize an empty 300*300 list
    new_arr = [[0 for k in range(256)] for l in range(256)]
    # Yeet numbers into list for crop
    for i in range(x, x+256):
        for j in range(y, y+256):
            new_arr[x-i][y-j] = arr[i][j]
    return new_arr
# --------------------------------------------
dataset = []
# --------------------------------------------
for category in CATEGORIES:
    path = os.path.join(DATASET, category)
    img_cat = CATEGORIES.index(category)
    for image in os.listdir(path):
        # Load image in B/W
        img_arr = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        # Insert into the dataset
        dataset.append([img_arr, img_cat])
print("Loaded dataset")
# --------------------------------------------
random.shuffle(dataset)
# --------------------------------------------
# Set image dimensions
OLD_IMG_SIZE = 400
NEW_IMG_SIZE = 256

augmentable_count = {
    0  : 211,
    1  :  63,
    2  :  82,
    3  : 129,
    4  : 162,
    5  : 168,
    6  : 108,
    7  :  76,
    8  :  10,
    9  : 148,
    10 : 150
}
# --------------------------------------------
aug_data = []
index_errors = 0
# --------------------------------------------
print("Augmenting Dataset")
# Augments the images
for img_arr, img_cat in dataset:
    try:
        if augmentable_count[img_cat] > 0:
            # Reduce count by 1, because we don't want infinite loops
            augmentable_count[img_cat] -= 1

            # Random number to figure out where we'll be cropping from or shrinking
            p = random.random()
            try:
                # Shrink Image (20%, since 5 cases)
                if p < (1/5):
                    new_img = cv2.resize(img_arr, (NEW_IMG_SIZE, NEW_IMG_SIZE))
                    aug_data.append([new_img, img_cat])

                # Crop from top (80 %, since it accounts for L and R (both top and bottom) )
                else:
                    x = random.randint(0, 144)
                    y = random.randint(0, 144)
                    new_img = cropImg(x, y, img_arr)
                    aug_data.append([new_img, img_cat])
                    
            # Image size is 256*256, not 400*400
            except IndexError:
                augmentable_count[img_cat] += 1
                index_errors += 1
    except KeyError:
        print("Key Error")
        print(augmentable_count)
        print(img_cat)
        plt.imshow(img_arr, cmap='gray')
        plt.show()
        print('---------------------------')
print("Dataset Augmented")
# --------------------------------------------
reshaped_data = []
# --------------------------------------------
# Convert images in Original into 256*256
for img_arr, cat_arr in dataset:
    new_img = cv2.resize(img_arr, (NEW_IMG_SIZE, NEW_IMG_SIZE))
    reshaped_data.append([new_img, img_cat])
# --------------------------------------------
print("Datasets Merged")
# Join Original and Augmented together
training_data = reshaped_data + aug_data
print(len(training_data))
# --------------------------------------------
X = []
y = []
# --------------------------------------------
for features, label in training_data:
    X.append(features)
    foo = [0 for i in range(11)]
    foo[label] = 1
    y.append(foo)
# --------------------------------------------
X = np.array(X).reshape(-1, NEW_IMG_SIZE, NEW_IMG_SIZE, 1)
print("Dataset converted to Numpy array")
# --------------------------------------------
# Create the Model
num_classes = 11

model = keras.Sequential()

# Add preprocessing layer to convert from [0:255] to [0:1]
model.add(layers.experimental.preprocessing.Rescaling(
    1./255, 
    input_shape=(X.shape[1:])
        ))

# Add First Conv Layer, from Alex
# 256*256*1 to 63*63*64
model.add(layers.Conv2D(
    filters=64, 
    kernel_size=11, 
    strides=(4,4), 
    activation='relu')
         )

# Do some BatchNorm
model.add(keras.layers.BatchNormalization())
# Throw in a Max Pooling layer
# 63*63*64 to 31*31*64
model.add(layers.MaxPooling2D(
    pool_size=(3,3), strides=(2,2)
))

# Add second Conv Layer
# 31*31*64 to 31*31*256
model.add(layers.Conv2D(
    filters=224, 
    kernel_size=(5,5), 
    strides=(1,1), 
    activation='relu', 
    padding="same"
         ))
          
# Do some BatchNorm
model.add(keras.layers.BatchNormalization())
# Throw in a Max Pooling layer
# 31*31*224 to 15*15*224
model.add(layers.MaxPooling2D(
    pool_size=(3,3), strides=(2,2)
))

# Add another Conv2D layer
# 15*15*224 to 15*15*96
model.add(layers.Conv2D(
    filters=96,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu',
    padding='same'
))

# Batch Norm is the norm
model.add(keras.layers.BatchNormalization())

# Add another Conv2D layer
# 15*15*96 to 15*15*32
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu',
    padding='same'
))

# MaxPool layer
# 15*15*32 to 7*7*32
model.add(layers.MaxPooling2D(
    pool_size=(3,3), strides=(2,2)
))

# Time to go Flat
# 7*7*32 to 1568*1
model.add(layers.Flatten())

# Add a Dense layer
model.add(layers.Dense(768, 'relu'))
# Dropout
model.add(layers.Dropout(0.4))
# Another Dense Layer
model.add(layers.Dense(320, 'relu'))
# One Final Dropout
model.add(layers.Dropout(0.5))
# Output Layer
model.add(layers.Dense(11, 'softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model compiled")
print(model.summary())
# --------------------------------------------
Z = []
for i in X:
    Z.append(tf.convert_to_tensor(i, dtype=tf.float32))
X = Z
print("Numpy to tensor converted")
# --------------------------------------------
# Train
epochs = 5
history = model.fit(X, y, batch_size=16, validation_split=0.2)
# --------------------------------------------
# Analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# --------------------------------------------
