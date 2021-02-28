# --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
# --------------------------------------------
random.seed(69)  # Because reproducibility matters
# --------------------------------------------
DATASET = "/home/cached/theCode/JupyterNotebooks/Clouds/Clouds/Dataset"
CATEGORIES = ["Ci", "Cs", "Cc", "Ac", "As", "Cu", "Cb", "Ns", "Sc", "St", "Ct"]
# Ci Cs Cc Ac As Cu Cb Ns Sc St Ct
# --------------------------------------------
# --------------------------------------------
dataset = []
# --------------------------------------------
# --------------------------------------------
random.shuffle(dataset)
# --------------------------------------------
# Set image dimensions
OLD_IMG_SIZE = 400
NEW_IMG_SIZE = 256


# --------------------------------------------
aug_data = []
index_errors = 0
# --------------------------------------------
print("Augmenting Dataset")

print("Dataset Augmented")
# --------------------------------------------
reshaped_data = []
# --------------------------------------------
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
Z = []
for i in X:
    Z.append(tf.convert_to_tensor(i, dtype=tf.float32))
X = Z
print("Numpy to tensor converted")
# --------------------------------------------
