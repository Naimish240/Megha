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
print("Datasets Merged")
# Join Original and Augmented together
training_data = reshaped_data + aug_data
print(len(training_data))
# ----------------------------------------
# --------------------------------------------
X = np.array(X).reshape(-1, NEW_IMG_SIZE, NEW_IMG_SIZE, 1)
print("Dataset converted to Numpy array")