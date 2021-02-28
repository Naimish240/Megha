import os
import cv2
import random
import matplotlib.pyplot as plt

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

def cropImg(x, y, arr, ARR_SIZE):
    """
    This function crops the input image

    Args:
        x (int): x-coord to start crop from 
        y (int): y-coord to start the crop from
        arr (np.array) : input image to crop into
        ARR_SIZE (int): size to crop to

    Returns:
        new_arr : the cropped image
    """
    # Initialize an empty n*n list
    new_arr = [[0 for k in range(ARR_SIZE)] for l in range(ARR_SIZE)]
    # Yeet numbers into list for crop
    for i in range(x, x+ARR_SIZE):
        for j in range(y, y+ARR_SIZE):
            new_arr[x-i][y-j] = arr[i][j]
    return new_arr

def loadRawData(CATEGORIES, DATASET):
    """
    Loads in the dataset

    Args:
        CATEGORIES (list): List of all folder names (classes)
        DATASET (string): Path to folder which has all classes in the dataset

    Returns:
        dataset : list of elements, where [image, category];
                                image    : np array
                                category : int 
    """
    dataset = []
    for category in CATEGORIES:
        path = os.path.join(DATASET, category)
        img_cat = CATEGORIES.index(category)
        for image in os.listdir(path):
            # Load image in B/W
            img_arr = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            # Insert into the dataset
            dataset.append([img_arr, img_cat])
    return dataset

def augmentImages(dataset, OLD_IMG_SIZE, NEW_IMG_SIZE):
    """
    Augments the images through cropping and shrinking

    Args:
        dataset (list): list of elements, [image, category]; category -> int
        OLD_IMG_SIZE (int): dimensions of the original image
        NEW_IMG_SIZE (int): required dimensions of the final image

    Returns:
        aug_data (list) : list of all images after augmentation; [image, category];
                                                    image    : np array
                                                    category : int
    """
    global augmentable_count
    aug_data = []
    index_errors = 0
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

                    # Crop from top (80 %, since it accounts for L and R (both top and bottom))
                    else:
                        x = random.randint(0, OLD_IMG_SIZE-NEW_IMG_SIZE)
                        y = random.randint(0, OLD_IMG_SIZE-NEW_IMG_SIZE)
                        new_img = cropImg(x, y, img_arr, NEW_IMG_SIZE)
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
    
    return aug_data

def resizeOriginalImages(dataset, NEW_IMG_SIZE):
    """
    Resizes all images in the original dataset to NEW_IMG_SIZE

    Args:
        dataset (list): list of elements, [image, category]; category -> int
        NEW_IMG_SIZE (int): new image dimensions

    Returns:
        [type]: List of elements, [image, category]
                                    image    : np array
                                    category : int
    """
    reshaped_data = []
    # Convert images in Original into 256*256
    for img_arr, cat_arr in dataset:
        new_img = cv2.resize(img_arr, (NEW_IMG_SIZE, NEW_IMG_SIZE))
        reshaped_data.append([new_img, img_cat])
    return reshaped_data

def saveImages(dataset, folder):
    """
    Writes all of the augmented images into their seperate folders

    Args:
        dataset ([type]): [description]
        folder ([type]): path to folder to save images into
    """
    # Index, to make writing filenames easier
    index = 0
    for img, cat in dataset:
        # Checks if the folder exists, and creates it if it doesn;t
        if not os.path.exsist("{}/{}".format(folder, cat)):
            os.mkdir("{}/{}".format(folder, cat))
        # Writes image into folder
        path = "{}/{}/{}.jpg".format(folder, cat, index)
        cv2.imwrite(path, img)
        # Increase index, for obv reasons
        index += 1

def loadProcessedImages(folder):
    """
    Loads the augmented and processed dataset for training
    Args:
        folder (str): path to the augmented data folder

    Returns:
        X : List of all images (input vector)
        Y : List of all output vectors
    """
    # Lists for Input and Output
    X = []
    Y = []
    # Loops through the processed images
    for i in range(11):
        path = "{}/{}".format(folder, i)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            # Add input image to X
            X.append(img)
            vec = [0] * 11
            vec[i] = 1
            # Add Output Vector to Y
            Y.append(vec)

    return X, Y