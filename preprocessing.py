import numpy as np
from os import listdir, mkdir
from os.path import join, exists
from cv2 import imread, imwrite, IMREAD_GRAYSCALE, resize
from random import random, randint, seed
from argparse import ArgumentParser
from matplotlib.pyplot import imshow, show

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
        path = join(DATASET, category)
        img_cat = CATEGORIES.index(category)
        for image in listdir(path):
            # Load image in B/W
            img_arr = imread(join(path, image), IMREAD_GRAYSCALE)
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
    # Augments the images with seed, for reproducability
    seed(69)
    for img_arr, img_cat in dataset:
        try:
            if augmentable_count[img_cat] > 0:
                # Reduce count by 1, because we don't want infinite loops
                augmentable_count[img_cat] -= 1

                # Random number to figure out where we'll be cropping from or shrinking
                p = random()
                try:
                    # Shrink Image (20%, since 5 cases)
                    if p < (1/5):
                        new_img = resize(img_arr, (NEW_IMG_SIZE, NEW_IMG_SIZE))
                        aug_data.append([new_img, img_cat])

                    # Crop from top (80 %, since it accounts for L and R (both top and bottom))
                    else:
                        x = randint(0, OLD_IMG_SIZE-NEW_IMG_SIZE)
                        y = randint(0, OLD_IMG_SIZE-NEW_IMG_SIZE)
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
            imshow(img_arr, cmap='gray') # MatPlotLib
            show()
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
    for img_arr, img_cat in dataset:
        new_img = resize(img_arr, (NEW_IMG_SIZE, NEW_IMG_SIZE))
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
    print(dataset[0][0])
    print(dataset[0][1])
    #for img, cat in dataset:
    for i in range(len(dataset)):
        img = np.array(dataset[i][0])
        cat = dataset[i][1]
        # Checks if the folder exists, and then creates it if it doesn't
        if not exists("{}".format(folder)):
            mkdir("{}".format(folder))
        # Checks if the folder exists, and creates it if it doesn;t
        if not exists("{}/{}".format(folder, cat)):
            mkdir("{}/{}".format(folder, cat))
        # Writes image into folder
        path = "{}/{}/{}.jpg".format(folder, cat, index)
        imwrite(path, img)
        # Increase index, for obv reasons
        index += 1

def loadProcessedImages(folder):
    """
    Loads the augmented and processed dataset for training
    Args:
        folder (str): path to the augmented data folder

    Returns:
        dataset : List of all images (input vector) and categories (output vector)
    """
    dataset = []
    # Loops through the processed images
    for i in range(11):
        path = "{}/{}".format(folder, i)
        for filename in listdir(path):
            img = imread(join(path, filename))
            vec = [0] * 11
            vec[i] = 1
            dataset.append([img, vec])
    # Returns dataset
    return dataset

if __name__ == '__main__':
    # Set-up parser
    parser = ArgumentParser("Preprocess the data")
    parser.add_argument('datasetPath', type=str, help="Path to folder which stores the dataset")
    parser.add_argument('savePath', type=str, help="Name of folder to save processed images to")
    # Parse args
    args = parser.parse_args()
    # Get values
    DATASET = args.datasetPath
    FOLDER = args.savePath
    CATEGORIES = ["Ci", "Cs", "Cc", "Ac", "As", "Cu", "Cb", "Ns", "Sc", "St", "Ct"]

    print("----------------------------------------------")
    print("Dataset Path : ", DATASET)
    print("Save Folder  : ", FOLDER)
    print("Categories   : ", CATEGORIES)
    print("----------------------------------------------")
    # Steps:
    # 1. Load dataset
    rawData = loadRawData(CATEGORIES, DATASET)
    print("Loaded Unprocessed Data")
    # 2. Augment some images, from 400*400 to 256*256
    augImgs = augmentImages(rawData, 400, 256)
    print("Images Augmented")
    # 3. Resize all old images to 256*256
    resized = resizeOriginalImages(rawData, 256)
    print("Original Images Resized")
    # 4. Merge datasets
    dataset = resized + augImgs
    print("Merged Dataset Created")
    # 5. Save Processed Images
    saveImages(dataset, FOLDER)
    print("Merged dataset saved in folder {}".format(FOLDER))
    print("----------------------------------------------")