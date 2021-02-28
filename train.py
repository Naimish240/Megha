import os
import random
import argparse
import numpy as np
from model import CNN
from preprocessing import loadProcessedImages

# NOTE : Run "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" on Jetson Nano to prevent errors

def main(folder, epochs, batch_size, val_frc, chkpts):
    # Initialize the arrays
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    # Set Seed
    random.seed(69)
    # Load Dataset
    dataset = loadProcessedImages(folder)
    print("Dataset Loaded")
    # Shuffle the dataset around, to avoid training "accidents"
    random.shuffle(dataset)
    print("Dataset Shuffled")
    # Split dataset into X and Y
    for i in range(len(dataset)):
        # Get values from dataset
        x, y = dataset[i]
        # Yeet into validation
        if i < val_frc:
            # Normalizing here, instead of in layer due to bugs
            x = x / 255.0
            x_val.append(x)
            y_val.append(y)
        # Add to train
        else:
            # Normalizing here, instead of in layer due to error issues
            x = x / 255.0
            x_train.append(x)
            y_train.append(x)
    print("Dataset split into training and testing")
    print("Samples in training set   : ", len(y_train))
    print("Samples in validation set : ", len(y_val))
    # Create Model
    model = CNN(epochs, batch_size)
    model.createCNN()
    print("Model Created")
    # Compile model
    model.compile()
    print("Model Compiled")
    # Print model summary
    model.summary()
    # Train the model
    print("Training Started")
    model.train(x_train, y_train, x_val, y_val, chkpts)
    # Show Model Analysis
    model.analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help="Folder to load data from")
    parser.add_argument('epochs', type=int, help="Number of epochs to train for")
    parser.add_argument('batchS', type=int, help="Batch Size")
    parser.add_argument('valfrc', type=float, help="Validation fraction")
    parser.add_argument('chkpts', type=str, help="Checkpoint folder")
    args = parser.parse_args()

    FOLDER = args.folder
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchS
    VAL_FRAC = args.valfrc
    CHK_PTS = args.chkpts

    if not os.path.exists("{}".format(CHK_PTS)):
        os.mkdir("{}".format(CHK_PTS))

    main(FOLDER, EPOCHS, BATCH_SIZE, VAL_FRAC, CHK_PTS)
    