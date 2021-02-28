import random
import argparse
import numpy as np
from model import CNN
from preprocessing import loadProcessedImages

def main(folder, epochs, batch_size, val_frc):
    # Initialize the arrays
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    # Set Seed
    random.seed(69)
    # Load Dataset
    dataset = loadProcessedImages(folder)
    # Shuffle the dataset around, to avoid training "accidents"
    random.shuffle(dataset)
    # Split dataset into X and Y
    for i in range(len(dataset)):
        # Get values from dataset
        x, y = dataset[i]
        # Yeet into validation
        if i < val_frc:
            x_val.append(x)
            y_val.append(y)
        # Add to train
        else:
            x_train.append(x)
            y_train.append(x)
    # Create Model
    model = CNN(epochs, batch_size)
    # Compile model
    model.compile()
    # Print model summary
    model.summary()
    # Train the model
    model.train(x_train, y_train, x_val, y_val)
    # Show Model Analysis
    model.analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help="Folder to load data from", required=True)
    parser.add_argument('epochs', type=int, help="Number of epochs to train for", required=True)
    parser.add_argument('batchS', type=int, help="Batch Size", required=True)
    parser.add_argument('valfrc', type=float, help="Validation fraction", required=True)
    args = parser.parse_args()

    FOLDER = args.folder
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    VAL_FRAC = args.valfrc
    main(FOLDER, EPOCHS, BATCH_SIZE, VAL_FRAC)
    