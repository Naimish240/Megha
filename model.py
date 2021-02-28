import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNN(object):
    def __init__(self, epochs, batchSize):
        """
        Constructor for the Model
        Args:
            epochs (int): Number of epochs to train for
            batchSize (int): Batch size for network
        """
        self.epochs=epochs
        self.history = None
        self.batch_size = batchSize
        self.model = keras.Sequential()
    
    def createCNN(self):
        """
        Creates the Model, in all it's glory
        """
        # Add preprocessing layer to convert from [0:255] to [0:1]
        self.model.add(layers.experimental.preprocessing.Rescaling(
            1./255, 
            input_shape=(-1,256,256,1)
                ))

        # Add First Conv Layer, from Alex
        # 256*256*1 to 63*63*64
        self.model.add(layers.Conv2D(
            filters=64, 
            kernel_size=11, 
            strides=(4,4), 
            activation='relu')
                )

        # Do some BatchNorm
        self.model.add(keras.layers.BatchNormalization())
        # Throw in a Max Pooling layer
        # 63*63*64 to 31*31*64
        self.model.add(layers.MaxPooling2D(
            pool_size=(3,3), strides=(2,2)
        ))

        # Add second Conv Layer
        # 31*31*64 to 31*31*256
        self.model.add(layers.Conv2D(
            filters=224, 
            kernel_size=(5,5), 
            strides=(1,1), 
            activation='relu', 
            padding="same"
                ))
                
        # Do some BatchNorm
        self.model.add(keras.layers.BatchNormalization())
        # Throw in a Max Pooling layer
        # 31*31*224 to 15*15*224
        self.model.add(layers.MaxPooling2D(
            pool_size=(3,3), strides=(2,2)
        ))

        # Add another Conv2D layer
        # 15*15*224 to 15*15*96
        self.model.add(layers.Conv2D(
            filters=96,
            kernel_size=(3,3),
            strides=(1,1),
            activation='relu',
            padding='same'
        ))

        # Batch Norm is the norm
        self.model.add(keras.layers.BatchNormalization())

        # Add another Conv2D layer
        # 15*15*96 to 15*15*32
        self.model.add(layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            strides=(1,1),
            activation='relu',
            padding='same'
        ))

        # MaxPool layer
        # 15*15*32 to 7*7*32
        self.model.add(layers.MaxPooling2D(
            pool_size=(3,3), strides=(2,2)
        ))

        # Time to go Flat
        # 7*7*32 to 1568*1
        self.model.add(layers.Flatten())

        # Add a Dense layer
        self.model.add(layers.Dense(768, 'relu'))
        # Dropout
        self.model.add(layers.Dropout(0.4))
        # Another Dense Layer
        self.model.add(layers.Dense(320, 'relu'))
        # One Final Dropout
        self.model.add(layers.Dropout(0.5))
        # Output Layer
        self.model.add(layers.Dense(11, 'softmax'))
    
    def compile(self):
        """
        Compiles the model
        """
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
            )

    def train(self, x_train, y_train, x_val, y_val, checkpoint_path):
        """
        Trains the model

        Args:
            x_train (list): Training dataset input
            y_train (list): Training dataset output
            x_val (list): Validation dataset input
            y_val (list): Validation dataset output
            checkpoint_path (str): Path to save checkpoints
        """
        # With stuff for saving, callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        self.history = self.model.fit(
            x_train, 
            y_train, 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            validation_data=(x_val, y_val),
            callbacks=[cp_callback]
        )

    def analyse(self):
        """
        Function to draw graphs for visualising the model's performance
        """
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

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
    
    def save(self, folder):
        """
        Function to save the model into a folder in the SavedModel format

        Args:
            folder (str): Path of folder to save model to
        """
        # Save model in folder, in SavedModel format
        self.model.save('{}/cloudModel'.format(folder))

    def load(self, path):
        """
        Function to load a model

        Args:
            path (str): Path to model to load
        """
        # Loads a saved model
        self.model = keras.models.load_model(path)

    def predict(self, x):
        """
        Function to get prediction from model

        Args:
            x (list): Input Image to get prediction for

        Returns:
            list; probability of each feature
        """
        # Run prediction, after training
        return self.model.predict(x)

    def summary(self):
        """
        Generate summary for the model
        """
        self.model.summary()

if __name__ == "__main__":
    print("Error! Import-only script!")