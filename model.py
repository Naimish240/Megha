import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNN(object):
    def __init__(self, epochs, batchSize):
        self.epochs=epochs
        self.history = None
        self.batch_size = batchSize
        self.model = keras.Sequential()

        # Add preprocessing layer to convert from [0:255] to [0:1]
        self.model.add(layers.experimental.preprocessing.Rescaling(
            1./255, 
            input_shape=(X.shape[1:])
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
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
            )

    def train(self, x_train, y_train, x_val, y_val):
        self.history = model.fit(
            x_train, 
            y_train, 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            validation_data=(x_val, y_val))

    def analyse(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

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

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    print("Error! Import-only script!")