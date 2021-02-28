import os
from model import CNN
from argparse import ArgumentParser

def export(pathFrom, pathTo):
    """
    Function to convert model from tf to tflite

    Args:
        pathFrom (str): Path to the SavedModel, for conversion
        pathTo (str): Path to save the TFLite model to
    """
    # Convert model to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model("{}".format(pathFrom))
    tflite_model = converter.convert()

    # Save the model
    with open('{}.tflite'.format(pathTo), 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    # Add Argument parser
    parser = ArgumentParser()
    parser.add_argument("pathFrom", type=str, help="Path to model to export", required=True)
    parser.add_argument("pathTo", type=str, help="Path to save exported model", required=True)
    args = parser.parse_args()
    # Run Main
    export(pathFrom, pathTo)