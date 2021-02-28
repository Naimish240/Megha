from model import CNN
from argparse import ArgumentParser
from cv2 import imread, resize
'''
Predictions :
0  - Cirrus        : A warm front is approaching.
1  - Cirrostratus  : A storm is coming.
2  - Cirrocumulus  : The weather is about to change!
3  - Altocumulus   : Rain is coming soon!
4  - Altostratus   : Rain, incoming.
5  - Cumulus       : Fair weather!
6  - Cumulonimbus  : Thunderstorms may be due.
7  - Nimbostratus  : Rain / Fog incoming.
8  - Stratocumulus : Bad weather incoming.
9  - Stratus       : Light rain.
10 - Contrails     : Airplanes!
'''
CATEGORIES = ["Cirrus", "Cirrostratus", "Cirrocumulus", "Altocumulus", "Altostratus", "Cumulus", "Cumulonimbus", "Nimbostratus", "Stratocumulus", "Stratus", "Contrails"]

def load(path):
    model = CNN()
    model.load(path)
    return model

def predict(model, image):
    global CATEGORIES
    # Loads and resizes image
    img = cv2.imread(image, 0)
    img = resize(img, (256, 256))
    # Get prediction for image
    output = model.predict(image)
    # Convert to dictionary
    score = {}
    for i in range(len(output)):
        score[CATEGORIES[i]] = output[i]
    # List of keys, sorted by score
    sorted_score = sorted(score, key=score.get)
    sorted_score = sorted_score[::-1]
    # Print top 3 confidence
    for i in range(3):
        print("{} : {}".format(sorted_score[i], score[i]))
    # Print prediction based off of most likely estimate
    if CATEGOTIES.index(sorted_score[0]) == 0:
        print("A warm front is approaching.")
    if CATEGOTIES.index(sorted_score[0]) == 1:
        print("A storm is coming.")
    if CATEGOTIES.index(sorted_score[0]) == 2:
        print("The weather is about to change!")
    if CATEGOTIES.index(sorted_score[0]) == 3:
        print("Rain is coming soon!")
    if CATEGOTIES.index(sorted_score[0]) == 4:
        print("Rain, incoming.")
    if CATEGOTIES.index(sorted_score[0]) == 5:
        print("Fair weather!")
    if CATEGOTIES.index(sorted_score[0]) == 6:
        print("Thunderstorms may be due.")
    if CATEGOTIES.index(sorted_score[0]) == 7:
        print("Rain / Fog incoming.")
    if CATEGOTIES.index(sorted_score[0]) == 8:
        print("Bad weather incoming.")
    if CATEGOTIES.index(sorted_score[0]) == 9:
        print("Light rain.")
    if CATEGOTIES.index(sorted_score[0]) == 10:
        print("Airplanes!")

def main(path, image):
    # Load the model
    model = load(model)
    res = predict(image)

if __name__ == '__main__':
    # Add Argument parser
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to model to run on", required=True)
    parser.add_argument("image", type=str, help="Image path to run prediction on", required=True)
    args = parser.parse_args()
    # Run Main
    main(path, image)