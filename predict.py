import argparse
import logging

import sys
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

target_size = (229, 229)

def predict(model, img, target_size):

    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def plot_preds(img, preds):

    plt.imshow(img)
    plt.axis('off')

    plt.figure()
    labels = ("cat", "dog")
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel("Probability")
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    logging.info("Parsing arguments")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/media/kuoppves/My Passport/CatsAndDogs/test_dir/cat.5.jpg")
    parser.add_argument("--image_url", type=str, default="")
    parser.add_argument("--model", type=str, default="./inceptionv3-ft.model")

    args = parser.parse_args()

    if args.image is None and args.image_url is None:

        parser.print_help()
        sys.exit(1)

    model = load_model(args.model)

    if args.image is not None:
        img = Image.open(args.image)
        preds = predict(model, img, target_size)
        plot_preds(img, preds)

    if args.image_url is not None:

        response = requests.get(args.image_url)
        img = Image.open(BytesIO(response.content))
        preds = predict(model, img, target_size)
        plot_preds(img, preds)

