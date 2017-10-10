#!/usr/bin/env python3

import functools
import struct

from tornado.concurrent import Future

import numpy as np
import cv2

import time
from matplotlib import pyplot as plt
from PIL import Image

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

from vacore import  VAService, main

def plot_preds(mat, preds):

    text = ""
    prob_text = ""

    if preds[0] > preds[1]:
        text = "Class: Cat"
        prob_text = "Probability: " + str(preds[0])
        print("Prediction: cat, probability: {}".format(preds[0]))
    elif preds[1] > preds[0]:
        text = "Class: Dog"
        prob_text = "Probability: " + str(preds[1])
        print("Prediction: dog, probability: {}".format(preds[1]))

    cv2.putText(mat, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.putText(mat, prob_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    #cv2.addText(img=mat, text=text, org=(10, 10), nameFont=cv2.FONT_HERSHEY_DUPLEX, pointSize=10, color=(255, 0, 0),
    #            weight=None, spacing=None)

    cv2.imshow("Prediction", mat)
    cv2.waitKey(1)

    #plt.imshow(img)
    #plt.axis('off')

    #plt.figure()
    #labels = ("cat", "dog")
    #plt.barh([0, 1], preds, alpha=0.5)
    #plt.yticks([0, 1], labels)
    #plt.xlabel("Probability")
    #plt.xlim(0, 1.01)
    #plt.tight_layout()
    #plt.show()

class resultviewer(VAService):

    def reload(self, *args, **kwargs):
        super().reload(*args, **kwargs)

        self.frame_no = 0
        self.target_size = (229, 229)
        self.model = load_model(self.config["model"])

        for taskname in self.config["listeners"].keys():
            taskspec = self.config["listeners"][taskname]
            self.subscribe(taskspec["socket"], taskspec["topic"], functools.partial(self.frame_callback, taskname))

    def frame_callback(self, taskname, msg):

        if isinstance(msg, Future):
            msg = msg.result()

        topic = msg[0]

        if topic == b"raw":
            msg_time_ms = float(struct.unpack(">Q", msg[1])[0]) / 1000
            msg_frameno = struct.unpack(">Q" , msg[2])[0]
            w = struct.unpack(">i", msg[3])[0]
            h = struct.unpack(">i", msg[4])[0]

            mat = np.frombuffer(msg[5], dtype=np.uint8).reshape([h, w, 3])

            img = Image.fromarray(mat)
            if img.size != self.target_size:
                img = img.resize(self.target_size)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            start_time = time.time()
            preds = self.model.predict(x)
            end_time = time.time()

            delta = end_time - start_time
            self.logger.info("Prediction took: {}".format(delta))

            self.logger.info("Predictions:  {}".format(preds))

            mat_copy = np.copy(mat)
            plot_preds(mat_copy, preds[0])


if __name__ == "__main__":

    instance = main(__file__, resultviewer)
