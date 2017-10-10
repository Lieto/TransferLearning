from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset

class CatsAndDogsData(Dataset):

    def __init__(self, subset):
        super(CatsAndDogsData, self).__init__("CatsAndDogs", subset)

    def num_classes(self):

        return 2

    def num_examples_per_epoch(self):

        if self.subset == "train":
            return 22778
        if self.subset == "validation":
            return 2222

    def download_message(self):

        print("Failed to find any CatsAndDogs {} files".format(
            self.subset
        ))
