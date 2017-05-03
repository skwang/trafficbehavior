import numpy as np, os
from sklearn.neighbors import KNeighborsClassifier

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class SpeedRecognizer:

    def __init__(self):
        train_digits = np.load(os.path.join(code_dir, 'speedestimation', 'digit', 'training_digits.npy'))
        labels = np.load(os.path.join(code_dir, 'speedestimation', 'digit', 'training_labels.npy'))
        self.neigh = KNeighborsClassifier(n_neighbors=1)
        self.neigh.fit(train_digits, labels) 

    def get_digit(self, img, digit=None):
        if digit == 0:
            return img[412:425, 549:556]
        elif digit == 1:
            return img[412:425, 555:562]
        elif digit == 2:
            return img[412:425, 561:568]
        else:
            return img[429:442, 590:597]

    def predict(self, img):
        ones = self.neigh.predict(self.get_digit(img, 2).flatten().reshape(1, -1))
        tens = self.neigh.predict(self.get_digit(img, 1).flatten().reshape(1, -1))
        huns = self.neigh.predict(self.get_digit(img, 0).flatten().reshape(1, -1))
        huns = [0] # TODO: hardcoded to 0 since we don't go over 100
        return huns[0]*100 + tens[0]*10 + ones[0]




