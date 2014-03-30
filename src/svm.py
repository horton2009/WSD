#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

from base_wsd import BaseWSDI


class SVMWSD(BaseWSDI):
    def __init__(self):
        super(SVMWSD, self).__init__()
        self._numeric_feature_value = True

    def train(self, features_label):
        svm = SklearnClassifier(SVC(C=10.0, gamma=0.0001))
        self._classifier = svm.train(features_label)
        return None


def main():
    wsd = SVMWSD()
    wsd.run()


if __name__ == '__main__':
    main()
