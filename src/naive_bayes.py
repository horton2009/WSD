#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NLTK_classify import NaiveBayesClassifier

from base_wsd import BaseWSDI
import util


class NaiveBayesWSD(BaseWSDI):
    def __init__(self):
        super(NaiveBayesWSD, self).__init__()
        pass

    def train(self, features_label):
        self._classifier = NaiveBayesClassifier.train(features_label)
        return None


def main():
    wsd = NaiveBayesWSD()
    wsd.run()
    util.evaluate("../result/NaiveBayesWSD_result.txt", "../result/test_answer")


if __name__ == '__main__':
    main()
