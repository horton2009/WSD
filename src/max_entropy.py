#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.classify import MaxentClassifier

from base_wsd import BaseWSDI


class MaxEntropyWSD(BaseWSDI):
    def __init__(self):
        super(MaxEntropyWSD, self).__init__()
        pass

    def train(self, features_label):
        self._classifier = MaxentClassifier.train(features_label, algorithm='iis', trace=0, max_iter=80)
        return None


def main():
    wsd = MaxEntropyWSD()
    wsd.run()


if __name__ == '__main__':
    main()
