#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.classify import DecisionTreeClassifier

from base_wsd import BaseWSDI


class DecisionTreeWSD(BaseWSDI):
    def __init__(self):
        super(DecisionTreeWSD, self).__init__()
        pass

    def train(self, features_label):
        self._classifier = DecisionTreeClassifier.train(features_label, entropy_cutoff=0.05, depth_cutoff=200, support_cutoff=20)
        return None


def main():
    wsd = DecisionTreeWSD()
    wsd.run()


if __name__ == '__main__':
    main()
