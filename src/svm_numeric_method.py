#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

from base_wsd import BaseWSDI
from util import evaluate


class SVMWSD(BaseWSDI):
    def __init__(self):
        super(SVMWSD, self).__init__()
        self._numeric_feature_value = True

    def load_features(self, path, test=False):
        features_label = []
        with open(path, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                tokens = line.split('|')
                feature = {}
                for t in tokens[:-1]:
                    # fname = t.strip()
                    # fvalue = 1
                    fname, fvalue = t.strip().split('=')
                    feature[fname] = fvalue
                # if loading test features, the label is the word_num, like 中医.1
                label = tokens[-1].strip()
                features_label.append((feature, label))
        # convert the numeric feature values from string to number for svm use
        if self._numeric_feature_value:
            self._convert_feature_value(features_label, test=test)
        return features_label


    def train(self, features_label):
        svm = SklearnClassifier(SVC(C=1000.0, gamma=0.0001))
        self._classifier = svm.train(features_label)
        return None

    @classmethod
    def run(cls, eval_flag=True):
        """
        @param eval_flag: if eval_flag is True, the evaluation output is given.
        """
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wsd = cls()
        # TRAIN_DIR = os.path.join(ROOT, 'train/')
        # TEST_DIR = os.path.join(ROOT, 'test/')
        TRAIN_DIR = os.path.join(ROOT, 'features_archive/0-2-2/train/')
        TEST_DIR = os.path.join(ROOT, 'features_archive/0-2-2/test/')
        TEST_NAME_FILE = os.path.join(ROOT, 'test/namefile')
        RESULT_PATH = os.path.join(ROOT, 'result/%s_result.txt' % cls.__name__)
        cls.result_path = RESULT_PATH

        # clear the file RESULT_PATH
        with open(RESULT_PATH, 'wb') as f:
            pass

        result_obj = open(RESULT_PATH, 'ab')

        count = 0
        test_words = cls.get_words(TEST_NAME_FILE)
        for word in test_words:
            test_path = os.path.join(TEST_DIR, word)
            train_path = os.path.join(TRAIN_DIR, word)
            features_label = wsd.load_features(train_path)
            wsd.train(features_label)
            result = wsd.classify(test_path)
            wsd.dump_result(result, result_obj)
            count += 1
            print 'Finish %d of %d: %s' % (count, len(test_words), word)
        result_obj.close()
        print 'Write testing results to %s' % RESULT_PATH
        if eval_flag:
            answerfile = os.path.join(ROOT, 'result/test_answer')
            evaluate(RESULT_PATH, answerfile)
        return None


def main():
    wsd = SVMWSD()
    wsd.run()


if __name__ == '__main__':
    main()
