#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os


class BaseWSDI(object):
    def __init__(self):
        self._classifier = None
        pass

    def load_features(self, path):
        features_label = []
        with open(path, 'rb') as f:
            for line in f:
                tokens = line.split('|')
                feature = {}
                for t in tokens[:-1]:
                    fname, fvalue = t.strip().split('=')
                    feature[fname] = fvalue
                # if loading test features, the label is the word_num, like 中医.1
                label = tokens[-1].strip()
                if len(feature) != 10:
                    assert len(feature) == 8, 'Invalid feature instance: %s' % line
                    # fill the chunk features with NULL
                    feature['PW'] = 'NULL'
                    feature['PT'] = 'NULL'
                features_label.append((feature, label))
        return features_label

    def train(self, features_label):
        """ To be overwrite by subclass. Train a model and assign to self._classifier"""
        return None

    def __classify_a_feature(self, feature):
        label = self._classifier.classify(feature)
        return label

    def classify(self, path):
        """
        return results like this:
        {'word': u'中医', 'label': ['traditional_Chinese_medical_science', 'traditional_Chinese_medical_science', ...]}
        """
        result = {}
        features = self.load_features(path)
        result['word'] = features[0][1].split('.')[0]
        labels = []
        for feature in features:
            label = self.__classify_a_feature(feature[0])
            labels.append(label)
        result['label'] = labels
        return result

    def dump_result(self, result, file_obj):
        word = result['word']
        for i, label in enumerate(result['label']):
            file_obj.write('%s.%d %s\n' % (word, i + 1, label))
        return None

    @staticmethod
    def get_words(path):
        with open(path, 'rb') as f:
            line = f.readline()
            words = line.strip().split(' ')
        return words

    @classmethod
    def run(cls):
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wsd = cls()
        TRAIN_DIR = os.path.join(ROOT, 'train/')
        TEST_DIR = os.path.join(ROOT, 'test/')
        TEST_NAME_FILE = os.path.join(ROOT, 'test/namefile')
        classname = str(cls).split('.')[-1][:-2]
        RESULT_PATH = os.path.join(ROOT, 'result/%s_result.txt' % classname)
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
        return None
