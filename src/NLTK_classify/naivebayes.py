#-*-coding:utf-8-*-
# Natural Language Toolkit: Naive Bayes Classifiers
#
# Copyright (C) 2001-2012 NLTK Project
# Author: Edward Loper <edloper@gradient.cis.upenn.edu>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A classifier based on the Naive Bayes algorithm.  In order to find the
probability for a label, this algorithm first uses the Bayes rule to
express P(label|features) in terms of P(label) and P(features|label):

|                       P(label) * P(features|label)
|  P(label|features) = ------------------------------
|                              P(features)

The algorithm then makes the 'naive' assumption that all features are
independent, given the label:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------
|                                         P(features)

Rather than computing P(featues) explicitly, the algorithm just
calculates the denominator for each label, and normalizes them so they
sum to one:

|                       P(label) * P(f1|label) * ... * P(fn|label)
|  P(label|features) = --------------------------------------------		# 这里的归一化主要是使和为 1 
|                        SUM[l]( P(l) * P(f1|l) * ... * P(fn|l) )
"""

from collections import defaultdict

from probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs

from api import ClassifierI

##//////////////////////////////////////////////////////
##  Naive Bayes Classifier
##//////////////////////////////////////////////////////

class NaiveBayesClassifier(ClassifierI):
    """
    A Naive Bayes classifier.  Naive Bayes classifiers are
    paramaterized by two probability distributions:

      - P(label) gives the probability that an input will receive each
        label, given no information about the input's features.

      - P(fname=fval|label) gives the probability that a given feature
        (fname) will receive a given value (fval), given that the
        label (label).

    If the classifier encounters an input with a feature that has
    never been seen with any label, then rather than assigning a
    probability of 0 to all labels, it will ignore that feature.

    The feature value 'None' is reserved for unseen feature values;
    you generally should not use 'None' as a feature value for one of
    your own features.
    """
    def __init__(self, label_probdist, feature_probdist):
        """
        :param label_probdist: P(label), the probability distribution
            over labels.  It is expressed as a ``ProbDistI`` whose
            samples are labels.  I.e., P(label) =
            ``label_probdist.prob(label)``.

        :param feature_probdist: P(fname=fval|label), the probability
            distribution for feature values, given labels.  It is
            expressed as a dictionary whose keys are ``(label, fname)``
            pairs and whose values are ``ProbDistI`` objects over feature
            values.  I.e., P(fname=fval|label) =
            ``feature_probdist[label,fname].prob(fval)``.  If a given
            ``(label,fname)`` is not a key in ``feature_probdist``, then
            it is assumed that the corresponding P(fname=fval|label)
            is 0 for all values of ``fval``.
        """
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = label_probdist.samples()

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        # Discard any feature names that we've never seen before.
        # Otherwise, we'll just assign a probability of 0 to
        # everything.
        featureset = featureset.copy()
	#-------------------------------------------------------- 删掉没有与任何类共现的特征
        for fname in featureset.keys():
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                #print 'Ignoring unseen feature %s' % fname
                del featureset[fname]

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label,fname]
                    logprob[label] += feature_probs.logprob(fval)	# 对于特征值类型的对数概率
                else:
                    # nb: This case will never come up if the
                    # classifier was created by
                    # NaiveBayesClassifier.train().
                    logprob[label] += sum_logs([]) # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)

    def show_most_informative_features(self, n=10):
        # Determine the most relevant features, and display them.
        cpdist = self._feature_probdist
        print 'Most Informative Features'

        for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):						# 函数的嵌套定义，内部函数可以直接使用外部函数的变量
                return cpdist[l,fname].prob(fval)
            labels = sorted([l for l in self._labels
                             if fval in cpdist[l,fname].samples()], 	# 将不同的类按产生，同一特征及特征值的概率从小到大排序
                            key=labelprob)
            if len(labels) == 1: continue				# 如果该特征及特征值只与一个类共现，不做比较，否则比较最后一个类和第一个类产生该特征及值的概率比例
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0,fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) /	# 最大/最小
                                  cpdist[l0,fname].prob(fval))
            print ('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname, fval, str(l1)[:6], str(l0)[:6], ratio))

    def most_informative_features(self, n=100):			# 特征值的最小概率与最大概率比值最大，即差异最小的特征极为：最具信息量特征
        """
        Return a list of the 'most informative' features used by this
        classifier.  For the purpose of this function, the
        informativeness of a feature ``(fname,fval)`` is equal to the
        highest value of P(fname=fval|label), for any label, divided by
        the lowest value of P(fname=fval|label), for any label:

        |  max[ P(fname=fval|label1) / P(fname=fval|label2) ]
        """
        # The set of (fname, fval) pairs used by this classifier.
        features = set()
        # The max & min probability associated w/ each (fname, fval)
        # pair.  Maps (fname,fval) -> float.
        maxprob = defaultdict(lambda: 0.0)				# value 初始化为0.0的字典
        minprob = defaultdict(lambda: 1.0)		

        for (label, fname), probdist in self._feature_probdist.items():
            for fval in probdist.samples():	# samples（）返回 key 的 list
                feature = (fname, fval)
                features.add( feature )
                p = probdist.prob(fval)
                maxprob[feature] = max(p, maxprob[feature])		# 存储每个 feature 的最大概率的 value
                minprob[feature] = min(p, minprob[feature])
                if minprob[feature] == 0:
                    features.discard(feature)

        # Convert features to a list, & sort it by how informative
        # features are.
        features = sorted(features,
            key=lambda feature: minprob[feature]/maxprob[feature])	# sorted（set）返回的是一个 List
        return features[:n]

    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):	# ELEProbDist:为类名，类名作为参数
        """
        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.
        """
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)	# value 为 Freqdict 的字典
        feature_values = defaultdict(set)		# value 为 set 的字典
        fnames = set()

        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:		# 原始通用特征 ［（{feature dict}，label） ，（ ）］
            label_freqdist.inc(label)
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname].inc(fval)	# featureset 为 dict; feature_freqdist[label, fname] 为 freqdict: 统计每个特征，某个值的出现次数
                # Record that fname can take the value fval.	# ！！！ 所以，不管特征

                feature_values[fname].add(fval)			# value 为 set 的字典
                # Keep a list of all feature names.
                fnames.add(fname)

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]			# 所有样本中 某类的总次数
            for fname in fnames:
                count = feature_freqdist[label, fname].N()	# freqdict.N(): 为freqdict()的所有频数之和，即：与特定类共现过的所有特征（key）的种的出现次数
                feature_freqdist[label, fname].inc(None, num_samples-count)	# freqdist.inc(key): 给 key 的 value 加 1 （第二个参数默认为 1）
										# 每个特征的某一值对于某个类的概率都是基于该类的样本总数计算

                feature_values[fname].add(None)			# 每个特征值的种类，都增加一个 None。

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)		# 默认平滑： gamma=0.5, bins = len（label_freqdist）

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))	# estimator:为类别名，用作概率平滑的类：LidstoneProbDist
            feature_probdist[label,fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)

##//////////////////////////////////////////////////////
##  Demo
##//////////////////////////////////////////////////////

def demo():
    from nltk.classify.util import names_demo
    classifier = names_demo(NaiveBayesClassifier.train)		# 方法作为参数
    classifier.show_most_informative_features()

if __name__ == '__main__':
    demo()


