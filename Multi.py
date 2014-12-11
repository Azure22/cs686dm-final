from sklearn import svm
from sklearn import linear_model
import util
import math
import csv


class Multi():

    def __init__(self, num_features=10):
        self.num_bags = num_features
        self.svm = svm.SVC()
        self.max_ent = linear_model.LogisticRegression
        self.Naive_Bayes = NaiveBayesClassifier(self.num_bags)

    def train(self, X, Y):
        self.svm.fit(X, Y)
        self.max_ent.fit(X, Y)

    def predict_svm(self, X, Y):
        return self.svm.score(X, Y)

    def predict_max(self, X, Y):
        return self.max_ent.score(X, Y)


class NaiveBayesClassifier():

    def __init__(self, num_features):
        self.legalLabels = [0, 1]
        self.labels = {}
        self.dict = {}
        self.temp_1 = {}
        self.temp_2 = {}
        self.k = 0.1
        self.features = []

        for i in range(num_features):
            self.features.append(i)

    def train(self, trainingData, trainingLabels):

        for label in self.legalLabels:
            self.labels[label] = 0

        for label in trainingLabels:
            if label in self.legalLabels:
                self.labels[label] += 1

        for label in self.legalLabels:
            self.dict[label] = {}
            for bag in self.features:
                self.dict[label][bag] = {}
                for i in range(2):
                    self.dict[label][bag][i] = 0

        count = 0
        for review in trainingData:
            for bag in range(len(review)):
                value = review[bag]
                current_label = trainingLabels[count]
                self.dict[current_label][bag][value] += 1
            count += 1

        s = 0
        for label in self.labels:
            s += self.labels[label]

        for label in self.labels:
            if self.labels[label] and s:
                self.temp_1[label] = float(self.labels[label]) / float(s)
            else:
                self.temp_1[label] = 0

            for label in self.legalLabels:
                self.temp_2[label] = {}
            for bag in self.features:
                self.temp_2[label][bag] = {}
                for value in range(2):
                    x = self.dict[label][bag][value] + self.k
                    y = self.labels[label] + 2 * self.k
                    if x and y:
                        self.temp_2[label][bag][value] = float(x) / float(y)
                    else:
                        self.temp_2[label][bag][value] = 0

    def classify(self, testData):
        guesses = []
        for review in testData:
            posterior = self.calculateLogJointProbabilities(review)
            guesses.append(posterior.argMax())
        return guesses

    def calculateLogJointProbabilities(self, review):
        logJoint = util.Counter()

        for label in self.legalLabels:
            p = math.log(self.temp_1[label])
            q = 0
            for bag in range(len(review)):
                value = review[bag]
                q += math.log(self.temp_2[label][bag][value])
            logJoint[label] = p + q

        return logJoint


if __name__ == '__main__':
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    f_train = open("train_data_b.csv", "r")
    for line in csv.reader(f_train.readline().splitlines()):
        train_data.append(line[0:-1])
        train_labels.append(line[-1])

    f_test = open("test_data_b.csv", "r")
    for line in csv.reader(f_test.readline().splitlines()):
        test_data.append(line[0:-1])
        test_labels.append(line[-1])

    if len(train_data):
        multi = Multi(len(train_data[0]))
        multi.Naive_Bayes.train(train_data, train_labels)
        guesses = multi.Naive_Bayes.classify(test_data)

        correct = 0
        for i in range(len(test_labels)):
            if test_labels[i] == guesses[i]:
                correct += 1
        print "Naive Bayes: "
        print float(correct) / len(test_labels)