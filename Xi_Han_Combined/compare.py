import pandas.io.data as pd
import numpy as np
from sklearn import svm, linear_model
import decisiontree_lzl
import naivebayes_lc

train_data = np.array(pd.read_csv("train_data.csv").values.tolist(), dtype=np.int_)
test_data = np.array(pd.read_csv("test_data.csv").values.tolist(), dtype=np.int_)

train_data_b = np.array(pd.read_csv("train_data_b.csv").values.tolist(), dtype=np.int_)
test_data_b = np.array(pd.read_csv("test_data_b.csv").values.tolist(), dtype=np.int_)

train_X = train_data[:,:10]
train_Y = train_data[:,10]
test_X = test_data[:,:10]
test_Y = test_data[:,10]

train_X_b = train_data_b[:,:10]
train_Y_b = train_data_b[:,10]
test_X_b = test_data_b[:,:10]
test_Y_b = test_data_b[:,10]

clf_svm = svm.SVC()
clf_svm.fit(train_X, train_Y)
svm_score = clf_svm.score(test_X, test_Y)
clf_svm.fit(train_X_b, train_Y_b)
svm_score_b = clf_svm.score(test_X_b, test_Y_b)

clf_lr = linear_model.LogisticRegression()
clf_lr.fit(train_X, train_Y)
lr_score = clf_lr.score(test_X, test_Y)
clf_lr.fit(train_X_b, train_Y_b)
lr_score_b = clf_lr.score(test_X_b, test_Y_b)

dt_score = decisiontree_lzl.pro_start("train_data_b.csv", "test_data_b.csv")
nb_score = naivebayes_lc.interface("train_data_b.csv", "test_data_b.csv")

print "SVM prediction accuracy for normal dataset:"
print svm_score
print "SVM prediction accuracy for binary dataset:"
print svm_score_b
print "Logistic Regression prediction accuracy for normal dataset:"
print lr_score
print "Logistic Regression prediction accuracy for binary dataset:"
print lr_score_b
print "Decision Tree prediction accuracy for binary dataset:"
print dt_score
print "Customized Naive Bayes prediction accuracy for binary dataset:"
print nb_score