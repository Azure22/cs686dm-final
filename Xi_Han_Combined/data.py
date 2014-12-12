import joblib
import math
import nltk
import os
import os.path
import numpy as np
from sklearn import preprocessing as pp
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd

words = []
scores = []
tokenizer = RegexpTokenizer(r'\w+')

def cat(score):
    if score == 1:
        return 9
    else:
        return int(math.floor(score * 10.0))

def org_words(filename):
    result = {}
    if not os.path.isfile(filename):

        file = open("imdb.vocab")
        for line in file:
            words.append(line.strip('\n'))
        file.close()
        file = open("imdbEr.txt")
        for line in file:
            scores.append(float(line.strip('\n')))
        file.close()

        scaler = pp.MinMaxScaler().fit(scores)
        t_scores = scaler.transform(scores)

        for i in range(len(words)):
            result[words[i]] = cat(t_scores[i])
        joblib.dump(result, filename)
    else:
        result = joblib.load(filename)
    return result

def org_docs(path, filename):
    result = []
    if not os.path.isfile(filename):
        list_dirs = os.listdir(path)
        for f in list_dirs:
            print f
            file = open(os.path.join(path, f))
            result.append(tokenizer.tokenize(file.read()))
            file.close()
        joblib.dump(result, filename)
    else:
        result = joblib.load(filename)
    return result

def org_doc_scores(docs, filename):
    result = []
    if not os.path.isfile(filename):
        for ws in docs:
            c_scores = [0,0,0,0,0,0,0,0,0,0]
            for w in ws:
                if wordDict.has_key(w):
                    c_scores[wordDict[w]] += 1
            result.append(c_scores)
        joblib.dump(result, filename)
    else:
        result = joblib.load(filename)
    return result

def to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename + '.csv')

def to_binary(data):
    result = []
    for s in data:
        t = []
        for v in s:
            if v > 0:
                t.append(1)
            else:
                t.append(0)
        result.append(t)
    return result

print tokenizer.tokenize("I caught this on IFC last week and I thought it was typical of the indie short subject film.")

#wordDict = org_words("wordbag")
#docList = org_docs("D:/Projects/ProP/CS686DMFinal/CS686DMFinal/data/train/neg", "document_train_neg")
#docScores = org_doc_scores(docList, "document_train_neg_scores")

#docList = org_docs("D:/Projects/ProP/CS686DMFinal/CS686DMFinal/data/train/pos", "document_train_pos")
#docScores = org_doc_scores(docList, "document_train_pos_scores")

#docList = org_docs("D:/Projects/ProP/CS686DMFinal/CS686DMFinal/data/test/neg", "document_test_neg")
#docScores = org_doc_scores(docList, "document_test_neg_scores")

#docList = org_docs("D:/Projects/ProP/CS686DMFinal/CS686DMFinal/data/test/pos", "document_test_pos")
#docScores = org_doc_scores(docList, "document_test_pos_scores")

#train_neg = joblib.load("document_train_neg_scores")
#train_pos = joblib.load("document_train_pos_scores")
#test_neg = joblib.load("document_test_neg_scores")
#test_pos = joblib.load("document_test_pos_scores")

#doc_train_neg = np.array(to_binary(train_neg))
#doc_train_pos = np.array(to_binary(train_pos))
#doc_test_neg = np.array(to_binary(test_neg))
#doc_test_pos = np.array(to_binary(test_pos))

#to_csv(doc_train_neg, "document_train_neg_scores_b")
#to_csv(doc_train_pos, "document_train_pos_scores_b")
#to_csv(doc_test_neg, "document_test_neg_scores_b")
#to_csv(doc_test_pos, "document_test_pos_scores_b")

