import numpy as np
import math
import csv

class TreeNode:
    def __init__(self):
        self.attr = None
        self.child = {}
        self.v = None

    def fit(self, data, attr = None):
        if (attr == []):
            self.child = None
            self.value = zeroR(label)
            return 

        self.attribute = self.bestInfoGainAttr(data, attr)
        #print self.attribute
        if self.attribute == None:
            alist = [array[-1] for array in data]
            self.v = self.zeroR(alist)
        else:
            index = attr.index(self.attribute)
            values = [array[index] for array in data]
            value = set(values)
            attr[index] = None
            #print value
            splitData = {}
            for v in value:
                d = []#select d from data where self.attr =  v
                for line in data:
                    #print line
                    if line[index] == v:
                        #print line[index]
                        d.append(line)
                #a = attr[:].remove(self.attr)
                #print d
                self.child[v] = TreeNode()
                self.child[v].fit(d,attr)
       
    def zeroR(self, data):
        '''
           Given a list or sklearn-style dictionary, return the most common value
        '''
        #print data
        if type(data) == dict:
            y_vals = data['target']
        else:
            y_vals = data
        class_counts = dict.fromkeys(y_vals, 0)
        for i in y_vals:
            class_counts[i] += 1
        #print class_counts
        return max(class_counts, key=class_counts.get)

    def classify(self, d, f):
        if len(self.child) == 0:
            return self.v
        else:
            #print f
            value = d[f.index(self.attribute)]
            #print value
            return self.child[value].classify(d, f)

    def bestInfoGainAttr(self, data, attr):
        maxgain = 0
        maxattr = None
        ent = self.entropy(data)
        #print ent
        for a in attr:
            index = attr.index(a)
            gain = 0 # get gain of data for a
            if a != None:
                attrList = [array[index] for array in data]
                newEntropy = 0.0
                values = set(attrList)
                for value in values:
                    temArr = []
                    for line in data:
                        if (line[index] == value):
                            temArr.append(line)
                    prob = len(temArr)/float(len(data))
                    #print prob
                    #print temArr
                    newEntropy += prob * self.entropy(temArr)
                #print newEntropy
                gain = ent - newEntropy
            else:
                gain = 0
            #print gain
            #print a
            if gain > maxgain:
                maxgain = gain
                maxattr = a
        if maxgain == 0: return None
        return maxattr

    def entropy(self, data):
        numEntries = len(data)  
        labelCounts = {}  
        for attrVec in data:
            currentLabel = attrVec[-1]
            if currentLabel not in labelCounts.keys():  
                labelCounts[currentLabel] = 0  
            labelCounts[currentLabel] += 1  
        ent = 0.0  
        for key in labelCounts:  
            prob = float(labelCounts[key])/numEntries  
            ent -= prob * math.log(prob, 2) 
        return ent 

def getAccur(trainData, testData):
    feature = trainData[0]
    feature[-1] = None
    f = feature[:]
    data = trainData[1:]
    deciTree = TreeNode()
    deciTree.fit(data, feature)
    correctRes = [array[-1] for array in testData]
    correctRes = correctRes[1:]
    teData = testData[1:]
    result = []
    for line in teData:
        result.append(deciTree.classify(line[:-1], f))
    #print deciTree.bestInfoGainAttr(data, feature)
    right  = 0
    for i in range(0, len(correctRes)):
        if result[i] == correctRes[i]:
            right += 1
    accur = float(right) / len(correctRes)
    
    return accur

def pro_start(trainFile, testFile):
    reader1 = csv.reader(file(trainFile, 'rb'))
    trainData = []
    for line in reader1:
        trainData.append(line)
    reader2 = csv.reader(file(testFile, 'rb'))
    testData = []
    for line in reader2:
        testData.append(line)
    return getAccur(trainData, testData)

if __name__ == '__main__':
    accur = pro_start('train_data_b.csv', 'test_data_b.csv')
    print "accuracy:"
    print accur
    ## Read input
    #options = readCommand(sys.argv[1:])
    ## Run clustering
    #runClustering(options)

    
    #feature = trainData[0]
    #feature[-1] = None
    #f = feature[:]
    #data = trainData[1:]
    #deciTree = TreeNode()
    #deciTree.fit(data, feature)
    #correctRes = [array[-1] for array in trainData]
    #correctRes = correctRes[1:]
    #result = []
    #for line in data:
    #    result.append(deciTree.classify(line[:-1], f))
    ##print deciTree.bestInfoGainAttr(data, feature)
    #right  = 0
    #for i in range(0, len(correctRes)):
    #    if result[i] == correctRes[i]:
    #        right += 1
    #accur = float(right) / len(correctRes)
    #print accur