import numpy as np
import math
def entropy(data):
    numEntries = len(data)  
    labelCounts = {}  
    for featVec in data:      #create the dictionary for all of the data  
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    ent = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        ent -= prob*math.log(prob,2) #get the log value  
    return ent 

def countLabel(data):
    labelCounts = {}
    for featVec in data:
        label = featVec[-1]
        if label not in labelCounts.keys():  
            labelCounts[label] = 0  
        labelCounts[label] += 1  
    print zeroR(labelCounts)

def zeroR(data):
    '''
       Given a list or sklearn-style dictionary, return the most common value
    '''
    if type(data) == dict:
        y_vals = data['target']
    else:
        y_vals = data
    class_counts = dict.fromkeys(y_vals, 0)
    for i in y_vals:
        class_counts[i] += 1
    return max(class_counts, key=class_counts.get)

arr = np.array([[1,1,1,1,1,1,1,1,1,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0,0,0,0,0]])
#print entropy(arr)
#print arr[-1]
#arrr = [0,1,2,3,5,8]
#print arrr.index(5)
featList = [example[-1] for example in arr]
print featList

print zeroR([example[-1] for example in arr])