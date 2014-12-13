import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import csv

f = open('train_data.csv', 'r')
data = list(csv.reader(f.read().splitlines()))

rows = np.array(data)
print 'the size of these data is '
print rows.shape

for i in range(10):
    print '========================='

    print i
    column = rows[:,i]
    column = [ float(column) for column in column[1:] if column ]
    arr = np.array(column, dtype = float)
    print  'the Maximum value of column ' + str(i) + ' is '+ str(arr.max())
    print  'the Minimum value of column ' + str(i) + ' is '+ str(arr.min())
    print  'the Mean of column ' + str(i) + ' is '+ str(arr.mean())
    print  'the Standard deviation of column ' + str(i) + ' is '+ str(arr.std())
    print 'the Median of column ' + str(i) + ' is ' + str(np.median(arr))
    mode = Counter(arr)
    print  'the mode of column ' + str(i) + ' is '+ str(mode.most_common(1))
