import numpy as np;
import math;
import mlUtil;

dataset = mlUtil.extract_data('tennis.csv');

print 'Load complete';
print '';
#######################################################

class Information:
    #Data
    OriginalFeatureIndex = -1;
    FeatureIndex = -1;
    FeatureName = '';
    Data = np.array([]);
    Target = np.array([]);
    SampleCount = 0;
    #ElementList
    FeatureElementDict = {};
    TargetElementDict = {};
    FeatureElementNames = [];
    TargetElementNames = [];
    #Entropy
    EntropyMatrix = np.empty((1,1), np.ndarray);
    Gain = -1.0;
    Remainder = -1.0;
    #SubIndexList
    SubIndexList = [];
    IsLeaf = False;
    
    def __init__(self):
        ############Reload###############
        #Data
        self.OriginalFeatureIndex = -1;
        self.FeatureIndex = -1;
        self.FeatureName = '';
        self.Data = np.array([]);
        self.Target = np.array([]);
        self.SampleCount = 0;
        #ElementList
        self.FeatureElementDict = {};
        self.TargetElementDict = {};
        self.FeatureElementNames = [];
        self.TargetElementNames = [];
        #Entropy
        self.EntropyMatrix = np.empty((1,1), np.ndarray);
        self.Gain = -1.0;
        self.Remainder = -1.0;
        #SubIndexList
        self.SubIndexList = [];
        self.IsLeaf = False;
        ##########End of Reload###########
    
    def loadData(self, data, target):
        self.Data = data;
        self.Target = target;
        self.SampleCount = len(data);
        self._calculate();
        self._checkLeaf();
        return self;

    def _calculate(self):
        ####################################GetEntropyMatrix################################################
        elementLength = 0;
        targetLength = 0;
        self.EntropyMatrix[0,0] = [self.SampleCount];
        for i in range(len(self.Data)):
            if not self.FeatureElementDict.has_key(self.Data[i]):
                self.FeatureElementDict[self.Data[i]] = elementLength;
                self.FeatureElementNames.append(self.Data[i]);
                yL, xL = self.EntropyMatrix.shape;
                self.EntropyMatrix = np.r_[self.EntropyMatrix, np.empty((1, xL), np.ndarray)];
                elementLength += 1;

            if not self.TargetElementDict.has_key(self.Target[i]):
                self.TargetElementDict[self.Target[i]] = targetLength;
                self.TargetElementNames.append(self.Target[i]);
                yL, xL = self.EntropyMatrix.shape;
                self.EntropyMatrix = np.c_[self.EntropyMatrix, np.empty((yL, 1), np.ndarray)];
                targetLength += 1;

            if self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, 0] != None:
                self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, 0] = np.append(self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, 0], i);
            else:
                self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, 0] = np.array([i]);

            if self.EntropyMatrix[0, self.TargetElementDict[self.Target[i]] + 1] != None:
                self.EntropyMatrix[0, self.TargetElementDict[self.Target[i]] + 1] = np.append(self.EntropyMatrix[0, self.TargetElementDict[self.Target[i]] + 1], i);
            else:
                self.EntropyMatrix[0, self.TargetElementDict[self.Target[i]] + 1] = np.array([i]);

            if self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, self.TargetElementDict[self.Target[i]] + 1] != None:
                self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, self.TargetElementDict[self.Target[i]] + 1] = np.append(self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, self.TargetElementDict[self.Target[i]] + 1], i);
            else:
                self.EntropyMatrix[self.FeatureElementDict[self.Data[i]] + 1, self.TargetElementDict[self.Target[i]] + 1] = np.array([i]);
        
        #print self.EntropyMatrix;

        ############################################################CalculateGain################################################
        yL, xL = self.EntropyMatrix.shape;
        result = 0.0;
        IT = 0.0;
        R = 0.0;
        PT = 0.0;
        for i in range(1, xL):
            if self.EntropyMatrix[0, i] != None:
                PT = float(len(self.EntropyMatrix[0, i])) / float(self.SampleCount);
                IT += -(PT*math.log(PT, 2));
        for j in range(1, yL):
            T = 0.0;
            PE = float(len(self.EntropyMatrix[j, 0])) / float(self.SampleCount);
            for i in range(1, xL):
                if self.EntropyMatrix[j, i] != None:
                    P = float(len(self.EntropyMatrix[j, i])) / float(len(self.EntropyMatrix[j, 0]));
                    T += -(P*math.log(P, 2));
            R += PE * T;
        ######################################################SetInformationGain################################################
        self.Remainder = R;
        self.Gain = IT - R;

    def getSubIndexList(self, originalIndexList):
        for i in range(1, len(self.EntropyMatrix)):
            self.SubIndexList.append(originalIndexList[self.EntropyMatrix[i, 0]]);

    def _checkLeaf(self):
        if self.Gain == 0.0:
            v=list(self.TargetElementDict.values());
            k=list(self.TargetElementDict.keys());
            self.FeatureName = k[v.index(max(v))];
            self.IsLeaf = True;
        elif self.Gain < 0:
            self.IsLeaf = True;
        
class DecisionTree:
    MaxGainInformation = None;
    Default = 'Default';
    #Data
    N_Samples = 0;
    N_Features = 0;
    SampleIndexList = np.array([]);
    FeatureIndexList = np.array([]);
    OriginalData = np.array([]);
    OriginalTarget = np.array([]);
    OriginalFeatureNames = np.array([]);
    Data = np.array([]);
    Target = np.array([]);
    FeatureNames = np.array([]);
    #Tree
    Level = 0;
    IsLeaf = False;
    Parent = None;
    Childs = [];

    def __init__(self, attribs=None, default_v=None):
        ##############Reload################
        self.MaxGainInformation = Information();
        self.MaxGainInformation.FeatureName = self.Default;
        #Data
        self.N_Samples = 0;
        self.N_Features = 0;
        self.SampleIndexList = np.array([]);
        self.FeatureIndexList = np.array([]);
        self.OriginalData = np.array([]);
        self.OriginalTarget = np.array([]);
        self.OriginalFeatureNames = np.array([]);
        self.Data = np.array([]);
        self.Target = np.array([]);
        self.FeatureNames = np.array([]);
        #Tree
        self.Level = 0;
        self.IsLeaf = False;
        self.Parent = None;
        self.Childs = [];
        ##############End of Reload###########
        self.Default = default_v;
        self.OriginalFeatureNames = np.array(attribs);
    
    def fit(self, data, target):
        data = np.array(data);
        target = np.array(target);
        
        n_samples, n_features = data.shape;
        self._loadSubData(np.arange(0, n_samples), np.arange(0, n_features), data, target);
        self._makeTree();
        return self;

    def _loadSubData(self, sampleIndexList, featureIndexList, data, target):
        self.SampleIndexList = sampleIndexList;
        self.FeatureIndexList = featureIndexList;
        self.OriginalData = data;
        self.OriginalTarget = target;
        self.FeatureNames = self.OriginalFeatureNames[featureIndexList];
        self.Data = self.OriginalData[sampleIndexList];
        self.Data = self.Data[:, featureIndexList];
        self.Target = target[sampleIndexList];
        self.N_Samples, self.N_Features = self.Data.shape;
        self._caculateMaxGain();
        return self;

    def _caculateMaxGain(self):
        InformationList = [];
        GainList = [];
        for i in range(self.N_Features):
            Info = Information();
            Info.FeatureIndex = i;
            Info.FeatureName = self.FeatureNames[i];
            Info.loadData(self.Data[:, i], self.Target);
            if not Info.IsLeaf:
                Info.OriginalFeatureIndex = self.OriginalFeatureNames.tolist().index(Info.FeatureName);
            else:
                Info.OriginalFeatureIndex = -1;
            InformationList.append(Info);
            GainList.append(Info.Gain);
        self.MaxGainInformation = InformationList[GainList.index(max(GainList))];
        print self.MaxGainInformation.FeatureName
        self.MaxGainInformation.getSubIndexList(self.SampleIndexList);
        if self.MaxGainInformation.Gain == 0.0:
            self.IsLeaf = True;

    def _makeTree(self):
        if not self.IsLeaf:
            SubLength = len(self.MaxGainInformation.SubIndexList);
            if SubLength != 0:
                for i in range(SubLength):
                    TreeNode = DecisionTree(self.OriginalFeatureNames, self.Default);
                    #Load Data after removing the max gain feature and breaking into sub dataset by each value of the feature.
                    if len(self.FeatureIndexList) > 1:
                        TreeNode._loadSubData(self.MaxGainInformation.SubIndexList[i], np.delete(self.FeatureIndexList, self.MaxGainInformation.FeatureIndex), self.OriginalData, self.OriginalTarget);
                    elif len(self.FeatureIndexList) == 1:
                        TreeNode.IsLeaf = True;
                    TreeNode.Level = self.Level + 1;
                    TreeNode.Parent = self;
                    TreeNode.Default = self.Default;
                    self.Childs.append(TreeNode);
                    TreeNode._makeTree();

    def printTree(self):
        if self.MaxGainInformation.FeatureName == 'temperature':
            print self.MaxGainInformation.EntropyMatrix;
        if self.Level == 0: 
            print (' ' * 5 * self.Level + self.MaxGainInformation.FeatureName).ljust(40) + ' Max gain = ' + str(self.MaxGainInformation.Gain);
            #print self.MaxGainInformation.SubIndexList;
        for i in range(len(self.Childs)):
            print (' ' * 5 * self.Level + '{ ' + str(self.MaxGainInformation.FeatureElementNames[i]) + ' } => ' + self.Childs[i].MaxGainInformation.FeatureName).ljust(40) + ' Max gain = ' + str(self.Childs[i].MaxGainInformation.Gain);
            #print self.Childs[i].MaxGainInformation.SubIndexList;
            self.Childs[i].printTree();

    def setDefault(self, default):
        self.Default = default;

    def predict(self, data):
        data = np.array(data);
        result = [];
        n_samples, n_features = data.shape;
        for j in range(n_samples):
            targetFound = False;
            Node = self;
            while (not Node.IsLeaf) and (not targetFound):
                featureElement = data[j, Node.MaxGainInformation.OriginalFeatureIndex];
                if not Node.MaxGainInformation.FeatureElementDict.has_key(featureElement):
                    targetFound = True;
                else:
                    SubIndex = Node.MaxGainInformation.FeatureElementDict[featureElement];
                    Node = Node.Childs[SubIndex];
            if targetFound:
                result.append(self.Default);
            else:
                result.append(Node.MaxGainInformation.FeatureName);
        return result;

def Main():

    #I = Information();
    #I.loadData(np.array(dataset['data'])[:, 3], dataset['target'])

    #print I.Gain

    
    #will need some args in constructor
    tree = DecisionTree(dataset['feature_names'], dataset['target'][0]);
    tree.fit(dataset['data'], dataset['target']);
    tree.printTree();
    #test on training data
    print tree.predict(dataset['data']);
    

Main()
#from timeit import Timer
#t = Timer(lambda: Main())
#print 'Running time: ' + str(t.timeit(number=1));