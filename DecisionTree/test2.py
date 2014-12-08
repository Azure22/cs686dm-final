import numpy as np;
import math;
import mlUtil;

dataset = mlUtil.extract_data('restaurant.csv');

print 'Load complete';
print '';

class Information:
    #Features & Samples
    FeatureName = '';
    FeatureIndex = -1;
    SampleIndexList = np.array([]);
    SampleCount = 0;
    #Data & Target
    Data = np.array([]);
    Target = np.array([]);
    #ElementList
    DataValueDict = {};
    TargetValueDict = {};
    DataValueNames = [];
    TargetValueNames = [];
    #Entropy
    EntropyMatrix = np.empty((1,1), np.ndarray);
    Gain = -1.0;
    Remainder = -1.0;
    #SubIndexList & Leaf
    SubSampleIndexList = [];
    IsLeaf = False;

    def __init__(self):
        ############Reload###############
        #Feature & Sample
        self.FeatureIndex = -1;
        self.FeatureName = '';
        self.SampleIndexList = np.array([]);
        self.SampleCount = 0;
        #Data & Target
        self.Data = np.array([]);
        self.Target = np.array([]);
        #ElementList
        self.DataValueDict = {};
        self.TargetValueDict = {};
        self.DataValueNames = [];
        self.TargetValueNames = [];
        #Entropy
        self.EntropyMatrix = np.empty((1,1), np.ndarray);
        self.Gain = -1.0;
        self.Remainder = -1.0;
        #SubSampleIndexList
        self.SubSampleIndexList = [];
        self.IsLeaf = False;
        ##########End of Reload###########

    def loadData(self, data, target, sampleIndexList, featureIndex):
        self.FeatureIndex = featureIndex;
        self.SampleIndexList = sampleIndexList;
        self.SampleCount = len(sampleIndexList);
        self.Data = np.array(data)[sampleIndexList][:, featureIndex];
        self.Target = np.array(target)[sampleIndexList];
        self._calculate();
        self._checkLeaf();

    def _calculate(self):
        ####################################GetEntropyMatrix################################################
        dataValueIndex = 0;
        targetValueIndex = 0;
        self.EntropyMatrix[0,0] = [self.SampleCount];
        for i in range(self.SampleCount):
            if self.Data[i] == 'French':
                A = 0;
            if not self.DataValueDict.has_key(self.Data[i]):
                self.DataValueDict[self.Data[i]] = dataValueIndex;
                self.DataValueNames.append(self.Data[i]);
                yL, xL = self.EntropyMatrix.shape;
                self.EntropyMatrix = np.r_[self.EntropyMatrix, np.empty((1, xL), np.ndarray)];
                dataValueIndex += 1;

            if not self.TargetValueDict.has_key(self.Target[i]):
                self.TargetValueDict[self.Target[i]] = targetValueIndex;
                self.TargetValueNames.append(self.Target[i]);
                yL, xL = self.EntropyMatrix.shape;
                self.EntropyMatrix = np.c_[self.EntropyMatrix, np.empty((yL, 1), np.ndarray)];
                targetValueIndex += 1;

            if self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, 0] != None:
                self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, 0] = np.append(self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, 0], self.SampleIndexList[i]);
            else:
                self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, 0] = np.array([self.SampleIndexList[i]]);

            if self.EntropyMatrix[0, self.TargetValueDict[self.Target[i]] + 1] != None:
                self.EntropyMatrix[0, self.TargetValueDict[self.Target[i]] + 1] = np.append(self.EntropyMatrix[0, self.TargetValueDict[self.Target[i]] + 1], self.SampleIndexList[i]);
            else:
                self.EntropyMatrix[0, self.TargetValueDict[self.Target[i]] + 1] = np.array([self.SampleIndexList[i]]);

            if self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, self.TargetValueDict[self.Target[i]] + 1] != None:
                self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, self.TargetValueDict[self.Target[i]] + 1] = np.append(self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, self.TargetValueDict[self.Target[i]] + 1], self.SampleIndexList[i]);
            else:
                self.EntropyMatrix[self.DataValueDict[self.Data[i]] + 1, self.TargetValueDict[self.Target[i]] + 1] = np.array([self.SampleIndexList[i]]);
        
        #print self.EntropyMatrix;
        #print self.DataValueNames;
        #print self.TargetValueNames;

        ############################################################CalculateGain################################################
        yL, xL = self.EntropyMatrix.shape;
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
                    DVC = len(self.EntropyMatrix[j, i]);
                    #print 'Data Value Count = ' + str(DVC);
                    TVC = len(self.EntropyMatrix[j, 0]);
                    #print 'Target Value Count = ' + str(TVC);
                    P = float(DVC) / float(TVC);
                    #print 'P = ' + str(P);
                    T += -(P*math.log(P, 2));
            R += PE * T;
        self.Remainder = R;
        self.Gain = IT - R;

        ####################################################SetSubSampleIndexList###############################################
        for i in range(1, len(self.EntropyMatrix)):
            self.SubSampleIndexList.append(self.EntropyMatrix[i, 0]);
        #print self.SubSampleIndexList;

    def _checkLeaf(self):
        if self.Gain <= 0.0:
            self.IsLeaf = True;
            v=list(self.TargetValueDict.values());
            k=list(self.TargetValueDict.keys());
            self.FeatureName = k[v.index(max(v))];

class DecisionTree:
    MaxGainInformation = Information();
    Default = 'Default';
    #Data
    N_Samples = 0;
    N_Features = 0;
    SampleIndexList = np.array([]);
    FeatureIndexList = np.array([]);
    Data = np.array([]);
    Target = np.array([]);
    FeatureNames = np.array([]);
    #Tree
    Level = 0;
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
        self.Data = np.array([]);
        self.Target = np.array([]);
        self.FeatureNames = np.array([]);
        #Tree
        self.Level = 0;
        self.Parent = None;
        self.Childs = [];
        ##############End of Reload###########
        self.Default = default_v;
        self.FeatureNames = np.array(attribs);

    def _loadData(self, data, target, sampleIndexList, featureIndexList):
        self.SampleIndexList = sampleIndexList;
        self.FeatureIndexList = featureIndexList;
        self.Data = np.array(data);
        self.Target = np.array(target);
        self.N_Samples = len(sampleIndexList);
        self.N_Features = len(featureIndexList);
        self._caculateMaxGain();

    def _caculateMaxGain(self):
        InformationList = [];
        GainList = [];
        for i in range(self.N_Features):
            Info = Information();
            Info.loadData(self.Data, self.Target, self.SampleIndexList, self.FeatureIndexList[i]);
            if not Info.IsLeaf:
                Info.FeatureName = self.FeatureNames[Info.FeatureIndex];
            InformationList.append(Info);
            GainList.append(Info.Gain);
        self.MaxGainInformation = InformationList[GainList.index(max(GainList))];

    def _makeTree(self):
        if not self.MaxGainInformation.IsLeaf:
            for i in range(len(self.MaxGainInformation.SubSampleIndexList)):
                TreeNode = DecisionTree(self.FeatureNames, self.Default);
                #Load Data after removing the max gain feature and breaking into sub dataset by each value of the feature.
                TreeNode._loadData(self.Data, self.Target, self.MaxGainInformation.SubSampleIndexList[i], np.delete(self.FeatureIndexList, self.FeatureIndexList.tolist().index(self.MaxGainInformation.FeatureIndex)));
                TreeNode.Level = self.Level + 1;
                TreeNode.Parent = self;
                self.Childs.append(TreeNode);
                TreeNode._makeTree();

    def printTree(self):
        if self.Level == 0: 
            print (' ' * 5 * self.Level + self.MaxGainInformation.FeatureName).ljust(40) + ' Max gain = ' + str(self.MaxGainInformation.Gain);
            #print self.MaxGainInformation.SubIndexList;
        for i in range(len(self.Childs)):
            print (' ' * 5 * self.Level + '{ ' + str(self.MaxGainInformation.DataValueNames[i]) + ' } => ' + self.Childs[i].MaxGainInformation.FeatureName).ljust(40) + ' Max gain = ' + str(self.Childs[i].MaxGainInformation.Gain);
            #print self.Childs[i].MaxGainInformation.SubIndexList;
            self.Childs[i].printTree();

    def fit(self, data, target):
        n_samples, n_features = np.array(data).shape;
        self._loadData(data, target, np.arange(0, n_samples), np.arange(0, n_features));
        self._makeTree();
        return self;

    def predict(self, data):
        data = np.array(data);
        result = [];
        n_samples, n_features = data.shape;
        for j in range(n_samples):
            targetFound = False;
            Node = self;
            while (not Node.MaxGainInformation.IsLeaf) and (not targetFound):
                dataValue = data[j, Node.MaxGainInformation.FeatureIndex];
                if not Node.MaxGainInformation.DataValueDict.has_key(dataValue):
                    targetFound = True;
                else:
                    SubIndex = Node.MaxGainInformation.DataValueDict[dataValue];
                    Node = Node.Childs[SubIndex];
            if targetFound:
                result.append(self.Default);
            else:
                result.append(Node.MaxGainInformation.FeatureName);
        return result;

D = DecisionTree(dataset['feature_names'], 'default');
D.fit(dataset['data'], dataset['target']);
D.printTree();
print dataset['target'];
print D.predict(dataset['data']);

