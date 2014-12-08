import argparse
from sklearn.externals import joblib
import mlUtil
import decisionTree

def k_fold_eval(traind, k):
    '''See the Word doc for specifications
    '''
    "*** YOUR CODE HERE ***"
    pass

if __name__ == '__main__':
    #parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="Name of file with training data", type=str)
    parser.add_argument("-k", help="number of folds", type=int, default=5)
    parser.add_argument("--ibm", help="Flag to indicate that input is IBM data, else plain CSV", action="store_true")
    args = parser.parse_args()

    #for you to add is logic for handling the --y_col flag if given (for tennis, for example)
    if args.ibm:
        data = joblib.load(args.train_file)
    else:
        data = mlUtil.extract_data(args.train_file)
    data = mlUtil.enhance_data(data)

    print k_fold_eval(data, args.k)




