import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

rootPath = './'
tempPath = './'
dataPath = rootPath + '/data/'
csvPath  = rootPath + '/csvfiles/'
dsetPath = tempPath + '/dataset/'
logsPath = tempPath + '/logs/'

# print setting.
pd.options.display.max_columns = None
pd.options.display.max_rows = None

def main():
    if not os.path.exists(dsetPath + '/features.csv'):
        if not os.path.exists(dsetPath + '/filelist.csv'):
            filelist = ExtractFileList(dataPath)
        else:
            filelist = pd.read_csv(dsetPath + '/filelist.csv')
        csvfeat = GetCSVFeatures(csvPath)
        features = MatchFeatures(filelist, csvfeat)
    else:
        features = pd.read_csv(dsetPath + '/features.csv')

    Y = features['label']
    X = features.drop(columns=['folder', 'name', 'label'])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, shuffle=True)
    print("size of training data is", XTrain.shape)
    print("size of testing data is", XTest.shape)

    # NaiveBayes.
    model = GaussianNB(priors=[0.672, 0.328])
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    Evaluation(YTest, YPred, 'Naive Bayes')

    # Least Square Regression.
    model = LinearRegression(normalize=True)
    model.fit(XTrain, YTrain)
    YReg = model.predict(XTest)
    YPred = (YReg > 0.5)
    Evaluation(YTest, YPred, 'Least Square Regression')
    print('Mean Squared Error: %.2f' % mean_squared_error(YTest, YReg))
    print('Coefficient of determination: %.2f' % r2_score(YTest, YReg))

    # Logistic Regression.
    model = LogisticRegression( penalty='l1', C=100)
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    Evaluation(YTest, YPred, 'Logistic Regression')

    # Decision Tree.
    model = DecisionTreeClassifier(criterion="gini", splitter="best", max_features=None, min_samples_split=300, min_samples_leaf=1, random_state=0)
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    Evaluation(YTest, YPred, 'Decision Tree')

    # Random Forest.
    model = RandomForestClassifier(n_estimators=1000, max_depth=None, criterion='gini', max_features='sqrt', n_jobs=-1, verbose=1, random_state=0)
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    Evaluation(YTest, YPred, 'Random Forest')

    # Support Vector Machine.
    model = SVC(kernel='rbf', C=1)
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    Evaluation(YTest, YPred, 'Support Vector Machine')

    # Neural Network.
    model = MLPClassifier(hidden_layer_sizes=(16, 4), activation='relu', solver='lbfgs', max_iter=100)
    model.fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    Evaluation(YTest, YPred, 'Neural Network')

    return 0

def ExtractFileList(path):
    foldername = []
    filename = []
    label = []

    for root, _, fs in os.walk(path + '/negatives/'):
        for file in fs:
            _, folder = os.path.split(root)
            foldername.append(folder)
            filename.append(file)
            label.append(0)

    for root, _, fs in os.walk(path + '/positives/'):
        for file in fs:
            _, folder = os.path.split(root)
            foldername.append(folder)
            filename.append(file)
            label.append(1)

    for root, _, fs in os.walk(path + '/security_patch/'):
        for file in fs:
            foldername.append('security_patch')
            filename.append(file)
            label.append(1)

    df = pd.DataFrame(list(zip(foldername, filename, label)), columns=['folder', 'name', 'label'])
    if not os.path.exists(dsetPath): os.makedirs(dsetPath)
    df.to_csv(dsetPath + '/filelist.csv', index=0)

    return df

def GetCSVFeatures(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        tmp = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df, tmp])

    df = df.drop(columns = ['Unnamed: 0'])
    df.reset_index(drop=True, inplace=True)

    for i in range(df.shape[0]):
        _, df.at[i, 'name'] = os.path.split(df.at[i, 'name'])

    return df

def MatchFeatures(flist, feat):

    df = pd.merge(flist, feat, on='name')

    if not os.path.exists(dsetPath): os.makedirs(dsetPath)
    df.to_csv(dsetPath + '/features.csv', index=0)

    return df

def Evaluation(YTest, YPred, method=''):
    print('====================== ' + method + ' ======================')
    acc = accuracy_score(YTest, YPred) * 100
    print('Accuracy is %.3f%%.' % acc)
    conf = confusion_matrix(YTest, YPred)
    print(pd.DataFrame(conf, columns=['Pred-Neg','Pred-Pos'], index=['Actl-Neg', 'Actl-Pos']))
    precision = conf[1][1] / (conf[0][1] + conf[1][1]) if (conf[0][1] + conf[1][1]) else 0
    recall = conf[1][1] / (conf[1][0] + conf[1][1]) if (conf[1][0] + conf[1][1]) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print('Precision is %.3f.' % precision)
    print('Recall is %.3f.' % recall)
    print('F1 score is %.3f.' % F1)

class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

if __name__ == '__main__':
    logfile = 'patch_ml.txt'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    main()