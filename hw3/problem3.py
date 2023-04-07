import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # np.log will display warnings sometimes, this is to beautify the display

traindata = pd.read_csv(r'train.csv')

# We shall remove PassengerId, Name, Ticket, and Cabin from each of the datasets where applicable, since these attributes are less important in the grand scheme of this tree

traindata.drop(traindata.columns[[0, 3, 8, 10]], axis=1, inplace=True) # PassengerId, Name, Ticket, and Cabin gone

# It's probably within our best interests to smooth out some of the data into labels rather than straight numbers
# Features that would benefit from this smoothing are Age, Fare, SibSp, and Parch

def binarize(x):    # works well for SibSp and Parch. Rather than how many, it's did they have any of the feature in question
    if x > 0:
        return 1
    else: 
        return 0
    
def agesort(x):     # allows for cleaner labeling and easier decisionmaking
    if x <= 12:
        return 0 # child
    elif x <= 19:
        return 1 # teen
    elif x <= 65:
        return 2 # adult
    else:
        return 3 # elderly

def sexsort(x):
    if x == "male":
        return 0
    else:
        return 1
    
mean = np.mean(traindata['Fare'].to_numpy())

def pricend(x):     # returns whether fare was low or high depending on the mean of the training set
    if x <= mean:
        return 0 # low fare
    else:
        return 1 # high fare

def embsort(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2 # Q

def modify(dset, col, func, default):
    for i in range(len(dset.index)):
        if dset.at[i, col] == 'NaN':
            dset.at[i, col] = default
        else: 
            dset.at[i, col] = func(dset.at[i, col])

modify(traindata, "Age", agesort, 2)
modify(traindata, "Fare", pricend, 0)
modify(traindata, "SibSp", binarize, 0)
modify(traindata, "Parch", binarize, 0)
modify(traindata, "Sex", sexsort, 0)
modify(traindata, "Embarked", embsort, 0)

train = traindata[0:((traindata.shape[0]//4))*3] 
test = traindata[(((traindata.shape[0]//4))*3)+1:-1]

def Hyx(dict): # Conditional Entropy
    result = []
    negtotal = np.sum(dict[0])
    postotal = np.sum(dict[1])
    total = np.sum(dict)
    for i in range(len(dict[0])):
        pxj = -(dict[0][i] / negtotal) * (dict[0][i] / total) * np.log((dict[0][i] / total)) 
        pyj = -(dict[1][i] / postotal) * (dict[1][i] / total) * np.log((dict[1][i] / total))
        result.append(pxj + pyj)
    return np.sum(result)

def Hy(dict): # Entropy
    negtotal = np.sum(dict[0])
    postotal = np.sum(dict[1])
    total = np.sum(dict)
    pxj = -(negtotal/total)*np.log(negtotal/total)
    pyj = -(postotal/total)*np.log(postotal/total)
    return pxj+pyj

def ig(dict): # Information Gain
    return Hy(dict) - Hyx(dict)

def gini(dict): # Gini Index
    total = np.sum(dict)
    result = [((dict[0][i]+dict[1][i])/total)**2 for i in range(len(dict[0]))]
    return 1 - np.sum(result)

def sorthelp(e):
    return e[0]

def ordering(data, fun):
    # Exist as templates 
    labels = {"Pclass": [0, 0, 0], "Sex": [0, 0], "Age": [0, 0, 0, 0], "SibSp": [0, 0], "Parch": [0, 0], "Fare": [0, 0], "Embarked": [0, 0, 0]}
    resDict = []
    labelnames = []
    for i in (data.drop(columns="Survived")).columns:
        template = labels[i]
        resDict.append([template, template]) # Left is negative (0) cases and right is positive (1) cases
        labelnames.append(i)
    for i in range(len(data.index)): # used to count how many of each possible value for each label in the given dataset
        idx = int(data.at[i, "Survived"])
        for j in range(len(labelnames)):
            if labelnames[j] == "Pclass":
                resDict[j][idx][int(data.at[i, labelnames[j]]-1)] += 1 # this case is specifically to make Pclass work
            else:
                resDict[j][idx][int(data.at[i, labelnames[j]])] += 1
    order = []
    for i in range(len(resDict)):
        order.append([fun(resDict[i]), labelnames[i]])
    order.sort(key = sorthelp) # to get the label while still sorting by lowest entropy/value
    return order[0]
    
class node:
    def __init__(self, feature, split, v=None, root=False):
        self.f = feature
        self.s = split
        self.v = v
        self.r = root
        self.y = len(feature[feature["Survived"]==1].index)
        self.n = len(feature[feature["Survived"]==0].index)
        self.children = []
        self.p = self.pred()
    
    def __repr__(self):
        return f"<{self.s},{self.v}>|<{self.p}>---[y: {self.y}, n: {self.n}]---/{self.children}/"
    
    def is_leaf(self):
        if self.children != []:
            return False
        else:
            return True
    
    def add(self, node):
        self.children.append(node)
    
    def pred(self):
        if self.r == True:
            return None
        else:
            if self.y > self.n:
                return 1
            else:
                return 0

class tree:
    def __init__(self, set):
        self.root = node(set, None, True)
        self.labels = {"Pclass": [1, 2, 3], "Sex": [0, 1], "Age": [0, 1, 2, 3], "SibSp": [0, 1], "Parch": [0, 1], "Fare": [0, 1], "Embarked": [0, 1, 2]}
    
    def build(self, nnode, fun): # Recursively builds the tree node by node applying the attribute selection on each node for the next best split
        if nnode != self.root and ((nnode.y == 0 or nnode.n == 0) or len(((nnode.f).drop(columns="Survived")).columns)<1):
            return 1
        split = ordering(nnode.f, fun)[1]
        for i in self.labels[split]:
            subset = nnode.f[nnode.f[split] == i]
            subset.drop(columns=[split], inplace= True)
            new = node(subset.reset_index(drop=True), split, i)
            nnode.add(new)
            self.build(new, fun)
    
    def __repr__(self):
        return self.root.__repr__()
    
    def testing(self, set):
        set = set.reset_index(drop=True)
        expected = set.pop("Survived")
        expected = expected.to_numpy()
        results = []
        for i in range(len(set.index)): # traverses the tree to find matches and "predict" the result
            current = self.root
            while current.is_leaf() != True:
                for j in current.children:
                    split = j.s
                    val = j.v
                    comp = set.at[i, split]
                    if int(comp) == int(val):
                        current = j
            results.append(current.pred())
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for k in range(len(results)):
            if expected[k] - results[k] == 0:
                if expected[k] + results[k] == 2:
                    tp += 1
                else:
                    tn += 1
            elif expected[k] - results[k] == -1:
                fp += 1
            else:
                fn += 1
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return recall, precision, accuracy
            
tt = tree(train)
tt.build(tt.root, Hyx)
tt_recall, tt_precision, tt_accuracy = tt.testing(test)
print("Tree built with Cross Entropy:")
print(f"Recall: {tt_recall}, Precision: {tt_precision}, Accuracy: {tt_accuracy}\n")
igt = tree(train)
igt.build(igt.root, ig)
igt_recall, igt_precision, igt_accuracy = igt.testing(test)
print("Tree built with Information Gain:")
print(f"Recall: {igt_recall}, Precision: {igt_precision}, Accuracy: {igt_accuracy}\n")
ginit = tree(train)
ginit.build(ginit.root, gini)
ginit_recall, ginit_precision, ginit_accuracy = ginit.testing(test)
print("Tree built with Gini Index:")
print(f"Recall: {ginit_recall}, Precision: {ginit_precision}, Accuracy: {ginit_accuracy}\n")

# It seems that overall information gain works the best for this dataset and tree, as it has the highest Accuracy and Precision.
# The 2nd place trophy goes to Gini Index, as it has the lowest Recall among the three while still having Accuracy and Precision
# comparable to Cross Entropy. Of course, last place goes to standard Cross-Entropy selection of attributes.