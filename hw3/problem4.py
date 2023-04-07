import pandas as pd
import numpy as np
import random
import sklearn.tree as sk
import matplotlib.pyplot as mp
from sklearn import metrics

traindata = pd.read_csv(r'train.csv')
testdata = pd.read_csv(r'test.csv')
testclass = pd.read_csv(r'gender_submission.csv')

# We shall remove PassengerId, Name, Ticket, and Cabin from each of the datasets where applicable, since these attributes are less important in the grand scheme of this tree

traindata.drop(traindata.columns[[0, 3, 8, 10]], axis=1, inplace=True) # PassengerId, Name, Ticket, and Cabin gone
testdata.drop(testdata.columns[[0, 2, 7, 9]], axis=1, inplace=True)
testdata["Survived"] = testclass.pop("Survived")

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
modify(testdata, "Age", agesort, 2)
modify(testdata, "Fare", pricend, 0)
modify(testdata, "SibSp", binarize, 0)
modify(testdata, "Parch", binarize, 0)
modify(testdata, "Sex", sexsort, 0)
modify(testdata, "Embarked", embsort, 0)

def adaboost (iterations):
    trweights = [1/len(traindata.index) for i in range(len(traindata.index))] # creating weight vectors
    tweights = [1/len(testdata.index) for i in range(len(testdata.index))]
    training_errors = []
    test_errors = []
    accuracy = []
    random.seed(4)
    for i in range(iterations):
        train = traindata.sample(len(traindata), replace=True, weights=trweights) # sampling from datasets
        test = testdata.sample(len(testdata), replace=True, weights=tweights)
        trainclass = train.pop("Survived") # removing output column after shuffling to preserve I/O matching
        testclass = test.pop("Survived")
        giniclf = sk.DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth = 1) # make stump
        clf = giniclf.fit(train, trainclass)
        trpred = clf.predict(train)
        tepred = clf.predict(test)
        trE = 0.5 - 0.5*sum(trainclass * trweights * trpred)
        tE = 0.5 - 0.5 * sum(testclass * tweights * tepred)
        training_errors.append(trE)
        test_errors.append(tE)
        accuracy.append(metrics.accuracy_score(testclass, tepred))
        tralpha = 0.5*np.log((1-trE)/trE)
        talpha = 0.5*np.log((1-tE)/tE)
        trweights = trweights * np.exp(-1 * trainclass.to_numpy() * tralpha * trpred)
        tweights = tweights * np.exp(-1 * testclass.to_numpy() * talpha * tepred)
        trz = sum(trweights)
        tz = sum(tweights)
        trweights = trweights/trz
        tweights = tweights/tz
    return training_errors, test_errors, [min(accuracy), accuracy.index(min(accuracy))]

training, testing, accuracy = adaboost(500)
print(f"Best Accuracy on Train Data: {accuracy[0]} at Iteration {accuracy[1]}")
mp.plot([i for i in range(500)], training, "r-") # red curve
mp.plot([i for i in range(500)], testing, "b-") # blue curve
mp.show()

# Strangely enough, the training errors dont really seem to be improving, just oscillating around a certain error value.
# The test errors on the other hand, show a much better decrease and generally improves with each iteration.
# The error decreases to its lowest point somewhere in the early-to-mid 400s, but its accuracy is usually best towards the
# later end of the iterations.

