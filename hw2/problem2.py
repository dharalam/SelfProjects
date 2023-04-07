import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.utils import shuffle
import math
import torch

'''
Author: Dimitrios Haralampopoulos
Pledge: I pledge my honor that I have abided by the Stevens Honor System
Overview: With the dataset given for this problem, I predict the likeliness of a patient having breast cancer (2 - negative, 4 - positive) given 9 features of importance using
Mini-Batch Gradient Descent, Stochastic Gradient Descent, and a Probabilistic Generative Model using the Maximum Likelihood Solution.
'''

bcancer = pd.read_csv (r'breast-cancer-wisconsin.csv')  # Retrieving features from set
bcdata = bcancer.values # Creating Datasets for testing and training
for i in range(len(bcdata)):
    for j in range(len(bcdata[i])-1):
        if (bcdata[i])[j] == '?':
            (bcdata[i])[j] = 0
        (bcdata[i])[j] = int((bcdata[i])[j])
    bcdata[i] = torch.from_numpy(np.asarray(bcdata[i], dtype=int))

# Splitting data accordingly, a quarter of the data will be used for testing while the remaining 3/4 will be used for training. Just convention to have that kind of proportion of test/training data
bX_train, bX_test, bY_train, bY_test = train_test_split(bcdata[:, 1:10], bcdata[:, -1], test_size = 0.25, random_state=1)

# Shuffling data around pre-training
bX_shuffle, bY_shuffle = shuffle(bcdata[:, 1:10], bcdata[:, -1])
        
w = np.random.randn(9,1)
b = np.copy(w)
recall = 0
precision = 0
accuracy = 0

# Creating paired training datasets for unpacking during gradient descent
d = []
for x,y in zip(bX_train, bY_train):   
    d.append((torch.from_numpy(np.asarray(x, dtype=np.float16)), y))

def sigmoid(x):
    return 1/(1+np.exp(-x))

# gradient of Logistic Regression + ML Estimator
def lrg (y_i, x_i, beta):
    return np.dot((sigmoid(np.dot(beta.T, x_i.T)) - y_i.T), x_i).T

def minibatch (D, step, epochs, batch, alpha = 1.0):
    global w
    steps = step
    for i in range(epochs):
        #I have to shuffle all the data yet keep the inputs still matched with the outputs, so a single dataset solution seemed most appropriate
        trainloader = DataLoader(dataset = D, batch_size = batch, shuffle= True) 
        for x, y in trainloader:
            xt = (x.detach().cpu().numpy())     #Split x and y and convert them from tensors to numpy arrays
            yt = (y.detach().cpu().numpy())
            lfun = lrg(yt, xt, w)
            if math.isclose(np.mean(w), 0, abs_tol=step):   # stopping condition when weight vector is close enough to 0
                return w
            w = w - (steps * lfun)   #updating our weight vector here
            steps = steps/10
    return w

def stochastic (D, step, epochs, alpha = 1.0):
    global b
    steps = step
    for i in range(epochs):
        #I have to shuffle all the data yet keep the inputs still matched with the outputs, so a single dataset solution seemed most appropriate
        trainloader = DataLoader(dataset = D, batch_size = 1, shuffle= True) 
        for x, y in trainloader:
            xt = (x.detach().cpu().numpy())     #Split x and y and convert them from tensors to numpy arrays
            yt = (y.detach().cpu().numpy())
            lfun = lrg(yt, xt, b)     #This produces our loss function with regularization term added
            if math.isclose(np.mean(b), 0, abs_tol=step):   # stopping condition when weight vector is close enough to 0
                return b
            b = b - (steps * lfun)   #updating our weight vector here
            steps = steps/10
    return b

# Short functions for repeated learning/training
def learning_batch (D, step, epochs, batch, gen, alpha = 1.0):
    for i in range(gen):
        minibatch(D, step, epochs, batch, alpha)
        print(f"Learning dataset: bcancer (Batch) . . . generation: {i+1}")
               
def learning_stoch (H, step, epochs, gen, alpha = 1.0):
    for j in range(gen):
        stochastic(H, step, epochs, alpha)
        print(f"Learning dataset: bcancer (Stoch) . . . generation: {j+1}")

def predicting_batch (D):
    global w   
    decisions = []
    predictions = np.dot(w.T, D.T).T    # makes predictions
    m = np.mean(predictions)
    for i in predictions:   # applies decision boundary
        if i > m:
            decisions.append(4)
        else:
            decisions.append(2)
    return decisions

def predicting_stoch (H):
    global b
    decisions = []
    predictions = np.dot(b.T, H.T).T    # makes predictions
    m = np.mean(predictions)
    for i in predictions:   # applies decision boundary
        if i > m:
            decisions.append(4)
        else:
            decisions.append(2)
    return decisions

# basically a wrapper function
def fpredicting_batch (D, T): 
    global w                                    
    print(f'Prediction for Mini-Batch: {predicting_batch(D)}\nActual: {T}')
    return "Success"

def fpredicting_stoch (H, T):
    global b
    print(f'Prediction for Stochastic: {predicting_stoch(H)}\nActual: {T}')
    return "Success"

# Here we define our cross validation (Repeated K-Fold in this case). We have defined splits as 10 and repeats as 3 somewhat arbitrarily, though they are commonly used parameters for it.
# We use 30 different model fittings in this case then to estimate the model efficacy
dcv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state= 0)
hcv = RepeatedKFold(n_splits= 10, n_repeats= 3, random_state = 0)

def batch_rkf(step, epochs, batch, alpha = 1.0):
    global w
    v = np.copy(w) # copy of w for restoration
    results = []
    for train_x, test_x in dcv.split(bX_shuffle): # this is where we make the actual k-fold
        d = []
        dx = []
        dy = []
        tx = []
        ty = []
        for x in train_x:   # builds the components of training set
            dx.append(bX_shuffle[x])
            dy.append(bY_shuffle[x])
        for y in test_x:    # creates the testing sets
            tx.append(bX_shuffle[y])
            ty.append(bY_shuffle[y])
        for x,y in zip(dx, dy):     # builds the training set from the components
            d.append((torch.from_numpy(np.asarray(x, dtype=np.float64)), y))
        minibatch(d, step, epochs, batch)  # mini-batch to train the model
        tx = np.array(tx)   # converting to np.array for compatibility with the math performed in the prediction
        ty = np.array(ty)
        pred = predicting_batch(tx)
        mse = 0
        for i in range(len(pred)):
            mse += (ty[i] - pred[i])**2
        results.append(np.mean(mse))         
    w = np.copy(v)  # resets w
    return np.mean(results)     # mean of MSEs is computed and we return that as our cross-validation with MSE scoring

def stoch_rkf(step, epochs):
    global b
    v = np.copy(b) # copy of b for restoration
    results = []
    for train_x, test_x in dcv.split(bX_shuffle): # this is where we make the actual k-fold
        d = []
        dx = []
        dy = []
        tx = []
        ty = []
        for x in train_x:   # builds the components of training set
            dx.append(bX_shuffle[x])
            dy.append(bY_shuffle[x])
        for y in test_x:    # creates the testing sets
            tx.append(bX_shuffle[y])
            ty.append(bY_shuffle[y])
        for x,y in zip(dx, dy):     # builds the training set from the components
            d.append((torch.from_numpy(np.asarray(x, dtype=np.float64)), y))
        stochastic(d, step, epochs)  # stochastic to train the model
        tx = np.array(tx)   # converting to np.array for compatibility with the math performed in the prediction
        ty = np.array(ty)
        pred = predicting_stoch(tx)
        mse = 0
        for i in range(len(pred)):
            mse += (ty[i] - pred[i])**2
        results.append(np.mean(mse))         
    b = np.copy(v)  # resets b
    return np.mean(results)     # mean of MSEs is computed and we return that as our cross-validation with MSE scoring

def rcrossval_batch (step, epochs, batch, gen):     # works quickly
    bcvl = []
    for i in range(gen):
        print(f"Validating {i+1} of {gen}...")
        bcvl.append(batch_rkf(step, epochs, batch))
    return np.mean(bcvl)

def rcrossval_stoch (step, epochs, gen):    # works a lot slower for some reason, introduced print statements to show that it is indeed working
    scvl = []
    for i in range(gen):
        print(f"Validating {i+1} of {gen}...")
        scvl.append(stoch_rkf(step, epochs))
    return np.mean(scvl)

#Learning . . .
learning_batch(d, 0.1, 10, 50, 5, 1.0)
learning_stoch(d, 0.1, 10, 5, 1.0)

# Predicting . . .
print(f"{fpredicting_batch(bX_test, bY_test)}")
print(f"{fpredicting_stoch(bX_test, bY_test)}")
# Cross Validating . . .
print("Beginning Cross Validation for Batch")
bcv = rcrossval_batch(0.1, 10, 50, 5)
print(f"Cross validation for Batch: {bcv}")
print("Beginning Cross Validation for Stoch")
scv = rcrossval_stoch(0.1, 10, 5)
print(f"Cross validation for Stoch: {scv}")

tp = 0
tn = 0
fp = 0
fn = 0
def rpa (bcv, scv):
    global tp, tn, fp, fn, recall, precision, accuracy
    pred = []
    if bcv < scv:   # This selects the best model in any situation
        pred = predicting_batch(bX_test)
        best = "Mini-Batch Gradient Descent"
    else: 
        pred = predicting_stoch(bX_test)
        best = "Stochastic Gradient Descent"
    for x in range(len(pred)):      # we count our true/false positives and t/f negatives here
        if pred[x] - bY_test[x] == 0:
            if pred[x] + bY_test[x] == 8:
                tp += 1
            else:
                tn += 1
        elif pred[x] - bY_test[x] == 2:
            fp += 1
        else:
            fn += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return best

print(f"Accuracy, Recall, and Precision of 'Best' Model {rpa(bcv, scv)}")
print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}")


# reuse sigmoid
# σ(a) = ln( p(x|C_1)p(C_1) / p(x|C_2)p(C_2) )
# P(C_1|x) = σ(w^T x + ω_0)
# w = Σ^-1 (μ_1 - μ_2)
# ω_0 = -1/2*μ_1^T*Σ^-1*μ_1+1/2*μ_2^Τ*Σ^-1*μ_2+ln(p(C_1)/p(C_2))
# γ = 1/N Σ_{n=1}^N y_n = N_1 / N = N_1 / (N_1 + N_2)
# μ_1 = 1/Ν_1*Σ_{n=1}^N y_n*x_n
# μ_2 = 1/N_2*Σ_{n=1}^N (1-y_n)x_n
# Σ_ML = N_1/N * S_1 + N_2/N * S_2
# S_i = 1/N_i * Σ_{n ε C_i}(x_n - μ_i)(x_n - μ_i)^T
# x_n is our training data, y_n is our expected output (0 or 1, 2 or 4)

N = len(bX_train)
bTpos = []
bTneg = []
bTfull = []
for x,y in zip(bX_train, bY_train):   # converting positive case to 1 and negative to 0 instead of 4 and 2 respectively to work with the ML solution properly
    if y == 2:
        bTneg.append(np.asarray(x, dtype=np.float16))
        bTfull.append([np.asarray(x, dtype=np.float16), 0])
    else:
        bTpos.append(np.asarray(x, dtype=np.float16))
        bTfull.append([np.asarray(x, dtype=np.float16), 1])
Np = len(bTpos)
Nn = len(bTneg)
gamma = Nn / N
m1sum = np.zeros((1,9))
m2sum = np.zeros((1,9))
for m in bTfull:
    m1sum = m1sum + (m[1]*m[0])
m1 = (1/Nn)*m1sum
for n in bTfull:
    m2sum = m2sum + ((1-n[1])*n[0])
m2 = (1/Np)*m2sum
s1 = np.dot((bTneg-m1).T, (bTneg-m1))
s2 = np.dot((bTpos-m2).T, (bTpos-m2))
SML = (1/N)*(s1+s2)
pc1 = gamma
pc2 = 1-gamma
wml = np.dot(np.linalg.inv(SML).T, (m1 - m2).T)
w0 = -0.5*(np.dot(m1, np.dot(np.linalg.inv(SML), m1.T)))+0.5*(np.dot(m2, np.dot(np.linalg.inv(SML), m2.T)))+np.log(pc1/pc2)
asig = np.dot(bX_test, wml)+w0
asig = np.asmatrix(asig, dtype=np.float64)
pc1x = sigmoid(asig)

def probgen_pred(pcon):  
    results=[]
    for i in pcon:
        if i > 0.5:
            results.append(2)
        else:
            results.append(4)
    return results

def prob_rpa(pcon):
    ptp = 0
    ptn = 0 
    pfp = 0 
    pfn = 0 
    p_recall = 0 
    p_precision = 0
    p_accuracy = 0
    pred = probgen_pred(pcon)
    for x in range(len(pred)):      # we count our true/false positives and t/f negatives here
        if pred[x] - bY_test[x] == 0:
            if pred[x] + bY_test[x] == 8:
                ptp += 1
            else:
                ptn += 1
        elif pred[x] - bY_test[x] == 2:
            pfp += 1
        else:
            pfn += 1
    p_recall = ptp / (ptp + pfn)
    p_precision = ptp / (ptp + pfp)
    p_accuracy = (ptp + ptn) / (ptp + ptn + pfp + pfn)
    print("Accuracy, Recall, and Precision of Probabilistic Generative Model")
    print(f"Accuracy: {p_accuracy}\nRecall: {p_recall}\nPrecision: {p_precision}")
    
prob_rpa(pc1x)
