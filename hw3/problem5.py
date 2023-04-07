import pickle, gzip
import numpy as np
import matplotlib.pyplot as mp
import sklearn.svm as sk
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding="latin 1")
f.close()

train_data = np.array_split(train_set[0], 5)
train_class = np.array_split(train_set[1], 5)

valid_data = valid_set[0]
valid_class = valid_set[1]

test_data = test_set[0]
test_class = test_set[1]

def params(neurons, data):
    shape = np.shape(data)
    j = shape[0]
    k = shape[1]
    w1 = np.random.randn(neurons, k)
    w2 = np.random.randn(1, neurons)
    b1 = np.random.randn(neurons, j)
    b2 = np.random.randn()
    return w1, b1, w2, b2
    
recall = 0
precision = 0
accuracy = 0

noise = 0.000000125 # to make sure there will be no divide by zeros or anything in np.log()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return (1-sigmoid(x))*sigmoid(x)

def feedforward(x, w1, b1, w2, b2):
    Z1 = np.dot(w1, x.T) + b1
    H1 = sigmoid(Z1)
    Z2 = np.dot(w2, H1) + b2
    H2 = sigmoid(Z2)
    return H1, H2

def backprop(H1, H2, x, y, w2):
    dZ2 = H2 - y
    dW2 = np.dot(dZ2, H1.T)
    db2 = np.mean(np.sum(dZ2))
    dZ1 = np.dot(w2.T, dZ2)*H1*(1-H1)
    dW1 = (np.dot(dZ1, x))
    db1 = np.mean(dZ1)
    return dW1, db1, dW2, db2

def update(w1, b1, w2, b2, dw1, db1, dw2, db2, step, g1, g2):
    adagrad1 = step/np.sqrt(np.mean(np.sum(g1)**2)+noise)
    adagrad2 = step/np.sqrt(np.mean(np.sum(g2)**2)+noise)
    w1 = w1 - adagrad1*dw1
    b1 = b1 - adagrad1*db1
    w2 = w2 - adagrad2*dw2
    b2 = b2 - adagrad2*db2
    return w1, b1, w2, b2

def booltoint(b):
    if b == True:
        return 1
    else:
        return 0

def get_acc(predictions, y):
    clf = make_pipeline(StandardScaler(), sk.SVC(gamma="auto"))
    clf.fit(predictions, y)
    pred = clf.predict(predictions)
    t = 0
    f = 0
    for k in range(len(pred)):
        if y[k] - pred[k] == 0:
            t += 1
        else:
            f += 1
    acc = t / (t+f)
    return acc

def geterr(pred, actual):
    return np.sum(-1*actual*np.log(pred+noise)-(1-actual)*np.log(1-pred+noise))

def validate(w1, b1, w2, b2, x, y, iterations, step):
    ag1 = []
    ag2 = []
    valE = []
    for i in range(iterations):
        h1, h2 = feedforward(x, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backprop(h1, h2, x, y, w2)
        ag1.append(dw1)
        ag2.append(dw2)
        w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, step, ag1, ag2)
        valE.append(geterr(h2, y))
    return valE

def test(w1, b1, w2, b2, x, y):
    h1, h2 = feedforward(x, w1, b1, w2, b2)
    return geterr(h2, y)
    
def gradescent(x, y, iterations, step, neurons):
    aW1 = [] # for the aggregate sets
    aW2 = []
    aB1 = []
    aB2 = []
    atrE = []
    atE = []
    avalE = []    
    for i in range(iterations):
        bW1 = []
        bW2 = []
        bB1 = []
        bB2 = []
        btrE = []
        btE = []
        bvalE = [] 
        for j in range(len(x)):
            w1, b1, w2, b2 = params(neurons, x[j])
            val = validate(w1, b1, w2, b2, valid_data, valid_class, iterations, step)
            ag1 = []
            ag2 = []
            trE = []
            tE = []
            h1, h2 = feedforward(x[j], w1, b1, w2, b2)
            dw1, db1, dw2, db2 = backprop(h1, h2, x[j], y[j], w2)
            ag1.append(dw1)
            ag2.append(dw2)
            E = test(w1, b1, w2, b2, test_data, test_class)
            tE.append(E)
            trE.append(geterr(h2, y[j]))
            if i<len(val)-1 and val[i+1] > val[i]:
                bW1.append(w1)
                bW2.append(w2)
                bB1.append(b1)
                bB2.append(b2)
                btrE.append(trE)
                btE.append(tE)
                bvalE.append(val)
                continue
            w1, b1, w2, b2 = update(w1, b1, w2, b2, dw1, db1, dw2, db2, step, ag1, ag2)
            bW1.append(w1)
            bW2.append(w2)
            bB1.append(b1)
            bB2.append(b2)
            btrE.append(trE)
            btE.append(tE)
            bvalE.append(val)
            continue
        if i!=0 and np.mean(avalE[i-1]) < np.mean(np.mean(bvalE, axis=1)): # application of early stopping
            break
        aW1.append(np.mean(bW1, axis=1))
        aW2.append(np.mean(bW2, axis=1))
        aB1.append(np.mean(bB1, axis=1))
        aB2.append(np.mean(bB2))
        atrE.append(np.mean(btrE, axis=1))
        atE.append(np.mean(btE, axis=1))
        avalE.append(np.mean(bvalE, axis=1))
    aW1 = np.mean(aW1, axis=1)
    aB1 = np.mean(aB1, axis=1)
    aW2 = np.mean(aW2, axis=1)
    aB2 = np.mean(aB2)
    atrE = np.mean(atrE, axis=1)
    avalE = np.mean(avalE, axis=1)
    atE = np.mean(atE, axis=1)
    return aW1, aB1, aW2, aB2, atrE, avalE, atE

modelrec = []
model = [1, 5, 10, 20, 50, 100]
for i in model:
    print(f"Neurons in hidden layer: {i}")
    w1, b1, w2, b2, trE, valE, tE = gradescent(train_data, train_class, 10, 0.1, i)
    best = np.mean([trE[i]/valE[i] for i in range(trE.size)])
    modelrec.append(np.abs(best))
    print(best)
    print("\n")
model = model[modelrec.index(np.amin(modelrec))]

iters = 10
w1, b1, w2, b2, trE, valE, tE = gradescent(train_data, train_class, iters, 0.1, model)
th1, th2 = feedforward(test_data, w1, b1, w2.T, b2)
vh1, vh2 = feedforward(valid_data, w1, b1, w2.T, b2)
xaxis = range(len(valE))
print(f"Best Test Error: {min(tE)}\n")
print(f"Best Validation Error: {min(valE)}\n")
mp.plot(xaxis, trE, 'b>-', label = "training")
mp.plot(xaxis, tE, 'r>-', label="test")
mp.plot(xaxis, valE, 'g>-', label="validation")
mp.show()

# The extreme errors for training, test, and validation are likely due to the fact that I had not used a classifier to separate the values into appropriate predictions, and rather left them
# unprocessed. The graph trends however seem to show that the model has about equal efficacy for training and test sets, with validation having consistently lower error but following the 
# same shape closely. Early stopping usually results in a one-point graph, but within a few runs a graph with two or more points will present itself. For a better look at the graph
# shapes, one can comment out the early stopping portion in the gradescent() function and view the full graph for 10 iterations.