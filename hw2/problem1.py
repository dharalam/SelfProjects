import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split

# split processed.cleveland.data into 2 different sections: 0 (no heart disease) and 1-4 (heart disease)
# m_1 = 1/(N_1) * Σ_(n ε C_1) x_n,      m_2 = 1/(N_2) * Σ_(n ε C_2) x_n
# m_2 - m_1 = w^T (m_2 - m_1), where m_k = w^T * m_k
# (s_k)^2 = Σ_(n ε C_k) (y_n - m_k)^2, where y_n = w^T * x_n
# J(w) = (m_2 - m_1)^2 / ((s_1)^2 + (s_2)^2) which can be rewritten as J(w) = (w^T * S_B * w) / (w^T * S_W * w), where S_B is the between-class cov matrix and is given by S_B = (m_2 - m_1)(m_2 - m_1)^T
# and S_W is the total within-class cov matrix given by S_W = Σ_(n ε C_1) (x_n - m_1)(x_n - m_1)^T + Σ_(n ε C_2) (x_n - m_2)(x_n - m_2)^T
# J(w) δ/δw = (w^T * S_B * w) * S_W * w = (w^T * S_W * w) * S_B * w
# w α (S_W)^-1 (m_2 - m_1) <- this is Fisher's Linear Descriminant  

pcd = pd.read_csv(r'processed.cleveland.csv')  
pcdata = pcd.values
for i in range(len(pcdata)):
    for j in range(len(pcdata[i])-1):
        if (pcdata[i])[j] == '?':
            (pcdata[i])[j] = 1.0
        (pcdata[i])[j] = float((pcdata[i])[j])
pcd_train, pcd_test = train_test_split(pcdata, test_size = 0.20, random_state=1, shuffle=True)
pcd_train = np.array(pcd_train)
pcd_test = np.array(pcd_test)
pcd_train_neg = []
pcd_train_pos = []
for x in pcd_train:
    if x[-1] > 0:
        pcd_train_pos.append(x[0:13])
    else:
        pcd_train_neg.append(x[0:13])
pcd_train_neg = np.array(pcd_train_neg)
pcd_train_pos = np.array(pcd_train_pos)
w = np.random.randn(len(pcd_train[0])-1, )
m1 = np.mean(pcd_train_pos)
m2 = np.mean(pcd_train_neg)
yp = np.dot(w, pcd_train_pos.T)
yn = np.dot(w, pcd_train_neg.T)
s1sq = np.sum((yp - m1)**2)
s2sq = np.sum((yn - m2)**2)
sW1 = np.zeros((13,13))
for i in pcd_train_pos:
    sW1 = np.add((np.dot((np.asmatrix(i) - m1).T,((np.asmatrix(i) - m1)))), sW1)
sW2 = np.zeros((13,13))
for j in pcd_train_neg:
    sW2 = np.add((np.dot((np.asmatrix(j) - m2).T,((np.asmatrix(j) - m2)))), sW2)
sW = sW1 + sW2
fisher = np.linalg.inv(np.asmatrix(sW, dtype=int)) * (m2 - m1)
print(fisher)
print(w)
k = np.dot(np.sum(np.linalg.inv(fisher.T), axis=0)/13, w)
print(f"k is: {k}")
weight = np.multiply(k, np.sum(fisher, axis=0))
print(weight)
posbound = np.mean(np.dot(weight, pcd_train_pos.T))
print(posbound)
negbound = np.mean(np.dot(weight, pcd_train_neg.T))
print(negbound)
dbound = (posbound + negbound)/2
print(dbound)

def sigmoid(x):
    return 1/(1+np.exp(-x))

poslist = []
neglist = []
predlist = []
reslist = []

for x in pcd_test:
    pred = np.dot(np.asmatrix(x[0:13]), weight.T)
    print(sigmoid(int(np.mean(pred))))
    if (np.abs(np.mean(pred))) < np.abs(posbound):
        print(f"Class Prediction: 0     Class Actual: {x[-1]}")
        neglist.append((0, x[-1]))
        reslist.append([0, x[-1]])
    if np.abs(np.mean(pred)) >= np.abs(negbound):
        print(f"Class Prediction: 1   Class Actual: {x[-1]}")
        poslist.append((1, x[-1]))
        reslist.append([1, x[-1]])
    predlist.append(np.asarray(pred)[0][0])

ncorr = 0
nfalse = 0
pcorr = 0
pfalse = 0
for x in neglist:
    if x[1] != 0:
        nfalse += 1
    else: 
        ncorr += 1
for y in poslist:
    if y[1] == 0:
        pfalse += 1
    else:
        pcorr += 1
recall = pcorr / (pcorr+nfalse)
precision = pcorr / (pcorr+pfalse)
accuracy = (pcorr+ncorr) / (ncorr+nfalse+pcorr+pfalse)
nmisclass = nfalse / (ncorr + nfalse)
pmisclass = pfalse / (pcorr + pfalse)
tmisclass = (pfalse + nfalse) / (ncorr+nfalse+pcorr+pfalse)
f1 = (precision * recall) / (precision + recall)
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

for x in range(len(predlist)):
    if reslist[x][0] == reslist[x][1]:
        mp.plot(x, predlist[x],  'g+')
        # true pos/neg
    else:
        mp.plot(x, predlist[x], 'rx')
        # false pos/neg
mp.plot([dbound for i in predlist])
mp.plot([posbound for i in predlist], 'g--')
mp.plot([negbound for i in predlist], 'r--')
mp.show()






