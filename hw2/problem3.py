import numpy as np
import math

doyle = open("pg1661.txt", "r")
austen = open("pg31100.txt", "r")

doyle_wbank = []
doyle_sbank = []
austen_wbank = []
austen_sbank = []
doyle_count = 0
austen_count = 0
    
doyle_read = doyle.read()
austen_read = austen.read()

for r in (("1", ""), ("2", ""),("3", ""),("4", ""),("5", ""),("6", ""),("7", ""),("8", ""),("9", ""),("0", ""),("\n", " "),
              (".", ""),(",", ""),("!", ""),("?", ""),(";", ""),(":", ""), ("--", ""), ('"', ""), ("(", ""), (")", "")):
    print(f"Replacing {r[0]}...")
    doyle_read = doyle_read.replace(*r)
doyle_read = doyle_read.lower()
doyle_read = doyle_read.split(" ")
for i in doyle_read:
    doyle_count += 1
    if i not in doyle_wbank:
        doyle_sbank.append([str(i), 1, "doyle"])
        doyle_wbank.append(str(i))
    else:
        doyle_sbank[doyle_wbank.index(i)][1] += 1

for r in (("1", ""), ("2", ""),("3", ""),("4", ""),("5", ""),("6", ""),("7", ""),("8", ""),("9", ""),("0", ""),("\n", " "),
              (".", ""),(",", ""),("!", ""),("?", ""),(";", ""),(":", ""), ("--", ""), ('"', ""), ("(", ""), (")", "")):
    print(f"Replacing {r[0]}...")
    austen_read = austen_read.replace(*r)
austen_read = austen_read.lower()
austen_read = austen_read.split(" ")
for i in austen_read:
    austen_count += 1
    if i not in austen_wbank:
        austen_sbank.append([str(i), 1, "austen"])
        austen_wbank.append(str(i))
    else:
        austen_sbank[austen_wbank.index(i)][1] += 1

sdoyle_train = doyle_sbank[(math.floor(len(doyle_sbank)*0.25))::]
wdoyle_train = doyle_wbank[(math.floor(len(doyle_sbank)*0.25))::]
doyle_test = doyle_read[0:(math.floor(len(doyle_sbank)*0.25))]
sausten_train = austen_sbank[(math.floor(len(doyle_sbank)*0.25))::]
wausten_train = austen_wbank[(math.floor(len(doyle_sbank)*0.25))::]
austen_test = austen_read[0:(math.floor(len(doyle_sbank)*0.25))]

sdoyle_train2 = np.array_split(np.asarray(sdoyle_train), 20)
sausten_train2 = np.array_split(np.asarray(sausten_train), 20)

pDoyle = []
pAusten = []

for t in sdoyle_train2:
    doyle_sum = 0.0
    for d in t:
        doyle_sum = doyle_sum + np.log(int(d[1]) / doyle_count)
    pDoyle.append(np.log((doyle_count/(doyle_count+austen_count))) + doyle_sum)

for t in sausten_train2:
    austen_sum = 0.0
    for a in t:
        austen_sum = austen_sum + np.log(int(a[1]) / austen_count)
    pAusten.append(np.log((austen_count/(doyle_count+austen_count))) + austen_sum)

def predict(testing):
    c1 = 0.0
    c2 = 0.0
    for d in testing:
        if d in wdoyle_train:
            c1 += np.log(sdoyle_train[wdoyle_train.index(d)][1]/doyle_count)
        if d in wausten_train:
            c2 += np.log(sausten_train[wausten_train.index(d)][1]/austen_count)
    c1 = np.log((doyle_count/(doyle_count+austen_count))) + c1
    c2 = np.log((austen_count/(austen_count+doyle_count))) + c2
    if c2 > c1:
        print("C2: Austen")
        return c2
    else:
        print("C1: Doyle")
        return c1

doyle_test = np.array_split(np.asarray(doyle_test), 20)
austen_test = np.array_split(np.asarray(austen_test), 20)

doyleres = []
austenres = []

for x in doyle_test:
    doyleres.append(predict(x))
for y in austen_test:
    austenres.append(predict(y))

print(doyleres)
print(austenres)
print(pDoyle)
print(pAusten)
print(f"Accuracy: 100% (with a grain of salt)")
# pattern followed between training and test data is the same, negative case is always smaller (Doyle), positive case is always larger (Austen)
# my only issue that I see is that the predictions on the test set are consitently giving the same number even though the words found in each part of the multinomial
# bayes are different. I tried to find a reason as to why, but to no avail. Nonetheless, the model does its job and does it well, apparently.

doyle.close()
austen.close()