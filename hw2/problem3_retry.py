import os
import re
import csv
import math
import nltk
import numpy as np

def log_p(n):
    # Get the probability's n natural algorithm
    return abs(math.log(abs(n))) \
        if n != 0 and n != 1 else 1

def get_fea_class(D,k):
    # Get all documents in D belonging to the k-th class in C
    return np.array([ d for d in D if k == int(d[0]) ])

def get_count_class(D,k):
    # Get the count p(Ck) of the class Ck in documents D
    return len(get_fea_class(D, k))

def get_counts_term(D,w):
    # Get the count of the term w occurrences in each document from D
    count_wt = np.array([ len([ term \
        for term in d[1] if w == term ]) for d in D ])
    # Get the total count of documents from D, containing the term w
    return len(np.array([ f_wt \
        for f_wt in count_wt if f_wt > 0 ]))

def get_prob_class(D,k):
    # Get the probability p(Ck) of the k-th class Ck
    return get_count_class(D,k) / len(D)

def get_probs_term(D,w):
    # Get the probability of the term w occurrence 
    # in each document from the class Ck
    return get_counts_term(D,w) / len(D)

def parse(S):
    W = S.lower().split()

    # Parse the string S, performing 
    # the normalization and word-stamming using NLTK library
    W = np.array([ re.sub(r"""[,.;@#?!&$\']+\ *""", '', w) for w in W])
    W = np.array([ tag[0] for tag in nltk.pos_tag(W) \
        if re.match('NN', tag[1]) != None or re.match('JJ', tag[1]) != None ])

    return np.array([ w for w in W if len(w) > 2 ])
    
def build_model(D):
    # Build the class prediction model, 
    # based on the corpus of documents in D
    D = np.array([ np.array([ d[0], parse(d[1]) ], \
        dtype=object) for d in D ], dtype=object)
    return np.array([ d for d in D if len(d[1]) > 0 ])

def compute(D,C,S):
    W = parse(S);                 # A set of terms W in the sample S
    Pr = np.empty(0);             # A set of posteriors Pr(Ck | W)

    n = len(W); m = len(C)        # n - # of terms W in S
                                  # m - # of classes in C

    # For each k-th class Ck, compute the posterior Pr(Ck | W)
    for k in range(m):
        pr_ck_w = 0                  # pr_ck_w - the likelihood P(Ck | wi) 
                                     # of Ck is the class of the term wi

        d_ck = get_fea_class(D,k)    # d_ck - A set of documents from the class Ck
        p_ck = get_prob_class(D,k)   # p_ck - Probability of the k-th class Ck in documents D

        # For each term W[i], compute the likelihood P(Ck | wi)
        for i in range(n):
            # Obtain the count and probability of the 
            # term W[i] in the documents from class Ck
            prob_wd_n = get_probs_term(d_ck, W[i])
            count_wt_n = get_counts_term(d_ck, W[i])
            
            pr_ck_w += count_wt_n * \
                log_p(prob_wd_n) if count_wt_n > 0 else 0

        pr_ck_w += p_ck

        # Append the posterior Pr(Ck | W) of the class Ck to the array Pr
        Pr = np.append(Pr, pr_ck_w)

    # Obtain an index of the class Cs as the class in C, 
    # having the maximum posterior Pr(Ck | W)
    Cs = np.where(Pr == np.max(Pr))[0][0]
   
    return Pr,Cs   # Return the array of posteriors Pr
                   # and the index of sample S class Cs

def evaluate(T,D,C):
    print('Classification:')
    print('===============\n')

    # For each sample S in the set T, compute the class of S
    # Estimate the real classification's multinomial entropy and its expectation
    for s in T[:,1]:
        pr_s = '\0'; \
            Pr,Cs = compute(D,C,s)
        for ci,p in zip(range(len(C)),Pr):
            pr_s += prob_stats % (C[ci][1],p)

        print(sampl_stats % (s, C[Cs][1] \
            if np.sum(Pr) > 0 else 'None', pr_s))
