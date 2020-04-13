import tensorflow as tf
import numpy as np
import time
import math as mt
import matplotlib.pyplot as plt
import random
import pickle
np.random.seed(2)
random.seed(2)

# Hyper-parameters
seed = 1  # random seed

capacity = 10  # Bin capacity
sMin = 1
sMax = capacity-1  # Piece size limits

# Data graphing variables
globalAverage = []  # training
### testing results
globalAverage_test = []  # testing
# Heuristics

def insertLast(bins, pieces, capacity):
    piece = pieces[0]
    pieces = np.delete(pieces, 0)
    nBin = np.size(bins) - 1
    spaceUsed = bins[nBin] + piece
    if spaceUsed < capacity:
        bins[nBin] = spaceUsed
    else:
        bins.append(piece)
    return pieces, bins

## insert to the first bin that has open space
def insertFirst(bins, pieces, capacity):
    piece = pieces[0]
    pieces = np.delete(pieces, 0)
    n = np.size(bins)
    for i in range(0, n):
        spaceUsed = bins[i] + piece
        if spaceUsed < capacity:
            bins[i] = spaceUsed
            return pieces, bins
    bins.append(piece)
    return pieces, bins

def observation(bins, pieces, capacity):
    o = np.zeros([1, 10])
    space = [v for i, v in enumerate(bins) if v < capacity]
    space2 = capacity - np.vstack(bins)
    o[0, 0] = pieces[0] # Next piece
    o[0, 1] = pieces.size # Pieces left
    o[0, 2] = np.sum(pieces) # Total space from pieces left
    o[0, 3] = round(np.mean(pieces)) # Piece average size
    o[0, 4] = np.amin(space2) # Min space left in a bin
    o[0, 5] = np.amax(space2) # Max space left in a bin
    o[0, 7] = np.sum(bins)/(len(bins)*capacity) # space used in %
    o[0, 8] = np.mean(space2) # Average space left from a bin
    o[0, 9] = len(space) # Number of opened bins
    return o

def score(bins, capacity):
    # max score per closed bin 10
    score = 0
    for i in range(0, np.size(bins)):
        b = bins[i]*1.0/capacity
        if  b == 1:
            score += 2
        elif b > .7:
            score += b
        else:
            score -= (1-b)
    return score


def discount(r, Episode, rewardLimit, gamma):
    dr = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add*gamma + r[t]
        dr[t] = running_add
        if t % Episode == 0:
            running_add = 0  # Reset reward for the next iter
    return dr

## used
def d_generator(c, sMin, sMax, N, seed=1):
    pieces = np.random.randint(sMin, sMax, size=[N, 1])
    return pieces.flatten()

first_gamma = .99  # Discount factor

nEpisodes = 10
nitersPerEpisode = 10
rewardLimit = 50
nPieces=10;
niters=3*10**5
learning_rate = 1e-4
heuristics = {0: insertLast, 1: insertFirst}
Decay = .99

# Model
model = {}
model['W1'] = np.random.randn(100,10)*.1 / np.sqrt(10)
model['W2'] =  np.random.randn(1,100) *.1/ np.sqrt(100)
model_buffer = {}
model_cache = {}
for k,v in model.iteritems():
    model_buffer[k] = np.zeros_like(v)
    model_cache[k] = np.zeros_like(v)



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def policy_forward(x):
    h = np.dot(model['W1'], x.T)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(epdlogp.T,eph)
            dh = np.dot(epdlogp, model['W2'])
                dh[eph <= 0] = 0 # backpro prelu
                    dW1 = np.dot(dh.T, epx)
                        return {'W1':dW1, 'W2':dW2}

# RL part
iters = 0
ep = 0
spaceUsed  = 0
##testing bin space used
spaceUsed_test = 0
while iters < niters: ##iterations first_niter
    
    # Start episode
    rewards = []
    observations = []
    targets = []
    hiddenStates = []
    dlogps = []
    
    ### batch size is the number of sets of objects in each episode
    for i in range(0,nitersPerEpisode):
        
        # Start
        iter = True
        pieces = d_generator(capacity, sMin, sMax, nPieces)
        ### test pieces
        pieces_test = d_generator(capacity, sMin, sMax, nPieces, seed=567)
        #print pieces, pieces_test
        bins = [0]
        
        ##test bins
        bins_test = [0]
        
        ## stop only when there is no object left
        while iter:
            
            x = observation(bins, pieces, capacity)
            
            # Take decision using an stochastic approach
            probs, h = policy_forward(x)
            r = np.random.random(1)
            j = 0
            if r < probs:
                j = 1
            
            # Take action
            pieces, bins = heuristics[j](bins, pieces, capacity)
            ## take action for test piece
            pieces_test, bins_test = heuristics[j](bins_test, pieces_test, capacity)
            
            r = score(bins, capacity)
            
            y = j
            rewards.append(r)
            observations.append(x)
            hiddenStates.append(h.T)
            targets.append(y)
            dlogps.append(y-np.vstack(probs).T)
            if np.size(pieces) == 0:
                iter = False

        iters +=1
        newAverage = np.sum(bins)*1.0/(capacity*np.size(bins))
        newAverage_test = np.sum(bins_test)*1.0/(capacity*np.size(bins_test))
        #print newAverage, newAverage_test
        spaceUsed += newAverage
        spaceUsed_test += newAverage_test
    
    ep += 1
    epr = np.vstack(rewards)
    epx = np.vstack(observations)
    epy = np.vstack(targets)
    eph = np.vstack(hiddenStates)
    epdlogp = np.vstack(dlogps)
    discountedRewards = discount(epr , nitersPerEpisode, rewardLimit, first_gamma)
    discountedRewards -= np.mean(discountedRewards)
    discountedRewards /= np.std(discountedRewards)
    
    # Get gradients and add them up!
    epdlogp *= discountedRewards
    
    gradss = policy_backward(eph, epdlogp)
    
    for k, v in model.iteritems():
        model_buffer[k] += gradss[k]

    if ep == nEpisodes:
        print 'entering backprop'
        # Backpropagate using  stacked gradients and rmsprop
        for k, v in model.iteritems():
            model_cache[k] = Decay*model_cache[k]+(1-Decay)*model_buffer[k]**2
            model[k] += learning_rate * model_buffer[k] / (np.sqrt(model_cache[k]) + 1e-5)
            model_buffer[k] = np.zeros_like(v) # Resetting grad buffer
        
        ep = 0 #Resetting episode run
        globalAverage.append(spaceUsed/iters)
        globalAverage_test.append(spaceUsed_test/iters)
        print 'Resetting environment and backpropagating. average space filled = {0} and test is {1}'.format(globalAverage[-1], globalAverage_test[-1])