import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
from math import log, pow, sqrt, log2
import math
import random

model = ''

class node:
    def __init__(self, outEdges, inEdges):
        self.outLen = len(outEdges)
        self.outEdges = outEdges
        self.inLen = len(inEdges)
        self.inEdges = inEdges


class edge:
    def __init__(self, dst, w):
        self.dst = dst
        self.w = w


class NetWork:
    def __init__(self, fileName):
        netFile = open(fileName, "r")
        n, m = map(int, netFile.readline().split())
        outEdges = []
        inEdges = []
        self.n = n
        for i in range(0, n+1):
            outEdges.append([])
            inEdges.append([])
        for line in netFile.readlines():
            u, v, w = map(float, line.split())
            u, v = int(u), int(v)
            outEdges[u].append(edge(v,w))
            inEdges[v].append(edge(u,w))
        self.node = []
        for i in range(0, n+1):
            self.node.append(node(outEdges[i], inEdges[i]))
        netFile.close()

    def setSeed(self, seedSet):
        self.seeds = []
        for seed in seedSet:
            self.seeds.append(seed)


RESULT = 0
def findIC(network):
    activitySet = network.seeds.copy()
    result = len(activitySet)
    activity = {}
    for src in activitySet:
        activity[src] = True

    while len(activitySet) != 0:
        newActibitySet = []
        for src in activitySet:
            for e in network.node[src].outEdges:
                if e.dst not in activity:
                    if e.w > np.random.random():
                        newActibitySet.append(e.dst)
                        activity[e.dst] = True
        result += len(newActibitySet)
        activitySet = newActibitySet
    global RESULT
    RESULT += result


def findLT(network):

    activitySet = network.seeds.copy()
    threshold = {}
    activity = {}
    for src in activitySet:
        activity[src] = 1

    for src in activitySet:
        for e in network.node[src].outEdges:
            if e.dst not in activity:
                threshold[e.dst] = np.random.random()
                activity[e.dst] = e.w
            elif activity[e.dst] != 1:
                activity[e.dst] += e.w
            else:
                continue
            if activity[e.dst] >= threshold[e.dst]:
                activity[e.dst] = 1
                activitySet.append(e.dst)
    result = len(activitySet)
    global RESULT
    RESULT += result


def getEpoch(p):
    epoch = int(4000/p)
    if epoch > 1000:
        epoch = 1000
    if epoch < 50:
        epoch = 50
    return epoch*20


def ISEtest(network, seedSet, model):
    network.setSeed(seedSet)
    epoch = getEpoch(len(network.seeds))
    global RESULT
    RESULT = 0
    t1 = time.time()

    if model == 'IC':
        for j in range(0, epoch):
            findIC(network)
    elif model == 'LT':
        for j in range(0, epoch):
            findLT(network)
    RESULT /= epoch
    # print(epoch)
    print("Model: ",model)
    print("Result: ", RESULT)
    print("ISE time: ", time.time()-t1)


def IMM(G:NetWork, k, epsilon, l):
    n = G.n
    l = l * ( 1 + log(2.0)/log(n) )
    R = Sampling(G, k, epsilon, l)
    S_star_k, _ = NodeSelection(R, n, k)
    return S_star_k


def Sampling(G:NetWork, k, epsilon, l):
    R = []
    LB = 1
    epsilon_2 = sqrt(2.0) * epsilon
    n = G.n

    lambada_2 = Equation9(epsilon_2, n, k, l)
    lambada_star = Equation6(l, n, k, epsilon)

    pow_2_i = 1
    selected_nodes = set()
    for i in range(1, int(log2(n))):

        pow_2_i *= 2
        x = n / pow_2_i
        theta_i = lambada_2 / x
        while len(R) <= theta_i:
            v = np.random.randint(n)
            while v in selected_nodes:
                v = np.random.randint(n)
            R.append(GenerateRR(G, v))

        S_i, F_R = NodeSelection(R, n, k)
        if n*F_R >= ( 1 + epsilon_2 )*x :
            LB = n * F_R / ( 1 + epsilon_2 )
            break

    theta = lambada_star / LB
    while len(R) <= theta:
        v = np.random.randint(n)
        while v in selected_nodes:
            v = np.random.randint(n)
        R.append(GenerateRR(G, v))
    return R


def Equation9(epsilon_2, n, k, l):
    C_n_k = 1
    for i in range(1,k+1):
        C_n_k *= n+1-i
        C_n_k /= i

    upper = ( 2 + 2/3*epsilon_2 ) * ( log(C_n_k) + l*log(n) + log(log2(n)) ) * n
    lower = epsilon_2 * epsilon_2

    return upper / lower


def Equation6(l, n, k, epsilon):
    C_n_k = 1
    for i in range(1, k+1):
        C_n_k *= n + 1 - i
        C_n_k /= i

    alpha = sqrt( l*log(n) + log(2) )
    beta = sqrt( ( 1 - 1/math.e ) * ( log(C_n_k) + l*log(n) + log(2) ) )

    intermediate = ( ( 1 - 1/math.e ) * alpha + beta ) / epsilon
    return 2 * n * intermediate * intermediate


def GenerateRR(G:NetWork, v):
    global model
    if model == 'IC':
        return GenerateRRIC(G, v)
    else:
        return GenerateRRLT(G, v)


def GenerateRRIC(G:NetWork, v):
    activitySet = [v]
    RR = [v]
    activity = {}
    for src in activitySet:
        activity[src] = True

    while len(activitySet) != 0:
        newActibitySet = []
        for src in activitySet:
            for e in G.node[src].inEdges:
                if e.dst not in activity:
                    if e.w > np.random.random():
                        newActibitySet.append(e.dst)
                        activity[e.dst] = True
        activitySet = newActibitySet
        RR.extend(newActibitySet)
    return RR


def GenerateRRLT(G:NetWork, v):
    activitySet = [v]
    activity = {}
    for src in activitySet:
        activity[src] = True

    for v in activitySet:
        if len(G.node[v].inEdges) == 0:
            continue
        u = random.sample(G.node[v].inEdges, 1)[0].dst
        if u not in activity:
            activity[u] = True
            activitySet.append(u)
    return activitySet


def NodeSelection(R, n, k):
    node_cover = {}
    cover_count = [ 0 for _ in range(n+1) ]
    for i in range(len(R)):
        RR = R[i]
        for u in RR:
            cover_count[u] += 1
            if u not in node_cover:
                node_cover[u] = set()
            node_cover[u].add(i)

    S_star_k = []
    F_R = 0
    for i in range(1, k+1):
        selected_node = cover_count.index(max(cover_count))
        F_R += len(node_cover[selected_node])
        S_star_k.append(selected_node)
        covered = node_cover[selected_node].copy()
        for RR_index in covered:
            RR = R[RR_index]
            for u in RR:
                cover_count[u] -= 1
                node_cover[u].remove(RR_index)
    F_R /= len(R)
    return S_star_k, F_R


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--seed_size', type=int, default=50)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.seed_size
    model = args.model
    file_name, k = 'NetHEPT.txt', 50
    model = 'LT'
    time_limit = args.time_limit
    network = NetWork(file_name)
    np.random.seed()

    epsilon = 0.1
    l = 1

    start = time.time()
    S_star_k = IMM(network, k, epsilon, l)

    for seed in S_star_k:
        print(seed)
    end = time.time()
    print("IMP time: ", end-start)
    ISEtest(network, S_star_k, 'IC')
    ISEtest(network, S_star_k, 'LT')