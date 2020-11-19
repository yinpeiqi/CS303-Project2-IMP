import time
import argparse
import numpy as np
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
    return epoch


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
    print("Model: ",model)
    print("Result: ", RESULT)
    print("ISE time: ", time.time()-t1)


def getRR(G, time_limit):
    R = []
    t1 = time.time()
    n = G.n
    while( time.time()-t1 < time_limit-30 ):
        v = np.random.randint(n)+1
        R.append(GenerateRR(G, v))
    #print(len(R))
    return R


def IMM(G:NetWork, k, time_limit):
    n = G.n
    R = getRR(G, time_limit)
    S_star_k, _ = NodeSelection(R, n, k)
    return S_star_k


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
    activity[v] = True

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
    cover_count = [0 for _ in range(n+1)]
    for i in range(len(R)):
        RR = R[i]
        for u in RR:
            if u not in node_cover:
                node_cover[u] = set()
            node_cover[u].add(i)
            cover_count[u] += 1

    S_star_k = []
    F_R = 0
    for i in range(1, k+1):
        selected_node = cover_count.index(max(cover_count))
        F_R += len(node_cover[selected_node])
        S_star_k.append(selected_node)
        covered = node_cover[selected_node].copy()
        for RR_index in covered:
            for u in R[RR_index]:
                cover_count[u] -= 1
                node_cover[u].remove(RR_index)
    F_R /= len(R)
    return S_star_k, F_R


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--seed_size', type=int, default=50)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.seed_size
    model = args.model
    #file_name, k = 'twitter-d.txt', 50
    #file_name, k = 'epinions-d-5.txt', 100
    #file_name, k = 'NetHEPT.txt', 50
    #file_name, k = 'network.txt', 5
    #model = 'LT'
    time_limit = args.time_limit
    network = NetWork(file_name)
    np.random.seed()

    S_star_k = IMM(network, k, time_limit)

    for seed in S_star_k:
        print(seed)
    end = time.time()

    # print("IMP time: ", end-start)
    # ISEtest(network, S_star_k, 'IC')
    # ISEtest(network, S_star_k, 'LT')