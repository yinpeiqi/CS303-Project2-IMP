import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np

core = 8

class node:
    def __init__(self, edges):
        self.len = len(edges)
        self.edges = edges

class edge:
    def __init__(self, dst, w):
        self.dst = dst
        self.w = w


class NetWork:
    def __init__(self, fileName, seedFileName):
        netFile = open(fileName, "r")
        n, m = map(int, netFile.readline().split())
        edges = []
        self.n = n
        for i in range(0, n+1):
            edges.append([])
        for line in netFile.readlines():
            u, v, w = map(float, line.split())
            u, v = int(u), int(v)
            edges[u].append(edge(v,w))
        self.node = []
        for i in range(0, n+1):
            self.node.append(node(edges[i]))
        netFile.close()

        seedFile = open(seedFileName, "r")
        self.seeds = []
        for line in seedFile.readlines():
            seed = int(line)
            self.seeds.append(seed)
        seedFile.close()


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
            for e in network.node[src].edges:
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
        for e in network.node[src].edges:
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
    return epoch*10


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    #file_name, seed = 'NetHEPT.txt', '1.txt'
    #model = 'LT'
    #seed = 'network_seeds.txt'
    time_limit = args.time_limit
    network = NetWork(file_name, seed)

    np.random.seed()
    epoch = getEpoch(len(network.seeds))
    RESULT = 0
    t1 = time.time()
    if model == 'IC':
        for j in range(0, epoch):
            findIC(network)
    elif model == 'LT':
        for j in range(0, epoch):
            findLT(network)
    RESULT /= epoch
    #print(epoch)
    print(time.time()-t1)
    print(RESULT)