import random
import os
import colorsys
import math
from PIL import Image
import heapq
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, V, E):
        self.V = V
        self.E = [self.make_edge(v, u) for v, u in E]

        self.double_bucket_sort()

    def make_edge(self, v, u):
        return (min(v,u), max(v,u))

    def double_bucket_sort(self):
        def bucket_sort(f):
            buckets = {v: [] for v in self.V}
            for v, u in self.E:
                buckets[f(v,u)].append((v,u))
            edges = []
            for v in self.V:
                edges += buckets[v]
            self.E = edges
        bucket_sort(lambda v, u: v)
        bucket_sort(lambda v, u: u)

class P:
    def __init__(self, n_id, v, u):
        self.n_id = n_id

        self.v = v
        self.u = u

        self.m = []

    def add(self, t_id):
        self.m.append(t_id)

    def __str__(self):
        return f"P(({self.v}, {self.u}), {self.m})"
    def __repr__(self):
        return str(self)

class Q:
    def __init__(self, n_id, v, u):
        self.n_id = n_id

        self.v = v
        self.u = u

    def __str__(self):
        return f"Q({self.v}, {self.u})"
    def __repr__(self):
        return str(self)

class R:
    def __init__(self, n_id, edges):
        self.n_id = n_id
        self.edges = edges
        self.m = []

    def add(self, t_id):
        self.m.append(t_id)

    def __str__(self):
        return f"R({self.edges}, {self.m})"
    def __repr__(self):
        return str(self)

class S:
    def __init__(self, n_id, edges):
        self.n_id = n_id
        self.edges = edges
        self.m = []

    def add(self, t_id):
        self.m.append(t_id)

    def __str__(self):
        return f"S({self.edges}, {self.m})"
    def __repr__(self):
        return str(self)

class SPQR:
    def __init__(self, G):
        self.children = []
        self.edges = []


    def collect_ps(self):
        ids = 0

        i = 0
        j = 1
        while i < len(G.E):
            if j < len(G.E) and G.E[i] == G.E[j]:
                p_node = P(ids, G.E[i][0], G.E[i][1])
                ids += 1
                p_node.add(ids); ids += 1 # i

                while G.E[i] == G.E[j]:
                    p_node.add(ids); ids += 1 # j's
                    j += 1

                self.children.append(p_node)
            else:
                q_node = Q(ids, G.E[i][0], G.E[i][1])
                ids += 1
                self.children.append(q_node)

            i = j
            j += 1

    # ON-LINE PLANARITY TESTING
    # planar st-graph (st-planar graph can be given a dominance drawing)
    def docompose_graph(G):
        ids = 0
        if len(G.E) == 1:
            return Q(ids, G.E[0][0], G.E[0][1])
        elif False: # G is not biconnected
            pass # TODO # S()
        elif False: # split pair
            pass # P()
        else:
            return R()
            


# 0   1
#  2 3
#  4 5
# 6   7  13
#      14
#      15
# 8   9  12
#  10
#  11
# n = 16

# manually
p_id = 0
s_id = 1
r_1_id = 2
r_2_id = 3
r_3_id = 4

p_node = P(p_id, 8,9)
p_node.add((s_id, (8, 9)))
p_node.add((r_3_id, (8, 9)))

s_node = S(p_id, [(6,8)])
s_node.add((r_1_id, (6,7)))
s_node.add((r_2_id, (7,9)))
s_node.add((p_id, (8,9)))

r1_node = R(r_1_id, [(0,1),(0,2),(0,6),(1,3),(1,7),(2,3),(2,4),(3,5),(4,5),(4,6),(5,7)])
r1_node.add((s_id, (6, 7)))

r2_node = R(r_2_id, [(7,13),(7,14),(9,12),(9,15),(12,13),(12,15),(13,14),(14,15)])
r2_node.add((s_id, (7,9)))

r3_node = R(r_3_id, [(8,10),(8,11),(9,10),(9,11)])
r3_node.add((p_id, (8,9)))

children = [p_node, s_node, r1_node, r2_node, r3_node]
edges = [((8, 9), (p_id, r_3_id)),
         ((8, 9), (p_id, s_id)),
         ((6, 7), (s_id, r_1_id)),
         ((7, 9), (s_id, r_2_id)),
         ]

print (children)
print (edges)
print ()

# automatically
G = Graph(list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])
# print (G.V)
# print (G.E)

spqr = SPQR(G)
print (spqr.children)



# G = Graph(list(range(4)), [(1,3),(3,2), (0, 2), (2, 0), (2, 1)])
# print (G.V)
# print (G.E)

# spqr = SPQR(G)
# print (spqr.children)
