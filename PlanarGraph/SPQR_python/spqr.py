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

class SPQR_graph():
    def __init__(self, ids_to_spqr_graph):
        self.edges = []
        self.vertices = set()
        self.g_id = len(ids_to_spqr_graph)
        ids_to_spqr_graph.append(self)

    def add_virtual_edge(self, f, t, g_id):
        self.edges.append((f, t, g_id))
        self.vertices.add(f)
        self.vertices.add(t)

    def add_edge(self, f, t):
        self.edges.append((f,t, -1))
        self.vertices.add(f)
        self.vertices.add(t)

    def nx_draw(self, G, color):
        for v in self.vertices:
            G.add_node(str(v))

        for a,b,g_id in self.edges:
            if g_id == -1:
                G.add_edge(str(a),str(b),color=color,weight=2)

        # G.add_edges_from([(str(a), str(b)) for a,b,g_id in self.edges])

class SPQR:
    def __init__(self):
        self.ids_to_spqr_graph = []

    def make_graph(self):
        return SPQR_graph(self.ids_to_spqr_graph)

    def draw(self, colors = ["red", "green", "orange", "yellow", "blue"]):
        G = nx.Graph()

        for i, spqr_graph in enumerate(self.ids_to_spqr_graph):
            spqr_graph.nx_draw(G, colors[i % len(colors)])

        nx.draw_planar(G, edge_color=[edge[2]['color'] for edge in G.edges(data=True)], with_labels=True, font_color='white')
        plt.show()

def SPQR_from_graph(G):
    G.double_bucket_sort()

    

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

# automatically
G = Graph(list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])

SPQR_from_graph(G)



# class S:
#     def __init__(self, n_id, edges):
#         self.n_id = n_id
#         self.edges = edges
#         self.m = []

#     def add(self, t_id):
#         self.m.append(t_id)

#     def __str__(self):
#         return f"S({self.edges}, {self.m})"
#     def __repr__(self):
#         return str(self)

# class P:
#     def __init__(self, n_id, v, u):
#         self.n_id = n_id

#         self.v = v
#         self.u = u

#         self.m = []

#     def add(self, t_id):
#         self.m.append(t_id)

#     def __str__(self):
#         return f"P(({self.v}, {self.u}), {self.m})"
#     def __repr__(self):
#         return str(self)

# class Q:
#     def __init__(self, n_id, v, u):
#         self.n_id = n_id

#         self.v = v
#         self.u = u

#     def __str__(self):
#         return f"Q({self.v}, {self.u})"
#     def __repr__(self):
#         return str(self)

# class R:
#     def __init__(self, n_id, edges):
#         self.n_id = n_id
#         self.edges = edges
#         self.m = []

#     def add(self, t_id):
#         self.m.append(t_id)

#     def __str__(self):
#         return f"R({self.edges}, {self.m})"
#     def __repr__(self):
#         return str(self)

# class SPQR:
#     def __init__(self, G):
#         self.children = []
#         self.edges = []

#         collect_ps

#     def collect_ps(self):
#         ids = 0

#         i = 0
#         j = 1
#         while i < len(G.E):
#             if j < len(G.E) and G.E[i] == G.E[j]:
#                 p_node = P(ids, G.E[i][0], G.E[i][1])
#                 ids += 1
#                 p_node.add(ids); ids += 1 # i

#                 while G.E[i] == G.E[j]:
#                     p_node.add(ids); ids += 1 # j's
#                     j += 1

#                 self.children.append(p_node)
#             else:
#                 q_node = Q(ids, G.E[i][0], G.E[i][1])
#                 ids += 1
#                 self.children.append(q_node)

#             i = j
#             j += 1

#     # ON-LINE PLANARITY TESTING
#     # planar st-graph (st-planar graph can be given a dominance drawing)
#     def docompose_graph(G):
#         ids = 0
#         if len(G.E) == 1:
#             print ("Construct Q")
#             return Q(ids, G.E[0][0], G.E[0][1])
#         elif False: # G is not biconnected
#             print ("Construct S")
#             pass # TODO # S()
#         elif False: # split pair
#             print ("Construct P")
#             pass # P()
#         else:
#             print ("Construct R")
#             return R()

# # 0   1
# #  2 3
# #  4 5
# # 6   7  13
# #      14
# #      15
# # 8   9  12
# #  10
# #  11
# # n = 16

# # automatically
# G = Graph(list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])
# # print (G.V)
# # print (G.E)

# spqr = SPQR(G)

# # G = Graph(list(range(4)), [(1,3),(3,2), (0, 2), (2, 0), (2, 1)])
# # print (G.V)
# # print (G.E)

# # spqr = SPQR(G)
# # print (spqr.children)
