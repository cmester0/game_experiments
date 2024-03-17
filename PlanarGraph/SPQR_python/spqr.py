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
        self.A = {v: set() for v in V}
        self.E = [self.make_edge(v, u) for v, u in E]

    def make_edge(self, v, u):
        a,b = (min(v,u), max(v,u))
        self.A[a].add(b)
        self.A[b].add(a)
        return a,b

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
        self.adjecency = {}
        self.g_id = len(ids_to_spqr_graph)
        ids_to_spqr_graph.append(self)

    def add_virtual_edge(self, f, t, g_id):
        self.edges.append((f, t, g_id))
        self.vertices.add(f)
        self.vertices.add(t)

        if not f in self.adjecency: self.adjecency[f] = set()
        if not t in self.adjecency: self.adjecency[t] = set()

        self.adjecency[t].add(f)
        self.adjecency[f].add(t)

    def add_edge(self, f, t):
        self.edges.append((f,t, -1))
        self.vertices.add(f)
        self.vertices.add(t)

        if not f in self.adjecency: self.adjecency[f] = set()
        if not t in self.adjecency: self.adjecency[t] = set()

        self.adjecency[t].add(f)
        self.adjecency[f].add(t)

    def from_graph(self, G):
        for f,t in G.E:
            self.add_edge(f,t)

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
        # nx.draw(G, edge_color=[edge[2]['color'] for edge in G.edges(data=True)], with_labels=True, font_color='white')
        plt.show()
        # plt.savefig("spqr_fig.png")

def SPQR_from_graph(G):
    G.double_bucket_sort()

    # TODO: Collect P into componenets!

    # 1. Perform depth first search and number based on ordering
    # and Calculate lowpt1, lowpt2, ND and father for each vertex

    number = {}

    tree_arch = set()
    frond = set()

    lowpt1 = {}
    lowpt2 = {}
    nd = {}
    parent = {}

    # DFS
    flag = {i: True for i in G.V}
    def dfs(v,u):
        number[v] = len(number)

        # a
        lowpt1[v] = number[v]
        lowpt2[v] = number[v]
        nd[v] = 1
        # /a

        for w in G.A[v]:
            if not w in number:
                tree_arch.add((v,w))
                # tree_arch.add((w,v)) # Add both directions?

                dfs(w,v)
                # b
                if lowpt1[w] < lowpt1[v]:
                    lowpt2[v] = min(lowpt1[v], lowpt2[w])
                    lowpt1[v] = lowpt1[w]
                elif lowpt1[w] == lowpt1[v]:
                    lowpt2[v] = min(lowpt2[v], lowpt2[w])
                else:
                    lowpt2[v] = min(lowpt2[v], lowpt1[w])
                nd[v] = nd[v] + nd[w]
                parent[w] = v
                # /b
            elif number[w] < number[v] and (w != u or not flag[v]):
                frond.add((v,w))
                # frond.add((w,v)) # Add both directions?
                # c
                if number[w] < lowpt1[v]:
                    lowpt2[v] = lowpt1[v]
                    lowpt1[v] = number[w]
                elif number[w] > lowpt1[v]:
                    lowpt2[v] = min(lowpt2[v], number[w])
                # /c

            if w == u: flag[v] = False

    dfs(next(iter(G.V)), 0)

    # 2. build adjecency structure based on lowpt1 and lowpt2

    # e[0] = v
    # e[1] = w
    phi = lambda e: \
        3 * lowpt1[e[1]] if e in tree_arch and lowpt2[e[1]] < e[0] else \
        (3 * e[1] + 1 if e in frond else \
         3 * lowpt1[e[1]] + 2)

    # TODO: Use bucket sort here!
    adjacency = [sorted(list(G.A[v]), key=lambda w: phi((v,w))) for v in G.A]

    # 3. do DFS using adjecency structure

    # recalculate lowpt1 and lowpt2?

    # calculate A1, degree, highpt

    def high(w):
        Fw = list(filter(lambda x: x[1] == w, frond))
        return 0 if len(Fw) == 0 else Fw[0] # TODO: ?

    paths = []
    s = 0
    m = len(G.V)-1
    newnum = {i: 0 for i in G.V}
    highpt = {i: 0 for i in G.V}
    path = []

    def pathfinder(v):
        nonlocal s, m, path

        newnum[v] = m - nd[v] + 1
        for w in adjacency[v]:
            if s == 0:
                s = v
                path = []
            path.append((v,w))
            if (v,w) in tree_arch:
                pathfinder(w)
                m = m - 1
            else:
                if highpt[newnum[w]] == 0:
                    highpt[newnum[w]] = newnum[v]
                paths.append(path)
                s = 0
    
    elem = list(number)[0]
    pathfinder(elem)

    tree = SPQR()
    tree.make_graph().from_graph(G)
    for p in paths:
        SPQR_G = tree.make_graph()
        for e in p:
            SPQR_G.add_edge(e[0], e[1])
    tree.draw()
    
    print ("\n".join(list(map(str, paths))))

    elem = list(number)[0]

    EOS = None
    TSTACK = [EOS]
    ESTACK = []
    
    # def PathSearch(v):
    #     for w in adjacency[v]:
    #         e = (v,w)
    #         e_start_path = e in map(lambda p: p[0], paths)

    #         if e in tree_arch:
    #             # e starts a path
    #             if e_start_path:
    #                 # pop all (h,a,b) with a > lowpt1[w] from TSTACK
    #                 deleted = []
    #                 while not TSTACK[-1] is None and TSTACK[-1][1] > lowpt1[w]: deleted.append(TSTACK.pop())
    #                 TSTACK.push()
    #                 if len(deleted) == 0:
    #                     TSTACK.append((w + nd[w] - 1, lowpt1[w], v))
    #                 else:
    #                     y = max(map(lambda x: x[0], deleted))
    #                     h,a,b = deleted[-1]
    #                     TSTACK.append((max(y, w + nd[w] - 1), lowpt1[w], b))
    #                 TSTACK.append(EOS)
    #             PathSearch(w)
    #             ESTACK.append((v,w))
    #             # Check for type 2
    #             while v != 1 and (any(map(lambda x: not x is None and x[1] == v, TSTACK)) or deg)
    #             # /Check for type 2
    #             # Check for type 1
    #             if e_start_path:
    #                 while not TSTACK[-1] is None: TSTACK.pop()
    #                 TSTACK.pop()
    #             while not TSTACK[-1] is None and TSTACK[-1][1] != v and TSTACK[-1][2] != v and high(v) > TSTACK[-1][0]: TSTACK.pop()
    #         else:
    #             if e_start_path:
    #                 deleted = []
    #                 while not TSTACK[-1] is None and TSTACK[-1][1] > w: deleted.append(TSTACK.pop())
    #                 if len(deleted) == 0:
    #                     TSTACK.append((v,w,v))
    #                 else:
    #                     y = max(map(lambda x: x[0], deleted))
    #                     h,a,b = deleted[-1]
    #                     TSTACK.append((y, w, b))
    #             if v in parent and w == parent[v]:
    #                 # TODO: Make new components
    #                 pass
    #             else:
    #                 ESTACK.append(e)

    # PathSearch(elem)

    # print ("ESTACK", ESTACK)

























    
    # # Find all cycles of the graph:
    # elem = next(iter(G.V))
    # stk = [(elem, [elem])]
    # order = {}

    # adjacency = [G.A[v] for v in G.A]
    # tree_arch = set()
    # frond = set()

    # while len(stk):
    #     i, p = stk.pop()

    #     if i in order:
    #         if len(p) > 1:
    #             frond.add((p[-2],i))
    #         continue
    #     order[i] = len(order)
    #     if len(p) > 1:
    #         tree_arch.add((p[-2],i))

    #     for j in G.A[i]:
    #         stk.append((j, p+[j]))

    # print (tree_arch)
    # print (frond)

    # for i in order:
    #     stk = [(i,[i])]
    #     possible = []
    #     while len(stk):
    #         j,p = stk.pop()
    #         possible.append(p)
    #         for k in G.A[j]:
    #             if order[k] < order[j]:
    #                 continue

    #             stk.append((k, p+[k]))


    
    # tree = SPQR()
    # SPQR_G = tree.make_graph()
    # # SPQR_G.directly_from_graph(G)



    # lowest vertex reachable by traversing zero or more tree arcs
    # lowpt1(v) = min({v} U {w | v -*> \-> w})
    lowpt1 = {v: 0 for v in G.V}

    # lowpt2(v) = min({v} U ({w | v -*> \-> w} \ {lowpt1(v)}))
    
    
    # for v in G.V:
        
    
    # for v,w in G.E:
    #         if v == w:
    #             continue

    #         stk = [next(i for i in set(G.V) - set([v,w]))]
    #         visited = set()
    #         while len(stk):
    #             i = stk.pop()

    #             if i in [v,w]:
    #                 continue

    #             if i in visited:
    #                 continue
    #             visited.add(i)

    #             for j in G.A[i]:
    #                 stk.append(j)

    #         if len(visited) < len(G.V):
    #             print (v,w)


## WIKI EXAMPLE

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
