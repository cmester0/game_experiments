import math

import random
import os
import colorsys
import math
from PIL import Image
import heapq
from collections import defaultdict

# Planar embedding
# https://en.wikipedia.org/wiki/Planar_straight-line_graph
# https://en.wikipedia.org/wiki/Planarity_testing

width = 2 * 2000 # int(input("Initial width: "))
height = 2 * 2000 # int(input("Initial height: "))

resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append((0,0,0))

point_width = 10
point_height = 10

def put_point(x,y,color):
    for xi in range(point_width):
        for yi in range(point_height):
            resMap[x+xi][y+yi] = color

def random_point():
    x_rand = random.randint(0, width-point_width)
    y_rand = random.randint(0, height-point_height)
    return x_rand, y_rand

def random_points_list(amount):
    l = []
    for i in range(amount):
        x_rand, y_rand = random_point()
        h = i / amount
        s = 1.0
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h,s,v)
        l.append((x_rand, y_rand, (int(255 * r),int(255 * g),int(255 * b))))
    return l

def print_line(x0,y0,x1,y1, color):
    steps = abs(x0 - x1) + abs(y0 - y1)
    for i in range(int(steps)):
        resMap[int(x0 + point_width / 2 + (x1 - x0) / steps * i)][int(y0 + point_height / 2 + (y1 - y0) / steps * i)] = color

def print_dashed_line(x0,y0,x1,y1, color):
    steps = abs(x0 - x1) + abs(y0 - y1)

    new_range = list(sorted(list(range(0,int(steps), 8)) +
                            list(range(1,int(steps), 8)) +
                            list(range(2,int(steps), 8))))

    for i in new_range:
        resMap[int(x0 + point_width / 2 + (x1 - x0) / steps * i)][int(y0 + point_height / 2 + (y1 - y0) / steps * i)] = color

def print_square(x0, y0, x1, y1, color):
    print_line(x0,y0,x0,y1,color)
    print_line(x1,y0,x1,y1,color)
    print_line(x0,y0,x1,y0,color)
    print_line(x0,y1,x1,y1,color)

def print_filled_square(x, y, w, h, color):
    for i in range(w):
        for j in range(h):
            resMap[x+i][y+j] = color

def print_polygon(polygon, color):
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + [polygon[0]]):
        print_line(ax, ay, bx, by, color)

def print_filled_polygon(polygon, color):
    for (i0, i1), i2 in zip(zip(polygon, polygon[1:] + polygon[:1]), polygon[2:] + polygon[:2]):
        steps = abs(i1[0] - i2[0]) + abs(i1[1] - i2[1])
        for j in range(steps):
            print_line(i0[0], i0[1],
                       int(i1[0] + (i2[0] - i1[0]) / steps * j),
                       int(i1[1] + (i2[1] - i1[1]) / steps * j),
                       color)

def print_graph(vertices, edges):
    for (x_rand, y_rand, c_rand) in vertices:
        put_point(x_rand, y_rand, c_rand)

    for i, j in edges:
        x0, y0, c0 = vertices[i]
        x1, y1, c1 = vertices[j]
        steps = abs(x0 - x1) + abs(y0 - y1)
        x2, y2 = x0 + (x1 - x0) / steps * steps/2, y0 + (y1 - y0) / steps * steps/2
        x2_, y2_ = x0 + (x1 - x0) / steps * steps//2, y0 + (y1 - y0) / steps * steps//2
        print_line(x0,y0,x2,y2,c0)
        print_line(x2_,y2_,x1,y1,c1)

def scale_graph(vertices):
    px = list(map(lambda x: x[0], vertices))
    py = list(map(lambda x: x[1], vertices))
    return [(int((x - min(px)) * (width-10) / (max(px) - min(px))),int((y - min(py)) * (height-10) / (max(py) - min(py))),c) for x,y,c in vertices]

def print_graph_scaled(G):
    vertices, edges = G
    print_graph(scale_graph(vertices), edges)

########################
# Input (planar) graph #
########################

# 0---\---\
# | \   \   \
# |  2--3--4
# | /   /   /
# 1---/---/

def calc_neighbors(v, E):
    return [y for x, y in E if x == v] + [y for y, x in E if x == v]

def size(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)

def sub(a,b):
    return (a[0] - b[0], a[1] - b[1])

def scale(a, c):
    return (a[0] * c, a[1] * c)

def add_v(a,b):
    return (a[0] + b[0], a[1] + b[1])

def add_outer_face(G, outer_degree=3):
    V, E = G

    V = list(V)
    E = list(E)

    additional_set = set()
    while len(additional_set) < outer_degree:
        additional_set.add(random.randint(0, len(V)-1))
    additional_set = list(additional_set)

    for _ in range(outer_degree):
        V.append(V[-1]+1)
    outer_face = list(V[-outer_degree:])

    for i in range(outer_degree):
        E.append((V[-1-i], additional_set[i]))

    for i in range(outer_degree-1):
        E.append((V[-1-i], V[-2-i]))
    E.append((V[-1], V[-outer_degree]))

    return (V,E), outer_face

def add_color(G, c = lambda i, n: tuple(map(lambda x: int(x*255), colorsys.hsv_to_rgb(i /n, 0.7, 1.0)))):
    V,E = G
    return [(x,y,c(i, len(V)))for i, (x,y) in enumerate(V)], E

# V,E = tutte_from_graph((list(range(5)),[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4)]))
# print_graph_scaled(V, E)

def spawn_in_area(G):
    V, E = G
    V, E = add_color((V, E))

    SV = scale_graph(V)
    print_graph([(x,y,(255,255,255)) for (x,y,c) in SV], E)

    sMap = []
    for xi in range(width):
        sMap.append([])
        for yi in range(height):
            sMap[xi].append(-1)

    queue = []
    colors = []
    for i, (x, y, c) in enumerate(SV):
        queue.append((x,y, i))
        colors.append(c)

    for (a,b) in E:
        x0, y0, _ = SV[a]
        x1, y1, _ = SV[b]
        steps = abs(x0 - x1) + abs(y0 - y1)
        for i in range(0, int(steps)//2):
            queue.append((int(x0 + point_width / 2 + (x1 - x0) / steps * i),
                          int(y0 + point_height / 2 + (y1 - y0) / steps * i),
                          a))
        for i in range(int(steps)//2, int(steps)):
            queue.append((int(x0 + point_width / 2 + (x1 - x0) / steps * i),
                          int(y0 + point_height / 2 + (y1 - y0) / steps * i),
                          b))

    iters = 0
    while len(queue) > 0:
        iters += 1
        if iters > 0 and iters % 1000 == 0:
            print (iters, len(queue))
        elem_x, elem_y, elem_i = queue.pop(random.randint(0,len(queue)-1))
        if sMap[elem_x][elem_y] != -1:
            continue
        # print (elem_i)
        sMap[elem_x][elem_y] = elem_i
        # print (elem_x, elem_y, elem_i)
        def add_elem(x, y, i):
            if x < 0 or y < 0 or x >= width or y >= height:
                return
            if sMap[x][y] == -1:
                queue.append((x, y, i))

        add_elem(elem_x,   elem_y+1, elem_i)
        add_elem(elem_x,   elem_y-1,  elem_i)
        add_elem(elem_x+1,elem_y,    elem_i)
        add_elem(elem_x-1, elem_y,    elem_i)

    for yi in range(height):
        for xi in range(width):
            if sMap[xi][yi] != -1:
                resMap[xi][yi] = colors[sMap[xi][yi]]

    # print_graph([(x,y,(255,255,255)) for (x,y,c) in SV], E)

# # # 0   1
# # #  2 3
# # #  4 5
# # # 6   7  13
# # #      14
# # #      15
# # # 8   9  12
# # #  10
# # #  11
# # # n = 16
# # G = (list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])

# # G = (list(range(4)),[(0,1),(0,2),(1,3),(2,3)])

## REFERENCE CODE:
# https://github.com/ogdf/ogdf/blob/76c4ded67def1e17ece7f01399910b76b2414c79/src/ogdf/decomposition/StaticSPQRTree.cpp
# And triconnectivity at:
# https://github.com/ogdf/ogdf/blob/76c4ded67def1e17ece7f01399910b76b2414c79/src/ogdf/graphalg/Triconnectivity.cpp#L679

class Comp:
    def __init__(self):
        self.m_edges = []
        self.m_type = -1 # -1 = unassigned, 0 = bond, 1 = polygon, 2 = triconnected

    def append(self, e):
        self.m_edges.append(e)

    def finishTricOrPoly(self, e):
        self.m_edges.append(e)
        self.m_type = 2 if (len(self.m_edges) >= 4) else 1

class Graph:
    def __init__(self, V, E):
        self.nodes = V
        self.edges = E

    def firstNode(self):
        return Node(1)

    def reverseEdge(self, e):
        pass

    def allEdges(self, edges):
        return edges

    def maxNodeIndex(self):
        return 0

    def minNodeIndex(self):
        return 0

# class GraphCopySimple():
#     def __init__(self, G):
#         self.nodes = G.V
#         self.edges = G.E

class adjEntry:
    def __init__(self, e):
        self.e = e

    def theEdge(self):
        return self.e

class Node:
    def __init__(self, val):
        self.val = val
        self.m_in_deg = 0
        self.m_out_deg = 0
        self.adjEntries = []

    def degree(self):
        return self.m_in_deg + self.m_out_deg

    def index(self):
        return 0

class Edge:
    def __init__(self, val):
        self.v = Node(val[0])
        self.w = Node(val[1])
        self.val = val
        pass

    def source(self):
        return self.v

    def target(self):
        return self.w

    # TODO:
    def opposite(self, v):
        if self.v == v:
            return w
        else:
            return self.v

class SPQR_tree:
    def __init__(self, G):
        # array of components
        self.m_component = []
        # number of components
        self.m_numComp = 0

        self.m_TSTACK = []

        self.m_NUMBER = []
        self.m_LOWPT1 = []
        self.m_LOWPT2 = []
        self.m_ND = []
        self.m_DEGREE = []
        self.m_NODE_AT = []
        self.m_FATHER = []
        self.m_TYPE = [] # -1 undefined, 0 = unseen, 1 = tree, 2 = frond, 3 = removed
        self.m_A = [] # Adjecency list

        self.m_NEWNUM = [] # second dfs-number

        self.m_START = [] # edge starts a path
        self.m_TREE_ARC = [] # tree arc entering v

        self.m_HIGHPT = []
        self.m_IN_ADJ = []
        self.m_IN_HIGH = []

        self.m_ESTACK = []

        self.m_start = 0 # start node of dfs traversal
        self.m_numCount = 0 # counter for dfs traversal
        self.m_newPath = False # true iff we start a new path

        # Compute triconnected components:
        self.GC = self.GraphCopySimple(G)

        self.n = len(self.GC.nodes)
        self.m = len(self.GC.edges)

        if self.n <= 2:
            C = newComp()
            for e in self.GC.edges:
                C.append(e) # C << e
            C.m_type = 0 # 0 = bond
            return

        self.m_TYPE = {e.val: 0 for e in self.GC.edges} # 0 = unseen
        self.split_off_multi_edges()

        self.m_NUMBER = [0 for v in self.GC.nodes]

        self.m_LOWPT1 = [v.val for v in self.GC.nodes]
        self.m_LOWPT2 = [v.val for v in self.GC.nodes]

        self.m_FATHER = [None for v in self.GC.nodes]
        self.m_ND = [None for v in self.GC.nodes]
        self.m_DEGREE = [None for v in self.GC.nodes]

        self.m_TREE_ARC = [None for v in self.GC.nodes]
        # self.m_NODE_AT = [1] * n # TODO

        self.m_start = self.GC.firstNode()
        self.DFS1(self.GC, self.m_start, None)

        for e in self.GC.edges:
            up = self.m_NUMBER[e.target().val] - self.m_NUMBER[e.source().val] > 0
            if ((up and self.m_TYPE[e.val] == 2) or (not up and self.m_TYPE[e.val] == 1)): # 2 = frond, 1 = tree
                self.GC.reverseEdge(e)

        self.m_A = [[] for v in self.GC.nodes]
        self.m_IN_ADJ = {e.val: None for e in self.GC.edges}
        self.buildAcceptableAdjStruct(self.GC)

        # TODO: DFS2 ....

    # TODO:
    def GraphCopySimple(self, G):
        return G

    # def remove_multi_edges(self, E):
    #     E_prime = []
    #     C = []

    #     i = 0
    #     while i < len(E):
    #         l = 1
    #         while i+l < len(E) and E[i] == E[i+l]:
    #             l+= 1

    #         C.append(E[i] * l)
    #         E_prime.append(E[i])
    #         i += l

    #     return E_prime, C

    # def bucket_sort(self, V, E, l, r):
    #     # sort edges
    #     buckets = {v: [] for v in V}
    #     for (v, w) in E:
    #         buckets[l(v, w)].append(r(v, w))
    #     E = []
    #     for v in buckets:
    #         for w in buckets[v]:
    #             E.append((l(v, w), r(v, w)))
    #     return E

    # TODO: Recode
    def split_off_multi_edges(self):
        # sort edges
        V, E = self.GC.nodes, self.GC.edges

        edges = []
        minIndex = {e.val: None for e in self.GC.edges}
        maxIndex = {e.val: None for e in self.GC.edges}

        # E = self.bucket_sort(V, E, lambda x, y: y, lambda x, y: x)
        # E = self.bucket_sort(V, E, lambda x, y: x, lambda x, y: y)
        self.parallelFreeSortUndirected(self.GC, edges, minIndex, maxIndex)

        # Remove multi edges:
        it = 0
        while it < len(edges):
            e = edges[it]
            minI = minIndex[e]
            maxI = maxIndex[e]
            it += 1
            if (it < len(edges) and minI == minIndex[edges[it]] and maxI == maxIndex[edges[it]]):
                C = newComp()
                C.m_type = 0 # bond

                C.append(self.GC.newEdge(e.source(), e.target()))
                C.append(e)
                C.append(edges[it])

                self.m_TYPE[e.val] = 3 # removed
                self.m_TYPE[edges[it].val] = 3 # removed

                it += 1
                while it < len(edges) and minI == minIndex[edges[it]] and maxI == maxIndex[it]:
                    C.append(edges[it])
                    self.m_TYPE[edges[it]] = 3 # removed
                    it += 1
            it += 1

    # def bucketSort(l, h, f):
    #     // if less than two elements, nothing to do
    #     if (m_head == m_tail) {
    #             return;
    #     }

    #     Array<SListElement<E>*> head(l, h, nullptr), tail(l, h);

    #     SListElement<E>* pX;
    #     for (pX = m_head; pX; pX = pX->m_next) {
    #             int i = f.getBucket(pX->m_x);
    #             if (head[i]) {
    #                     tail[i] = (tail[i]->m_next = pX);
    #             } else {
    #                     head[i] = tail[i] = pX;
    #             }
    #     }

    #     SListElement<E>* pY = nullptr;
    #     for (int i = l; i <= h; i++) {
    #             pX = head[i];
    #             if (pX) {
    #                     if (pY) {
    #                             pY->m_next = pX;
    #                     } else {
    #                             m_head = pX;
    #                     }
    #                     pY = tail[i];
    #             }
    #     }

    #     m_tail = pY;
    #     pY->m_next = nullptr;

    def bucketSort(self, edges, min_v, max_v, f):
        bucket = {v: [] for v in range(min_v, max_v+1)}

        for e in edges:
            bucket[f(e)].append(e) # f.getBucket()

        i = 0
        for j in range(min_v, max_v+1):
            edges[i] = bucket[j]
            i += 1

        return bucket
        
    
    def parallelFreeSortUndirected(self, G, edges, minIndex, maxIndex):
        G.allEdges(edges);

        for e in G.edges:
            srcIndex = e.source().index()
            tgtIndex = e.target().index()
            if (srcIndex <= tgtIndex):
                minIndex[e] = srcIndex
                maxIndex[e] = tgtIndex
            else:
                minIndex[e] = tgtIndex
                maxIndex[e] = srcIndex

        bucketMin = BucketEdgeArray(minIndex)
        bucketMax = BucketEdgeArray(maxIndex)

        edges.bucketSort(0, G.maxNodeIndex(), bucketMin)
        edges.bucketSort(0, G.maxNodeIndex(), bucketMax)


    def buildAcceptableAdjStruct(self, G):
        max_v = 3 * len(G.nodes) + 2
        BUCKET = {i: [] for i in range(1, max_v+1)}

        for e in G.edges:
            t = self.m_TYPE[e.val]
            if t == 3: # 3 = removed
                continue

            w = e.target()
            phi = 3 * self.m_NUMBER[w.val] + 1 if (t == 2) else (3 * self.m_LOWPT1[w.val] if (self.m_LOWPT2[w.val] < self.m_NUMBER[e.source().val]) else 3 * self.m_LOWPT1[w.val] + 2) # 2 == frond
            BUCKET[phi].append(e)


        for i in range(1, max_v+1):
            for e in BUCKET[i]:
                self.m_IN_ADJ[e.val] = self.m_A[e.source().val].append(e)

    # def TSTACK_push(self, h, a, b):
    #     self.m_TSTACK.append((h,a,b))

    # def TSTACK_pushEOS(self):
    #     self.m_TSTACK.append(None) # EOS

    # def TSTACK_notEOS(self):
    #     return self.m_TSTACK[-1] != None # EOS

    # def newComp():
    #     c = Comp()
    #     self.m_component.append(c)
    #     return self.m_component[-1]


    # # Update functions
    # def new_component(es):
    #     # comp = []
    #     # for e in es:
    #     #     E.remove(e)
    #     #     comp.append(e)
    #     # C.append(comp)
    #     pass

    # def edge_list_union(C, es):
    #     # comp = []
    #     # for e in es:
    #     #     E.remove(e)
    #     #     comp.append(e)
    #     # C += es
    #     pass

    # # end of update functions
    # def phi(e):
    #     if e == arrow(v, w) and lowpt2[w] < v:
    #         return 3 * lowpt1[w]
    #     elif e == hook_arrow(v,w):
    #         return 3 * w + 1
    #     elif e == arrow(v,w) and lowpt2[w] >= v: # should be else
    #         return 3 * lowpt1[w] + 2

    # # support function

    # def firstChild(v):
    #     pass

    # def high(w):
    #     pass

    # def top_stack_bool_triple(stack, f):
    #     return bool(*[f(h,a,b) for (h,a,b) in [stack[-1]]])

    # def top_stack_bool_double(stack, f):
    #     return bool(*[f(x,y) for (x,y) in [stack[-1]]])

    # # Algorithm 5
    # def check_for_type_2_pairs(v, w, TSTACK, ESTACK):
    #     while (v != 1 and (top_stack_bool_triple(TSTACK, lambda h,a,b: a == v) or (deg[w] == 2 and firstChild[w] > w))):
    #         if top_stack_bool_triple(TSTACK, lambda h,a,b: a == v and parent[b] == a):
    #             TSTACK.pop()
    #         else:
    #             e_ab = None
    #             if deg[w] == 2 and firstChild[w] > w:
    #                 C = new_component([])
    #                 (v, w) = ESTACK.pop()
    #                 C.append((v,w))
    #                 (w, b) = ESTACK.pop()
    #                 C.append((w,b))
    #                 e_ = new_virtual_edge(v,x,C)
    #                 if ESTACK[-1] == (v,b):
    #                     e_ab = ESTACK.pop()
    #             else:
    #                 h,a,b = TSTACK.pop()
    #                 C = new_component([])
    #                 while top_stack_bool_double(ESTACK, lambda x,y: a <= x <= h and a <= y <= h):
    #                     if (x,y) == (a,b):
    #                         e_ab = ESTACK.pop()
    #                     else:
    #                         C = edge_list_union(C, ESTACK.pop())
    #                 e_ = new_virtual_edge(a,b,C)
    #             if e_ab != None:
    #                 C = new_component([e_ab, e_])
    #                 e_ = new_virtual_edge(v,b,C)
    #             ESTACK.append(e_)
    #             make_tree_edge(e_, arrow(v, b)) # v -> b
    #             w = b

    # # Algorithm 6
    # def check_for_type_1_pair(v,w, TSTACK, ESTACK):
    #     if lowpt2[w] >= v and lowpt1[w] < v and (parent[v] != 1 or adj_to_a_not_yet_visited_tree_arc(v)):
    #         C = new_component([])
    #         while top_stack_bool_double(ESTACK, lambda x,y: w <= x < w + ND[w] or w <= y < w + ND[w]):
    #             C = edge_list_union(C, [ESTACK.pop()])
    #         e_ = new_virtual_edge(v, lowpt1[w]. C)
    #         if ESTACK[-1] == (v,lowpt1[w]):
    #             C = new_component(ESTACK.pop(), e_)
    #             e_ = new_virtual_edge(v, lowpt1[w], C)
    #         if lowpt1[w] != parent[v]:
    #             ESTACK.append(e_)
    #             make_tree_edge(e_, arrow(lowpt1[w], v)) # lowpt1[w] -> v
    #         else:
    #             C = new_component(e_, arrow(lowpt1[w], v)) # lowpt1[w] -> v
    #             e_ = new_virtual_edge(lowpt1[w], v, C)
    #             make_tree_edge(e_, arrow(lowpt1[w], v)) # lowpt1[w] -> v

    # # Algorithm 4
    # def PathSearch(v):
    #     for e in Adj[v]:
    #         if e == arrow(v, w):
    #             # e starts a path
    #             if starts_path(e):
    #                 deleted = []
    #                 while top_stack_bool_triple(TSTACK, lambda h,a,b: a > lowpt1(w)):
    #                     deleted.append(TSTACK.pop())
    #                 if not deleted:
    #                     TSTACK.append((w + ND[w] - 1, lowpt1[w], v))
    #                 else:
    #                     y = max(h for (h,a,b) in deleted)
    #                     h,a,b = deleted[-1]
    #                     TSTACK.append((max(y,w + ND[w] - 1), lowpt1[w], b))
    #                 TSTACK.append(None)
    #             PathSearch(w)
    #             ESTACK.append((v,w)) # v -> w
    #             # check for type 2 pairs
    #             check_for_type_2_pairs(v, w, TSTACK, ESTACK)
    #             # check for a type 1 pair
    #             check_for_type_1_pair(v, w, TSTACK, ESTACK)

    #             if starts_path(e):
    #                 # pop until and including EOS
    #                 while TSTACK[-1] != None:
    #                     TSTACK.pop()
    #                 TSTACK.pop()
    #             while top_stack_bool_triple(TSTACK, lambda h,a,b: a != v and b != v and high[v] > h):
    #                 TSTACK.pop()
    #         else:
    #             e = hook_arrow(v,w) # v \-> w
    #             if starts_path(e):
    #                 deleted = []
    #                 while top_stack_bool_triple(TSTACK, lambda h,a,b: a > w):
    #                     deleted.append(TSTACK.pop())
    #                 if not deleted:
    #                     TSTACK.append((v,w,v))
    #                 else:
    #                     y = max(h for (h,a,b) in deleted)
    #                     h,a,b = deleted[-1]
    #                     TSTACK.append((y,w,b))
    #             if w == parent[v]:
    #                 C = new_component([e, arrow(w, v)]) # w -> v
    #                 e_ = new_virtual_edge(w,v,C)
    #                 make_tree_edge(e_, arrow(w,v)) # w -> v
    #             else:
    #                 ESTACK.append(e)

    # # Algorithm 3
    # def find_split_components():
    #     TSTACK = []
    #     ESTACK = []

    #     TSTACK.push(None) # EOS

    #     PathSearch(1)

    #     C = new_component(ESTACK)

    # computes NUMBER, FATHER, LOWPT 1 and 2, ND, TYPE and DEGREE
    def DFS1(self, G, v, u):
        self.m_numCount += 1
        self.m_NUMBER[v.val] = self.m_numCount

        self.m_FATHER[v.val] = u
        self.m_DEGREE[v.val] = v.degree() # m_indeg + m_outdeg

        self.m_LOWPT1[v.val] = self.m_NUMBER[v.val]
        self.m_LOWPT2[v.val] = self.m_NUMBER[v.val]

        self.m_ND[v.val] = 1

        for adj in v.adjEntries:
            e = adj.theEdge()
            if self.m_TYPE[e.val] != 0: # 0 = unseen
                continue

            w = e.opposite(v)
            if (self.m_NUMBER[w.val] == 0):
                self.m_TYPE[e.val] = 1 # 1 = tree

                self.m_TREE_ARC[w.val] = e

                self.DFS1(G, w, v)

                if self.m_LOWPT1[w.val] < self.m_LOWPT1[v.val]:
                    self.m_LOWPT2[v.val] = min(self.m_LOWPT1[v.val], self.m_LOWPT2[w.val])
                    self.m_LOWPT1[v.val] = self.m_LOWPT1[w.val]
                elif m_LOWPT1[w.val] == m_LOWPT1[v.val]:
                    self.m_LOWPT2[v.val] = min(self.m_LOWPT2[v.val], self.m_LOWPT2[w.val])
                else:
                    self.m_LOWPT2[v.val] = min(self.m_LOWPT2[v.val], self.m_LOWPT1[w.val])

                self.m_ND[v.val] += self.m_ND[w.val]

            else:
                self.m_TYPE[e.val] = 2 # 2 = frond

                if self.m_NUMBER[w.val] < self.m_LOWPT1[v.val]:
                    self.m_LOWPT2[v.val] = self.m_LOWPT1[v.val]
                    self.m_LOWPT1[v.val] = self.m_NUMBER[w.val]

                elif self.m_NUMBER[w.val] > self.m_LOWPT1[v.val]:
                    self.m_LOWPT2[v.val] = min(self.m_LOWPT2[v.val], self.m_NUMBER[w.val])


    # def build_triconnected_components(V, E, C):
    #     for i in range(len(C)):
    #         # Ci <> Ø and Ci is a bond or polygon
    #         if C[i] != [] and bond_or_polygon(C[i]):
    #             for e in C[i]:
    #                 # exists j <> i with e in Cj and type(Ci) = type(Cj)
    #                 if j != i and e in C[j] and Ctype[i] == Ctype[j]:
    #                     C[i] = (C[i] + C[j]).remove(e)
    #                     C[j] = []



# 0   1
#  2 3
#  4 5
# 6   7
G = Graph(list(Node(i) for i in range(8)),[Edge(e) for e in [(7,6),(7,5),(6,4),(2,3),(3,5),(4,2),(6,0),(0,1),(1,7),(0,2),(1,3),(5,4),(4,2)]])

tree = SPQR_tree(G)
print ([e.val for e in tree.GC.edges])
print ([v.val for v in tree.GC.nodes])

# V, E, C = tree.split_off_multi_edges(G[0], G[1])
# find_split_components()

# print (build_triconnected_components(V, E, C))

print ("SPQR tree")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('SPQR.png')
