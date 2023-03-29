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

width = 2 * 200 # int(input("Initial width: "))
height = 2 * 200 # int(input("Initial height: "))

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

def random_graph(V_size, E_chance = 1):
    vertices = random_points_list(V_size)
    edges = set()

    for i in range(V_size):
        # Have 2 edges out of each vertex
        for k in range(2):
            p = random.randint(0,V_size-1)
            while p == i:
                p = random.randint(0,V_size-1)
            edges.add((i,p))

        for j in range(i+1, V_size):
            if random.randint(0,E_chance) == 0:
                edges.add((i,j))

    return vertices, edges

def print_line(x0,y0,x1,y1, color):
    steps = abs(x0 - x1) + abs(y0 - y1)
    for i in range(steps):
        resMap[int(x0 + point_width / 2 + (x1 - x0) / steps * i)][int(y0 + point_height / 2 + (y1 - y0) / steps * i)] = color

def print_graph(vertices, edges):
    for (x_rand, y_rand, c_rand) in vertices:
        put_point(x_rand, y_rand, c_rand)

    for i, j in edges:
        print (i,j)
        x0, y0, c0 = vertices[i]
        x1, y1, c1 = vertices[j]
        print_line(x0,y0,x1,y1,c0)

vertices, edges = random_graph(10, 100)
print_graph(vertices, edges)

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('random.png')

def st_numbering(vertices, edges, s, t):
    # mark s, t and {s,t} old, all other vertices and edges new

    edge_map = {v: list() for v in range(len(vertices))} # list should be set
    # edge_map[t].append(s) # make sure dfs visits s first
    for u,v in edges:
        edge_map[u].append(v)
        edge_map[v].append(u)

    # print ("EM:", edge_map)

    old_edges = set()
    old_vertices = set()

    tree_edge = {v: [] for v in range(len(vertices))}
    back_edge = {v: [] for v in range(len(vertices))}
    low_node = [0 for v in range(len(vertices))]

    dfs_number = [0 for v in range(len(vertices))]

    parrent = [0 for v in range(len(vertices))]

    pre = {v: 0 for v in range(len(vertices))}
    current = 1
    pre[s] = current

    preorder = []

    def dfs(v):
        dfs_number[v] = max(dfs_number) + 1
        preorder.append(v)

        low = (dfs_number[v], v)
        for w in edge_map[v]:
            if dfs_number[w] != 0:
                if not v in back_edge[w] and not w in back_edge[v] and not v in tree_edge[w] and not w in tree_edge[v]:
                    back_edge[v].append(w)
                    low = min(low, (dfs_number[w], w)) # min w for (v,w) is back edge
                continue

            tree_edge[v].append(w)
            parrent[w] = v

            low = min(low, dfs(w)) # min of low(tree edge)

        low_node[v] = low[1]
        return low
    dfs(s) # Get dfs spanning tree.

    # print ("S<T",s,t)
    # print ("DFS", dfs_number)
    # print (tree_edge)
    # print (back_edge)
    # print (low_node)
    # print (parrent)
    # print (preorder)

    L = [s,t]

    sign = dict()
    sign[s] = -1

    for _, v in list(sorted(zip(dfs_number,range(len(vertices)))))[2:]: # for v in preorder
        pvi = 0
        for i, pv in enumerate(L):
            if pv == parrent[v]:
                pvi = i
                break
        if sign[low_node[v]] == -1:
            # print ("before (before",parrent[v], v,")", L)
            L.insert(pvi, v)
            # print ("after (before)", L)
            sign[parrent[v]] = 1
        else: # sign(low(v)) == 1
            L.insert(pvi+1, v)
            sign[parrent[v]] = -1
    # print (sign)
    # print (L)

    return {v: i for i,v in enumerate(L)}

# edges = []
# edges.append((0,9)) # s to t
# edges.append((9,0))

# edges.append((8,7)) # g to h
# edges.append((7,8))

# edges.append((6,7)) # g to f
# edges.append((7,6))


# edges.append((0,4))
# edges.append((0,2))
# edges.append((0,3))
# edges.append((0,1))

# edges.append((1,0))
# edges.append((1,2))
# edges.append((1,5))

# edges.append((2,0))
# edges.append((2,1))
# edges.append((2,6))

# edges.append((3,0))
# edges.append((3,4))
# edges.append((3,6))

# edges.append((4,0))
# edges.append((4,3))
# edges.append((4,7))

# edges.append((5,1))
# edges.append((5,7))

# edges.append((6,2))
# edges.append((6,3))

# edges.append((7,4))
# edges.append((7,5))
# edges.append((7,9))

# edges.append((8,9))

# edges.append((9,7))
# edges.append((9,8))

# nums = st_numbering(list(range(10)), edges, 0, 9) # 0 = s, 9 = t, 1..8 = a..h

# print (nums)

def virtual_edges (edges, st_nums):
    ret_edges = set()
    for a,b in edges:
        ret_edges.add((st_nums[a], st_nums[b]))
    return ret_edges

def Gk (edges, k):
    Gk_edge = set()
    cross_edge = set()
    for a, b in edges:
        if a < k:
            if b < k:
                Gk_edge.add((a,b))
            else:
                cross_edge.add((a,b))
        else:
            if b < k:
                cross_edge.add((a,b)) # Reverse direction

    return Gk_edge, cross_edge

def upward_embed(edges):
    pass

# template matchings

def not_pertient():
    return True

# Classifications, is_full, is_empty
def classify_children(P):
    return []

def Tt(U, S): # S subset U, PQ-tree with S being a subnode
    Uc = list(filter(lambda x: x not in U, [i for i in S]))
    if len(Uc) == 0:
        return ("P", [i for i in U]) # full tree
    elif len(U) > 0:
        return ("P", [i for i in U] + [("P", Uc)])
    else:
        return ("nill", []) # nil tree

# GLOBAL DEFINTIONS
mark = defaultdict(lambda: "unmarked")
immediate_siblings = defaultdict(lambda: set())
parent = defaultdict(lambda: None)
pertinent_child_count = defaultdict(lambda: 0)
pertinent_leaf_count = defaultdict(lambda: 0)

def bubble(T,S):
    queue = [] # FIFO
    block_count = 0
    block_nodes = 0
    off_the_top = 0
    for x in S:
        queue.append(x)
    while len(queue) + block_count + off_the_top > 1:
        if len(queue) == 0:
            T = T({}, {}) # nill tree
            break

        x = queue.pop()
        mark[x] = "blocked"
        BS = list(filter(lambda y: mark[y] == "blocked", immediate_siblings[x]))
        US = list(filter(lambda y: mark[y] == "unblocked", immediate_siblings[x]))

        if len(US) > 0:
            y = next(US) # choose any Y in US
            parent[x] = parent[y]
            mark[x] = "unblocked"
        elif len(immediate_siblings[x]) < 2:
            mark[x] = "unblocked"

        if mark[x] == "unblocked":
            y = parent[x]
            if len(BS) > 0:
                LIST = [] # the maximal consecutive set of blocked siblings adjacent to x
                for z in LIST:
                    mark[z] = "unblocked"
                    parent[z] = y
                    pertinent_child_count[y] = pertinent_child_count[y] + 1
            if y == None:
                off_the_top = 1
            else:
                pertinent_child_count[y] = pertinent_child_count[y] + 1
                if mark[y] == "unmarked":
                    queue = [y] + queue
                    mark[y] = "queued"
                block_count = block_count - len(BS)
                blocked_nodes = blocked_nodes - len(LIST)
        else:
            block_count = block_count + 1 - len(BS)
            blocked_nodes = blocked_nodes + 1
    return T

def reduce_T(T,S):
    queue = []
    for x in S:
        queue.append(x)
        pertinent_leaf_count[x] = 1
    while len(queue) > 0:
        x = queue.pop()
        if pertinent_leaf_count[x] < len(S):
            # x is not root(T,S)
            y = parent[x]
            pertinent_leaf_count[y] = pertinent_leaf_count[y] + pertinent_leaf_count[x]
            pertinent_child_count[y] = pertinent_child_count[y] - 1
            if pertinent_child_count[y] == 0:
                queue = [y] + queue
            if (not template_L1(x)
                or not template_P1(x)
                or not template_P3(x)
                or not template_P5(x)
                or not template_Q1(x)
                or not template_Q2(x)):
                print ("no matches")
                T = Tt({}, {}) # TODO nill tree
        else:
            # x is root(T, S);
            if (not template_L1(x)
                or not template_P1(x)
                or not template_P2(x)
                or not template_P4(x)
                or not template_P6(x)
                or not template_Q1(x)
                or not template_Q2(x)
                or not template_Q3(x)):
                T = Tt({}, {}) # TODO nill tree
                break
    return T


def  template_L1(x):
    print ("L1")
    return True
def  template_P1(x):
    return True
def  template_P2(x):
    return True
def  template_P3(x):
    return True
def  template_P4(x):
    return True

def template_P5(x):
    if type[x] != "P":
        return False
    if len(partial_children[x] != 1):
        return False
    Y = partial_children[x][0] # the unique element
    EC = list(filter(lambda x: label[x] == "empty", endmost_children[y]))[0] # the unique element
    FC = list(filter(lambda x: label[x] == "full", endmost_children[y]))[0] # the unique element
    # the following statement may be performed in time on the order of number of pertient children of x through the use of the circular_link fields.
    if True: # Y has an empty sibling
        ES = [] # an empty sibling of Y
    # Y will be the root of the replacement
    parent[y] = parent[x]
    pertinent_leaf_count[y] = pertinent_leaf_count[x]
    label[y] = "partial"
    partial_children[parent[y]] = partial_children[parent[y]] + set([y])
    # remove y from the list of children of x formed by the circular_link fields

    ## TODO ##
    return True

def  template_P6(x):
    return True
def  template_Q1(x):
    return True
def  template_Q2(x):
    return True
def  template_Q3(x):
    return True


def reduction(U,Ss):
    T = Tt(U,U)
    print ("SS", Ss)
    for S in Ss:
        print ("bubble")
        T = bubble(T,S)
        print (T)
        print ("reduce")
        T = reduce_T(T,S)
        print (T)
    return T

def planar (vertices, edges):
    s, t = next(iter(edges))
    nums = st_numbering(list(range(len(vertices))), edges, s, t) # 0 = s, 9 = t, 1..8 = a..h
    print (nums)
    nums = {i: i for i in range(len(vertices))} # Fake numbering to match paper!
    print (nums)

    v_edges = virtual_edges(edges, nums)
    v_verts = list(range(len(vertices)))

    S, G1_ = Gk(edges, 1)
    T = Tt(G1_, G1_)
    print (T)

    for v in range(2,len(vertices)):
        for S in G1_:
            print ("bubble")
            T = bubble(T,S)
            print (T)
            print ("reduce")
            T = reduce_T(T,S)
            print (T)

        
        # while not_pertient():
        #     if not template_matching():
        #         print ("Not planar")
        #         return False
        # # Replace full node of PQ tree by a P-node
        # # Add all neighbord of v larger than v as the sons of the P-node
    print ("G is planar")
    return True

# edges = []
# edges.append((0,4)) # s to t
# edges.append((4,0))

# edges.append((0,1))
# edges.append((1,0))

# edges.append((0,2))
# edges.append((2,0))

# edges.append((3,4))
# edges.append((4,3))

# edges.append((2,3))
# edges.append((3,2))

# edges.append((2,4))
# edges.append((4,2))

# edges.append((1,2))
# edges.append((2,1))

# edges.append((1,3))
# edges.append((3,1))

# edges.append((1,4))
# edges.append((4,1))

# planar(list(range(5)), edges)
# planar(vertices, edges)

edges = []
edges.append((0,5)) # s to t

edges.append((0,1))
edges.append((0,2))
edges.append((0,4))

edges.append((1,4))
edges.append((1,3))
edges.append((1,2))

edges.append((2,5))
edges.append((2,3))

edges.append((3,4))
edges.append((3,5))

edges.append((4,5))

planar(list(range(6)), edges)

# st_numbering(vertices, edges)

# def planar(vertices, edges):
#     for v in range(2, n):
