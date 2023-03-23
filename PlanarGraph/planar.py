import random
import os
import colorsys
import math
from PIL import Image
import heapq

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

vertices, edges = random_graph(10, 2)
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

    print ("EM:", edge_map)

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

    print (s,t)
    print ("DFS", dfs_number)
    print (tree_edge)
    print (back_edge)
    print (low_node)
    print (parrent)
    print (preorder)

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
            print ("before (before",parrent[v], v,")", L)
            L.insert(pvi, v)
            print ("after (before)", L)
            sign[parrent[v]] = 1
        else: # sign(low(v)) == 1
            L.insert(pvi+1, v)
            sign[parrent[v]] = -1
    print (sign)
    print (L)

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

def planar ():



# st_numbering(vertices, edges)

# def planar(vertices, edges):
#     for v in range(2, n):

