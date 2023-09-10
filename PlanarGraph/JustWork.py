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
        print_line(x0,y0,x1,y1,c0)

###################
# Input (planar) graph #
###################

# 0---\---\
# | \   \   \
# |  2--3--4
# | /   /   /
# 1---/---/

inp_graph = {0: [1,2,3,4], 1: [0,2,3,4], 2: [0,1,3], 3: [0,1,2,4], 4:[0,1,3]}

#########################
# Make a planar embedding  #
#########################
inp_graph = {0: [4,3,2,1], 1: [0,2,3,4], 2: [0,3,1], 3: [0,4,1,2], 4:[0,1,3]}

###################
# Triangulated  graph #
###################
inp_graph = {0: [4,3,2,1], 1: [0,2,3,4], 2: [0,3,1], 3: [0,4,1,2], 4:[0,1,3]} # already triangulated

#########################
# Draw with schnyder woods #
#########################

# Next function in rotation
def index_of_edge_in_rotation(a,b):
    return inp_graph[b][(inp_graph[b].index(a)+1) % len(inp_graph[b])]

# choose outer face
a = 0
b = inp_graph[a][0]
c = index_of_edge_in_rotation(a,b)

print (index_of_edge_in_rotation(b,c),"=",a) # should be 0, if triangulated
assert (index_of_edge_in_rotation(b,c) == a)

outer_face = [a,b,c]

# Schnyder labelings

def numberOfEdges(G):
    return 7 # Calculated manually for now..

def twinNode(n):
    return inp_graph[n][0]

def adjEntries(n):
    return inp_graph[n]

def contract(G,a,b,c):
    L = []
    candidates = []
    marked = [False for _ in range(len(G))]
    deg = [0 for _ in range(len(G))]

    N = numberOfEdges(G)

    marked[a] = marked[b] = marked[c] = True
    deg[a] = deg[b] = deg[c] = N

    for adj1 in adjEntries(a):
        marked[twinNode(adj1)] = True
        for adj2 in adjEntries(twinNode(adj1)):
            deg[twinNode(adj2)] += 1

    for adj1 in G[a]:
        if deg[twinNode(adj1)] <= 2:
            candidates.append(twinNode(adj1))

    while len(candidates) > 0:
        u = candidates[0]
        candidates = candidates[1:]
        if deg[u] == 2:
            L = [u] + L
            deg[u] = N
            for adj1 in G[u]:
                v = twinNode(adj1)
                deg[v] -= 1
                if not marked[v]:
                    marked[v] = true
                    for adj2 in G[v]:
                        deg[twinNode(adj2)] += 1
                    if deg[v] <= 2:
                        candidates.append(v)
                else:
                    if deg[v] <= 2:
                        candidates.append(v)
    return L

contract(inp_graph, a, b, c)

# 0---\---\
# | \   \   \
# |  2--3--4
# | /   /   /
# 1---/---/

put_point(10,10,(255,255,0))

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('random.png')

