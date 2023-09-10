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

width = 2 * 1000 # int(input("Initial width: "))
height = 2 * 1000 # int(input("Initial height: "))

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

def print_graph_scaled(vertices, edges):
    px = list(map(lambda x: x[0], vertices))
    py = list(map(lambda x: x[1], vertices))
    V = [(int((x - min(px)) * (width-10) / (max(px) - min(px))),int((y - min(py)) * (height-10) / (max(py) - min(py))),c) for x,y,c in vertices]
    print_graph(V, edges)

###################
# Input (planar) graph #
###################

# 0---\---\
# | \   \   \
# |  2--3--4
# | /   /   /
# 1---/---/

def ForceDirected(G, epsilon, K):
    V, E = G
    area = width * height

    k = math.sqrt(area / len(V))

    t = 1000
    def cool(t):
        return t * 0.999
    iterations = 10000

    def f_a(d): # attraction force
        return d ** 2 / k
    def f_r(d): # repulsion force
        return k ** 2 / d

    disp = [(0,0) for _ in V]
    pos = [(random.randint(0,width),random.randint(0,height)) for _ in V]

    def sub(a,b):
        return (a[0] - b[0], a[1] - b[1])

    def scale(a, c):
        return (a[0] * c, a[1] * c)

    def add(a,b):
        return (a[0] + b[0], a[1] + b[1])

    def size(a):
        return max(math.sqrt(a[0]**2 + a[1]**2),10e-10)

    for i in range(1,iterations+1):
        for v in V:
            disp[v] = (0,0)
            for u in V:
                if u != v:
                    delta = sub(pos[v], pos[u])
                    disp[v] = add(disp[v], scale(delta , 1 / size(delta) * f_r(size(delta))))

        for ev, eu in E:
            delta = sub(pos[ev], pos[eu])
            disp[ev] = sub(disp[ev], scale(delta , 1 / size(delta) * f_a(size(delta))))
            disp[eu] = add(disp[eu], scale(delta , 1 / size(delta) * f_a(size(delta))))

        for v in V:
            pos[v] = add(pos[v] , scale(disp[v], 1/size(disp[v]) * min(size(disp[v]), t)))
            pos[v] = (min(width/2, max(-width / 2, pos[v][0])), min(height/2, max(-height / 2, pos[v][1])))

        t = cool(t)
    return pos

def FRUCHTERMAN_REINGOLD(G):
    V,E = G
    p = ForceDirected((V,E), 10e-10, 10e10)
    return [(x,y,(255,random.randint(0,255),random.randint(0,255)))for x,y in p], E

# G = (list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])
G = (list(range(8)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2)])
V, E = FRUCHTERMAN_REINGOLD(G)
print_graph_scaled(V, E)

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('random.png')

