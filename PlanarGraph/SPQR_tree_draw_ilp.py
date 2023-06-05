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
    for i in range(steps):
        resMap[int(x0 + point_width / 2 + (x1 - x0) / steps * i)][int(y0 + point_height / 2 + (y1 - y0) / steps * i)] = color

def print_square(x0, y0, x1, y1, color):
    print_line(x0,y0,x0,y1,color)
    print_line(x1,y0,x1,y1,color)
    print_line(x0,y0,x1,y0,color)
    print_line(x0,y1,x1,y1,color)

def print_filled_square(x, y, w, h, color):
    print (x, y, w, h)
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


SPQR_tree = {}

# WIKI pedia example
# SPQR_tree[0] = ("S", [6,7,9,8], [(1,(6,7)),(2,(7,9)),(3,(9,8))], [(6,8)])
# SPQR_tree[1] = ("R", [0,1,7,6,2,3,5,4], [], [(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(5,7),(6,4),(2,3),(3,5),(5,4),(4,2)])
# SPQR_tree[2] = ("R", [7,13,12,9,14,15], [], [(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(12,15),(9,15),(14,15)])
# SPQR_tree[3] = ("P", [8,9], [(4,(8,9))], [(8,9),(8,9)])
# SPQR_tree[4] = ("R", [8,9,11,10], [], [(8,9),(9,11),(11,8),(8,10),(9,10),(11,10)])

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

SPQR_tree[0] = ("S", [0,1,2,3,4], [(1,(0,1)),(2,(1,2)),(3,(2,3)),(4,(3,4))], [(0,4)])

SPQR_tree[1] = ("S", [5,6,7], [(5,(0,5)),(6,(5,6)),(7,(6,1))], [])
SPQR_tree[5] = ("Q", [0, 5], [], [(0,5)])
SPQR_tree[6] = ("Q", [5, 6], [], [(5,6)])
SPQR_tree[7] = ("Q", [6, 1], [], [(6,1)])

SPQR_tree[2] = ("S", [1,7,2], [(8,(1,7)), (9, (7, 2))], [])
SPQR_tree[8] = ("Q", [1,7], [], [(1,7)])
SPQR_tree[9] = ("Q", [7,2], [], [(7,2)])

SPQR_tree[3] = ("P", [2, 3], [(10, (2,3)), (11, (2,3)), (12, (2,3))], [(2,3)])
SPQR_tree[10] = ("S", [2, 8, 3], [(13, (2,8)), (14, (8,3))], [])
SPQR_tree[13] = ("Q", [2, 8], [], [(2,8)])
SPQR_tree[14] = ("Q", [8, 3], [], [(8,3)])

SPQR_tree[11] = ("Q", [2, 3], [], [(2,3)])

SPQR_tree[12] = ("S", [2, 9, 3], [(15, (2,9)), (16, (9,3))], [])
SPQR_tree[15] = ("Q", [2, 9], [], [(2,9)])
SPQR_tree[16] = ("Q", [9, 3], [], [(9,3)])

SPQR_tree[4] = ("Q", [3, 4], [], [(3,4)])

print_graph([], [])
print_square(0,0,width - point_width, height - point_height, (255,255,255))

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name="small-problem", sense=LpMaximize)


print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('random.png')
