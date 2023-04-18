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

width = 2 * 300 # int(input("Initial width: "))
height = 2 * 300 # int(input("Initial height: "))

def resetMap():
    resMap = []
    for xi in range(width):
        resMap.append([])
        for yi in range(height):
            resMap[xi].append((0,0,0))
    return resMap
resMap = resetMap()

point_width = 10
point_height = 10

def put_point(x,y,color):
    for xi in range(-int(point_width/2), int(point_width/2)):
        for yi in range(-int(point_height/2), int(point_height/2)):
            resMap[int(x+xi) % (width-point_width+1)][int(y+yi) % (height-point_height+1)] = color

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
    for i in range(int(steps)):
        resMap[int(x0 + (x1 - x0) / steps * i) % (width-point_width+1)][int(y0 + (y1 - y0) / steps * i) % (height-point_height+1)] = color

def print_dashed_line(x0,y0,x1,y1, color):
    steps = abs(x0 - x1) + abs(y0 - y1)
    for i in range(int(steps)):
        for j in range(0,int(steps / 8),2):
            if j * steps / 8 < i < (j+1) * steps / 8:
                break
        else:
            continue

        resMap[int(x0 + (x1 - x0) / steps * i) % (width-point_width+1)][int(y0 + (y1 - y0) / steps * i) % (height-point_height+1)] = color

def print_graph(vertices, edges, virtual_edges=[]):
    for index in vertices:
        (x_rand, y_rand, c_rand) = vertices[index]
        put_point(x_rand, y_rand, c_rand)

    for i, j in edges:
        x0, y0, c0 = vertices[i]
        x1, y1, c1 = vertices[j]
        print_line(x0,y0,x1,y1,c0)

    for i, j in virtual_edges:
        x0, y0, c0 = vertices[i]
        x1, y1, c1 = vertices[j]
        print_dashed_line(x0,y0,x1,y1,c0)

vertices = [(x, y, (255,random.randint(0,255),0)) for x, y in [random_point() for _ in list(range(16))]]

SPQR_tree = []
SPQR_tree.append(("S", [2,3,9,8], [(2,3),(3,8),(8,9)], [(2,9)]))
SPQR_tree.append(("R", [0,1,2,3,4,5,6,7], [(2,3)], [(0,1),(1,3),(2,0),(0,4),(1,5),(3,7),(2,6),(6,4),(4,5),(5,7),(7,6)]))
SPQR_tree.append(("R", [3,12,13,14,15,8], [(3,8)], [(3,12),(12,15),(15,8),(8,14),(14,15),(3,13),(12,13),(13,14)]))
SPQR_tree.append(("P", [9,8], [(9,8),(9,8)], [(9,8)]))
SPQR_tree.append(("R", [9,8,10,11], [(9,8)], [(9,10),(9,11),(8,11),(10,11),(8,10)]))

# 0  1
#  45
#  67
# 2  3  12
#     13
#     14
# 9  8  15
#  10
#  11

# vertices, edges = random_graph(10, 100)

map_iter = 0
def output_map():
    global map_iter

    flat_m = []
    for yi in range(height):
        for xi in range(width):
            flat_m.append(resMap[xi][yi])

    img = Image.new('RGB', (width, height)) # width, height
    img.putdata(flat_m)
    img.save('out/random' + str(map_iter) + '.png')
    map_iter += 1

vertices = dict()
edges = []

for j, (t, vert, virt, edg) in enumerate(SPQR_tree):
    print (j)
    sub_verts = dict()
    one_width = (width-point_width)/len(SPQR_tree)
    for vj, v in enumerate(vert):
        cx, cy = j * one_width + one_width / 2, height / 2
        r = one_width / 3
        x_pos = cx + r * math.cos(vj / len(vert) * 2 * math.pi)
        y_pos = cy + r * math.sin(vj / len(vert) * 2 * math.pi)

        # print ("V", v)
        # x_rand = random.randint(int(j * one_width), int((j+1) * one_width))
        # y_rand = random.randint(0, height-point_height)

        # if not v in vertices:
        sub_verts[v] = (int(x_pos), int(y_pos), (255, j * 80, 0))
        # else:
        #     sub_verts[v] = vertices[v]

        if not v in vertices:
            vertices[v] = (int(x_pos), int(y_pos), (255, j * 80, 0))

    print_line(j*one_width,0,
               j*one_width,height-point_height,
               (255,255,255))

    print_line(    j*one_width,0,
               (j+1)*one_width,0,
               (255,255,255))

    print_line(    j*one_width,height-point_height,
               (j+1)*one_width,height-point_height,
               (255,255,255))

    print_line((j+1)*one_width,0,
               (j+1)*one_width,height-point_height,
               (255,255,255))

    print_graph(sub_verts, edg, virt)

    for a,b in edg:
        edges.append((a,b))

# print_graph(vertices, edges)
output_map()
resMap = resetMap()
