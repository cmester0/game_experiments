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
    for xi in range(point_width):
        for yi in range(point_height):
            resMap[int(x+xi) % (width-point_width)][int(y+yi) % (height-point_height)] = color

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
        resMap[int(x0 + point_width / 2 + (x1 - x0) / steps * i) % (width-point_width)][int(y0 + point_height / 2 + (y1 - y0) / steps * i) % (height-point_height)] = color

def print_graph(vertices, edges):
    for (x_rand, y_rand, c_rand) in vertices:
        put_point(x_rand, y_rand, c_rand)

    for i, j in edges:
        x0, y0, c0 = vertices[i]
        x1, y1, c1 = vertices[j]
        print_line(x0,y0,x1,y1,c0)

edges = set()
# 0-1-3-2
edges.add((0,1))
edges.add((1,3))
edges.add((2,3))
edges.add((0,2))

edges.add((0,4))
edges.add((1,5))
edges.add((3,7))
edges.add((2,6))

edges.add((4,5))
edges.add((5,7))
edges.add((7,6))
edges.add((4,6))

# # 3-12-15-8
# edges.add((3,12))
# edges.add((12,15))
# edges.add((15,8))
# edges.add((8,3))

# edges.add((3,13))
# edges.add((12,13))
# edges.add((13,14))
# edges.add((14,8))
# edges.add((14,15))

# # 2-3-8-9
# edges.add((9,8))
# edges.add((2,9))

# # 9-8-11
# edges.add((8,11))
# edges.add((9,11))

# edges.add((8,10))
# edges.add((9,10))
# edges.add((11,10))


# 0  1
#  45
#  67
# 2  3  12
#     13
#     14
# 9  8  15
#  10
#  11

vertices = [(x, y, (255,random.randint(0,255),0)) for x, y in [random_point() for _ in list(range(max(max(edges))+1))]]

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

# print_graph(vertices, edges)
# print ("Random map")
# output_map()
# resMap = resetMap()

zoom = 1 / 10
while True:
    done = False
    scale = 0.01
    point_charge = 0.08

    for _ in range(100):
        vertices_diff = [(0,0) for _ in range(len(vertices))]

        for va, (vax, vay, ca) in enumerate(vertices):
            for vb, (vbx, vby, cb) in enumerate(vertices):
                if va == vb:
                    continue

                size = math.sqrt((vax - vbx) ** 2 + (vay - vby) ** 2)

                # if size < 10**-8:
                #     continue

                # Columnbs law
                vertices_diff[va] = (vertices_diff[va][0] + point_charge * (vax - vbx) / size ** 3,
                                     vertices_diff[va][1] + point_charge * (vay - vby) / size ** 3)
                vertices_diff[vb] = (vertices_diff[vb][0] + point_charge * (vbx - vax) / size ** 3,
                                     vertices_diff[vb][1] + point_charge * (vby - vay) / size ** 3)

        for va, vb in edges:
            vax, vay, ca = vertices[va]
            vbx, vby, cb = vertices[vb]

            size = math.sqrt((vax - vbx) ** 2 + (vay - vby) ** 2)

            # if size < 10**-8:
            #     continue

            # hooks law
            vertices_diff[va] = (vertices_diff[va][0] + min(scale, size) * (vbx - vax) / size,
                                 vertices_diff[va][1] + min(scale, size) * (vby - vay) / size)
            vertices_diff[vb] = (vertices_diff[vb][0] + min(scale, size) * (vax - vbx) / size,
                                 vertices_diff[vb][1] + min(scale, size) * (vay - vby) / size)

        largest = max(map(lambda x: abs(x[0]) + abs(x[1]), vertices_diff))
        if largest < 10**-30:
            done = True
        for a, (x, y, c) in enumerate(vertices):
            vertices[a] = (x + 10 * vertices_diff[a][0],
                           y + 10 * vertices_diff[a][1],
                           c)
        if random.randint(0, 3000) == 0:
            a = random.randint(0, len(vertices)-1)
            vertices[a] = (vertices[a][0] + random.randint(-10, 10), vertices[a][1] + random.randint(-10, 10), vertices[a][2])

    print_graph([(x / zoom,
                  y / zoom, c) for x, y, c in vertices], edges)
    output_map()
    print (map_iter,"\r",end="")
    resMap = resetMap()

    if done:
        break
