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

def random_SPQR_tree(max_tree_nodes = 10, max_vertices_per = 7):
    SPQR_tree = {}

    from_node = 0
    to_node = 1
    curr_node = 2

    tree_nodes = random.randint(2,max_tree_nodes)

    for i in range(tree_nodes):
        tree_type = ["S","P","R"][random.randint(0,2)]

        vertices = [from_node, to_node]
        if not (tree_type == "P" or tree_type == "Q"):
            for _ in range(random.randint(1,max_vertices_per)):
                vertices.append(curr_node)
                curr_node += 1

        edges = []
        for a,b in zip(vertices, vertices[1:]):
            edges.append((a,b))

        SPQR_tree[i] = (tree_type, vertices, [], edges)

        offspring = random.randint(0,i)
        while (len(SPQR_tree[offspring][3]) == 0):
            offspring = random.randint(0,i)

        offspring_edge = random.randint(0,len(SPQR_tree[offspring][3])-1)
        from_node, to_node = SPQR_tree[offspring][3][offspring_edge]
        # del SPQR_tree[offspring][3][offspring_edge]
        SPQR_tree[offspring][2].append((i+1,(from_node, to_node)))

    for i in range(tree_nodes):
        print (i,SPQR_tree[i][0], SPQR_tree[i][1],SPQR_tree[i][2], SPQR_tree[i][3])
    return SPQR_tree

# random_SPQR_tree()

SPQR_tree = {}

# # WIKI pedia example
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

SPQR_tree[0] = ("S", [0,1,2,3,4], [(1,(0,1)),(2,(1,2)),(3,(2,3))], [(0,1),(1,2),(2,3),(3,4),(0,4)])

#     4
#  /    \
# 3      0
#  \     /
#   2 - 1

SPQR_tree[1] = ("S", [0,5,6,7,1], [(0,(0,1))], [(0,1),(0,5),(5,6),(6,7),(1,7)])

#     1
#  /    \
# 7      0
#  \     /
#   6 - 5

SPQR_tree[2] = ("S", [1,8,9,2], [(0,(1,2))], [(1,2),(1,8),(8,9),(2,9)])

#      8
#     / \
#    1   9
#     \ /
#      2

SPQR_tree[3] = ("P", [2, 3], [(4, (2,3)), (5, (2,3))], [(2,3),(2,3),(2,3)])

#       __
#      /  \
#     2 -- 3
#      \__/

SPQR_tree[4] = ("S", [2, 10, 3], [(3, (2,3))], [(2,3), (2,10), (3,10)])
SPQR_tree[5] = ("S", [2, 11, 3], [(3, (2,3))], [(2,3), (2,11), (3,11)])

vertices = []
edges = []

vt, et = dict(), []
def box_xy_from_i(i, m):
    if i <= m:
        return i, 0
    if m < i <= 2 * m:
        return m, i - m
    if 2 * m < i <= 3 * m:
        return m - (i - 2 * m), m
    return 0, m - (i - 3 * m)

visited = set()
drawn_ve = dict()
# drawn_ve[(0,1)] = ((width / 2 - 100, height / 2), (width / 2 + 100, height / 2), 0)
def draw_skeleton(j, c, angle, radius, parent = (-1, -1)):
    if j in visited:
        return
    visited.add(j)

    cx, cy = c

    tree_type, vertices, ve, e = SPQR_tree[j]
    ve_edges = set(map(lambda x: x[1], ve))

    if tree_type == "P":
        x1 = radius * math.cos(0 + angle) + cx
        y1 = radius * math.sin(0 + angle) + cy

        x2 = radius * math.cos(math.pi + angle) + cx
        y2 = radius * math.sin(math.pi + angle) + cy

        if (len(e) > len(ve)): # Actual edge exists
            print_line(x1, y1, x2, y2, (255,0,0)) # Actual edge
        drawn_ve[e[0]] = ((x1, y1), (x2, y2), math.atan2(y1 - y2, x1 - x2)) # (math.pi * ((len(e) - i + angle) + 4)) / (4 * len(e)))

    elif tree_type == "S":
        cel = list(filter(lambda x: not (x == parent), e)) # circle edge list
        ces = len(cel) # circle edge size
        ta = 2 * math.pi if len(cel) == len(e) else math.pi # total_angle

        for i, (a,b) in enumerate(cel):
            x1 = radius * math.cos(ta * i / ces + angle) + cx
            y1 = radius * math.sin(ta * i / ces + angle) + cy

            x2 = radius * math.cos(ta * (i+1) / ces + angle) + cx
            y2 = radius * math.sin(ta * (i+1) / ces + angle) + cy

            if not (a,b) in ve_edges:
                print_line(x1, y1, x2, y2, (255,0,0))
            else:
                drawn_ve[(a,b)] = ((x1, y1), (x2, y2), math.atan2(y1 - y2, x1 - x2)) # (math.pi * ((len(e) - i + angle) + 4)) / (4 * len(e)))
                # print_line(x1, y1, x2, y2, (0,0,255)) # Virtual edge

    if tree_type == "S" or tree_type == "R":
        for k, (a,b) in ve:
            x1 = drawn_ve[(a,b)][0][0]
            y1 = drawn_ve[(a,b)][0][1]

            x2 = drawn_ve[(a,b)][1][0]
            y2 = drawn_ve[(a,b)][1][1]
            draw_skeleton(k,
                          ((x1 + x2) / 2, (y1 + y2) / 2),
                          drawn_ve[(a,b)][2],
                          math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2,
                          (a,b))
    elif tree_type == "P":
        for off, (k, (a,b)) in enumerate(ve):
            x1 = drawn_ve[e[0]][0][0]
            y1 = drawn_ve[e[0]][0][1]

            x2 = drawn_ve[e[0]][1][0]
            y2 = drawn_ve[e[0]][1][1]
            draw_skeleton(k,
                          ((x1 + x2) / 2, (y1 + y2) / 2),
                          2 * math.pi * (off / len(ve)) + drawn_ve[e[0]][2],
                          math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2,
                          (a,b))

    print (tree_type, vertices, ve, e)

draw_skeleton(0, (width / 2, height / 2), 0, 100, (-1, -1))
# draw_skeleton(1, dict())

print_square(0,0,width - point_width, height - point_height, (255,255,255))

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('random.png')

