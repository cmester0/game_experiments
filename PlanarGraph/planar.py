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

def st_numbering(vertices, edges):
    # mark s, t and {s,t} old, all other vertices and edges new
    s, t = next(iter(edges))

    edge_map = {v: set() for v in range(len(vertices))}
    for u,v in edges:
        edge_map[u].add(v)
        edge_map[v].add(u)

    print ("EM:", edge_map)

    old_edges = set()
    old_vertices = set()

    cycle_edges_v_w = {v: [] for v in range(len(vertices))}
    cycle_edges_w_v = {v: [] for v in range(len(vertices))}
    tree_edges_v_w = {v: [] for v in range(len(vertices))}
    tree_edges_u_v = {v: [] for v in range(len(vertices))}
    edge_v_w = {v: [] for v in range(len(vertices))}

    dfs_number = [0 for v in range(len(vertices))]
    def dfs(v):
        dfs_number[v] = max(dfs_number) + 1

        for w in edge_map[v]:
            if dfs_number[w] != 0:
                continue

            tree_edges_v_w[v].append(w)
            # tree_edges_u_v[v].append(w)

            dfs(w)
    dfs(s)

    print ("DFS", dfs_number)

    def pathfinder(v):
        # there is a new cycle edge (v, w) with w ancestor of v (w -*> v) (multi parent)
        if len(cycle_edges_v_w[v]) > 0:
            w = cycle_edges_v_w[v][0]
            old_edges.add((v, w))
            path = [(v,w)]
            # old_vertices.add(v)
            # old_vertices.add(w)
        # there is a new tree edge (v, w)
        elif len(tree_edges_v_w[v]) > 0:
            w = tree_edges_v_w[v][0]
            old_edges.add((v, w))
            path = [(v,w)]
            while not w in old_vertices:
                # find new edge {w, x} with (x = L(w) or L(x) = L(w))
                old_vertices.add(v)
                old_edges.add((w, x))
                path.append((w,x))
                w = x
        # there is a new cycle edge (v, w) with v ancestor of w (v -*> w)
        elif len(cycle_edges_w_v[v]) > 0:
            w = tree_edges_v_w[v][0]
            old_edges.add((v, w))
            path = [(v,w)]
            while not w in old_vertices:
                # find new edge {w, x} with (x -> w)
                old_vertices.add(v)
                old_edges.add((w, x))
                path.append((w,x))
                w = x
        else:
            path = []

        return path

    old_vertices.add(s)
    old_vertices.add(t)
    old_edges.add((s,t))

    # # initialize stack to contain s, t on top of it
    stack = []
    stack.append(t)
    stack.append(s)

    st_number = []
    while stack:
        v = stack.pop()
        vs = pathfinder(v)
        if vs != []:
            for vk in vs[::-1]:
                stack.append(vk)
            stack.append(v)
        else:
            st_number.append(v)

    print ("S,T Numbers:", st_number)



st_numbering(vertices, edges)

# def planar(vertices, edges):
#     for v in range(2, n):

