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

width = 2 * 500 # int(input("Initial width: "))
height = 2 * 500 # int(input("Initial height: "))

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

def scale_graph(vertices):
    px = list(map(lambda x: x[0], vertices))
    py = list(map(lambda x: x[1], vertices))
    return [(int((x - min(px)) * (width-10) / (max(px) - min(px))),int((y - min(py)) * (height-10) / (max(py) - min(py))),c) for x,y,c in vertices]

def print_graph_scaled(G):
    vertices, edges = G
    print_graph(scale_graph(vertices), edges)

###################
# Input (planar) graph #
###################

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

def calculate_outer_face(V,E):
    def n_polygon(n):
        return [(width * math.cos(math.pi*2*i/n), height * math.sin(math.pi*2*i/n)) for i in range(n)]

    def find_outer_face():
        # Cycle is tree + 1 edge (smallest is given by finding the common ancestor)
        stk = [(0,0)]
        visited = [False for _ in V]
        dfs_number = [0 for _ in V]
        dfs_num = 0
        cycles = []
        tree = [0 for _ in V]
        while len(stk) > 0:
            node, n_parrent = stk.pop()
            if visited[node]:
                cycles.append((node, n_parrent))
                continue
            visited[node] = True
            tree[node] = n_parrent

            dfs_number[node] = dfs_num
            dfs_num += 1
            neighbors = calc_neighbors(node,E) # do better!
            stk = stk + [(y,node) for y in neighbors]

        result_cycles = []
        for x, y in cycles:
            x_to_root = [x]
            while x_to_root[-1] != tree[x_to_root[-1]]:
                x_to_root.append(tree[x_to_root[-1]])
            y_to_root = [y]
            while y_to_root[-1] != tree[y_to_root[-1]]:
                y_to_root.append(tree[y_to_root[-1]])
            min_to_root, max_to_root = (x_to_root, y_to_root) if len(x_to_root) < len(y_to_root) else (y_to_root, x_to_root)
            min_to_root = list(reversed(min_to_root))
            max_to_root = list(reversed(max_to_root))
            i, j = 0, 0
            while i < len(min_to_root) and j < len(max_to_root) and min_to_root[i] == max_to_root[j]:
                i += 1
                j += 1
            j -= 1
            i -= 1
            result_set = list(dict.fromkeys(min_to_root[i:] + max_to_root[j:]))

            if len(result_set) > 2:
                result_cycles.append(result_set)

        result_cycles = list(sorted(result_cycles, key=lambda x: (len(x), x)))[::2]
        # for result_set in result_cycles:
        #     print ("C", result_set)

        return result_cycles[0]
        # print (result_cycles[-1])
        # return result_cycles[-1]
        # return result_cycles[random.randint(0,len(result_cycles))]

    outer_face = find_outer_face() # [0,1,7,13,12,9,11,8,6] (SPQR outer)
    outer_n_gon = n_polygon(len(outer_face))
    return outer_face, outer_n_gon

def ForceDirected(G, epsilon, K):
    V, E = G

    V = list(V)
    E = list(E)

    for _ in range(3):
        V.append(V[-1]+1)
    outer_face = list(V[-3:])

    for i in range(3):
        E.append((V[-1-i], V[random.randint(0, len(V)-4)]))
    E.append((V[-1], V[-2]))
    E.append((V[-1], V[-3]))
    E.append((V[-2], V[-3]))

    # outer_face, outer_n_gon = calculate_outer_face(V,E)

    def n_polygon(n):
        return [(width * math.cos(math.pi*2*i/n), height * math.sin(math.pi*2*i/n)) for i in range(n)]

    outer_n_gon = n_polygon(len(outer_face))

    p = [random_point() for _ in V]
    fixed = [False for v in V]

    for i, v in enumerate(outer_face):
        fixed[v] = True
        p[v] = outer_n_gon[i]

    # repulsive force
    def f_rep(u,v):
        return (0,0)
        # if fixed[u]:
        #     return (0,0)
        # else:
        #     diff = add(p[u], p[v])
        #     scalar = -20 / size(diff)
        #     return scale(diff, scalar)

    def deg(u):
        return len(list(filter(lambda x: x[0] == u or x[1] == u, E)))

    def f_attr(u,v):
        if fixed[u]:
            return (0,0)
        else:
            diff = sub(p[u], p[v])
            scalar = -1 / deg(u) # size(diff)
            return scale(diff, scalar)

    F = []
    first = True

    t = 0
    while t < K and (first or max([size(F[t-1][v]) for v in V]) > epsilon):
        first = False
        F.append([(0,0) for v in V])

        for u in V:
            f_res = (0,0)

            for a,b in [f_rep(u,v) for v in V]:
                f_res = (f_res[0] + a, f_res[1] + b)

            for a,b in ([f_attr(u,v) for w,v in E if w == u] + [f_attr(u,v) for v,w in E if w == u]):
                f_res = (f_res[0] + a, f_res[1] + b)

            F[t][u] = f_res

        if max([size(F[t-1][v]) for v in V]) < max([size(F[t][v]) for v in V]):
            break

        for u in V:
            p[u] = (p[u][0] + F[t][u][0], p[u][1] + F[t][u][1])

        t += 1

    V, E = G
    outer_face, outer_n_gon = calculate_outer_face(V, E)
    return p[:-3], outer_face

def tutte_from_graph(G):
    V,E = G
    p, outer = ForceDirected((V,E), 10e-10, 10e10)
    return (p,E), set(outer)

def add_color(G, c = lambda : (255,random.randint(0,255),random.randint(0,255))):
    V,E = G
    return [(x,y,c())for x,y in V], E

# V,E = tutte_from_graph((list(range(5)),[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(3,4)]))
# print_graph_scaled(V, E)

def triangulate_graph(G):
    V,E = G
    # Cycle is tree + 1 edge (smallest is given by finding the common ancestor)
    stk = [(0,0)]
    visited = [False for _ in V]
    dfs_number = [0 for _ in V]
    dfs_num = 0
    cycles = []
    tree = [0 for _ in V]
    while len(stk) > 0:
        node, n_parrent = stk.pop()
        if visited[node]:
            cycles.append((node, n_parrent))
            continue
        visited[node] = True
        tree[node] = n_parrent

        dfs_number[node] = dfs_num
        dfs_num += 1
        neighbors = calc_neighbors(node,E) # do better!
        stk = stk + [(y,node) for y in neighbors]

    Eprime = list(E)

    for n, v in sorted(zip(dfs_number, V)):
        neighbors = list(filter(lambda x: dfs_number[x] > dfs_number[v], sorted(calc_neighbors(v, E), key=lambda x: dfs_number[x]))) # do better!
        if len(neighbors) > 0:
            for w, z in zip(neighbors, neighbors[1:] + [neighbors[0]]):
                if not (w,z) in Eprime and not (z,w) in Eprime and not z == w:
                    Eprime.append((w,z))

    return V,Eprime

# # 0   1
# #  2 3
# #  4 5
# # 6   7  13
# #      14
# #      15
# # 8   9  12
# #  10
# #  11
# # n = 16
# G = (list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])

# 0   1
#  2 3
#  4 5
# 6   7
G = (list(range(8)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2)])

def dual(G, outer):
    V, E = G

    V_ = []
    E_ = []

    Evals = [sorted(([v for u,v in E if u == w] + [v for v,u in E if u == w]), key=lambda x: math.atan2(sub(V[x], V[w])[1], sub(V[x], V[w])[0])) for w in range(len(V))]
    faces = []
    for v in range(len(V)):
        for c in Evals[v]:
            prev = v
            curr = c

            face = set()
            while curr != v:
                face.add(curr)
                prev, curr = curr, Evals[curr][(Evals[curr].index(prev)+1) % len(Evals[curr])]
            face.add(curr)

            if not face in faces and not face == outer:
                faces.append(face)

    for face in faces:
        total = (0,0)
        for f in face:
            total = add_v(total, (V[f][0], V[f][1]))
        total = scale(total, 1/len(face))
        V_.append(total)

    for u,v in E:
        for f in range(len(faces)):
            for g in range(len(faces)):
                if f == g:
                    continue

                if (u in faces[f] and v in faces[f]) and (u in faces[g] and v in faces[g]):
                    if not (f,g) in E_ and not (g,f) in E_:
                        E_.append((f,g))

    return V_, E_

# def print_tutte(G):
#     V, E = triangulate_graph(G)
#     (V, E), outer = tutte_from_graph((V, E))
#     V, E = add_color((V, G[1]))
#     print_graph_scaled((V, E))

# def print_tutte_dual(G):
#     V, E = triangulate_graph(G)
#     (V, E), outer = tutte_from_graph((V, E))
#     V, E = dual((V,G[1]), outer)

#     G = (range(len(V)), E)

#     V, E = triangulate_graph(G)
#     (V, E), outer = tutte_from_graph((V, E))
#     V, E = add_color((V, G[1]))

#     # Va, Ea = dual((V,E))
#     # Va, Ea = add_color((Va, Ea), c = lambda : (255,255,255))

#     # print_graph_scaled((V + Va, E + [(u+len(V),v+len(V)) for u, v in Ea]))
#     print_graph_scaled((V, E))

def spawn_in_area(G):
    V, E = triangulate_graph(G)
    print (len(V))
    (V, E), outer_a = tutte_from_graph((V, E))
    print (len(V))
    V, E = dual((V,G[1]), outer_a)
    print (len(V))

    G = (range(len(V)), E)

    V, E = triangulate_graph(G)
    print (len(V))
    (V, E), outer_b = tutte_from_graph((V, E))
    print (len(V))
    Vb, Eb = add_color((V, G[1]))

    Va, Ea = dual((Vb,Eb), outer_b)
    print (len(Va))
    Va, Ea = add_color((Va, Ea), c = lambda : tuple(map(lambda x: int(x*255), colorsys.hsv_to_rgb(random.random(), 1.0, 1.0))))

    Vs = scale_graph(Vb + Va)
    print_graph(Vs[:len(Vb)], Eb)
    print_graph(Vs[len(Vb):], []) # Ea
    print ("VA issue:", len(Va))
    print (len(Vb))
    sMap = []
    for xi in range(width):
        sMap.append([])
        for yi in range(height):
            sMap[xi].append(-1)

    queue = []
    colors = []
    for i, (x, y, c) in enumerate(Vs[len(Vb):]):
        queue.append((x,y, i))
        colors.append(c)

    iters = 0
    while len(queue) > 0: # and iters < 1000000
        iters += 1
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

        # add_elem(elem_x+1,elem_y-1, elem_i)
        # add_elem(elem_x-1,elem_y-1, elem_i)
        # add_elem(elem_x+1,elem_y+1, elem_i)
        # add_elem(elem_x-1,elem_y+1, elem_i)

    # print(colors)
    # print (sMap )

    for yi in range(height):
        for xi in range(width):
            if sMap[xi][yi] != -1:
                # print (sMap[xi][yi])
                # print (colors)
                resMap[xi][yi] = colors[sMap[xi][yi]]

    print_graph(Vs[:len(Vb)], Eb)
    print_graph(Vs[len(Vb):], []) # Ea

spawn_in_area(G)
# print_tutte_dual(G)

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('random.png')
