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
        steps = abs(x0 - x1) + abs(y0 - y1)
        x2, y2 = x0 + (x1 - x0) / steps * steps/2, y0 + (y1 - y0) / steps * steps/2
        x2_, y2_ = x0 + (x1 - x0) / steps * steps//2, y0 + (y1 - y0) / steps * steps//2
        print_line(x0,y0,x2,y2,c0)
        print_line(x2_,y2_,x1,y1,c1)

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

def add_outer_face(G, outer_degree=3):
    V, E = G

    V = list(V)
    E = list(E)

    additional_set = set()
    while len(additional_set) < outer_degree:
        additional_set.add(random.randint(0, len(V)-1))
    additional_set = list(additional_set)

    for _ in range(outer_degree):
        V.append(V[-1]+1)
    outer_face = list(V[-outer_degree:])

    for i in range(outer_degree):
        E.append((V[-1-i], additional_set[i]))

    for i in range(outer_degree-1):
        E.append((V[-1-i], V[-2-i]))
    E.append((V[-1], V[-outer_degree]))

    return (V,E), outer_face

def ForceDirected(G, epsilon, K, outer_face):
    V, E = G

    V = list(V)
    E = list(E)

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
        if t > 0 and t % 1 == 0:
            print (t, max([size(F[t-1][v]) for v in V]), epsilon)
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

    return p, E

def tutte_from_graph(G):
    V,E = G
    (V,E), outer = add_outer_face((V,E), 3) # >= 4 break guarantee?
    p, E_ = ForceDirected((V,E), 10e3, 50, outer)
    return (p,E_), set(outer)

def add_color(G, c = lambda i, n: tuple(map(lambda x: int(x*255), colorsys.hsv_to_rgb(i /n, 0.7, 1.0)))):
    V,E = G
    return [(x,y,c(i, len(V)))for i, (x,y) in enumerate(V)], E

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

def spawn_in_area(G):
    V, E = G
    V, E = add_color((V, E))

    SV = scale_graph(V)
    print_graph([(x,y,(255,255,255)) for (x,y,c) in SV], E)

    sMap = []
    for xi in range(width):
        sMap.append([])
        for yi in range(height):
            sMap[xi].append(-1)

    queue = []
    colors = []
    for i, (x, y, c) in enumerate(SV):
        queue.append((x,y, i))
        colors.append(c)

    for (a,b) in E:
        x0, y0, _ = SV[a]
        x1, y1, _ = SV[b]
        steps = abs(x0 - x1) + abs(y0 - y1)
        for i in range(0, int(steps)//2):
            queue.append((int(x0 + point_width / 2 + (x1 - x0) / steps * i),
                          int(y0 + point_height / 2 + (y1 - y0) / steps * i),
                          a))
        for i in range(int(steps)//2, int(steps)):
            queue.append((int(x0 + point_width / 2 + (x1 - x0) / steps * i),
                          int(y0 + point_height / 2 + (y1 - y0) / steps * i),
                          b))

    iters = 0
    while len(queue) > 0:
        iters += 1
        if iters > 0 and iters % 1000 == 0:
            print (iters, len(queue))
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

    for yi in range(height):
        for xi in range(width):
            if sMap[xi][yi] != -1:
                resMap[xi][yi] = colors[sMap[xi][yi]]

    print_graph([(x,y,(255,255,255)) for (x,y,c) in SV], E)

# Bowyer Watson
def delaunay_triangulation(pointList):
    largestX = max(map(lambda x: x[0], pointList))
    smallestX = min(map(lambda x: x[0], pointList))

    largestY = max(map(lambda x: x[1], pointList))
    smallestY = min(map(lambda x: x[1], pointList))

    new_point_list = [((x - smallestX) / (largestX - smallestX),
                       (y - smallestY) / (largestY - smallestY)) for x,y in pointList]

    triangulation = []
    outer_triangle = ((-0.7,0),(0.6,2),(1.5,-0.1))
    triangulation.append(outer_triangle)

    def inside_circumcircle(point, triangle):
        (Ax, Ay), (Bx, By), (Cx, Cy) = triangle

        Bx_, By_ = Bx - Ax, By - Ay
        Cx_, Cy_ = Cx - Ax, Cy - Ay

        D_ = 2 * (Bx_ * Cy_ - By_ * Cx_)

        Ux_ = 1/D_ * (Cy_ * ( Bx_ ** 2 + By_ **2 ) - By_ * (Cx_ ** 2 + Cy_ ** 2))
        Uy_ = 1/D_ * (Bx_ * ( Cx_ ** 2 + Cy_ **2 ) - Cx_ * (Bx_ ** 2 + By_ ** 2))

        r = math.sqrt(Ux_**2 + Uy_**2)

        Ux = Ux_ + Ax
        Uy = Uy_ + Ay

        return (point[0] - Ux) ** 2 + (point[1] - Uy) ** 2 <= r ** 2

    for point in new_point_list:
        badTriangles = set()
        for triangle in triangulation:
            if inside_circumcircle(point, triangle):
                badTriangles.add(triangle)
        polygon = set()
        for triangle in badTriangles:
            for edge in [(triangle[0], triangle[1]),
                         (triangle[1], triangle[2]),
                         (triangle[0], triangle[2])]:
                for other_triangle in badTriangles:
                    if (edge in [(other_triangle[0], other_triangle[1]),
                                 (other_triangle[1], other_triangle[2]),
                                 (other_triangle[0], other_triangle[2])]
                        and triangle != other_triangle):
                        break
                else:
                    polygon.add(edge)
        for triangle in badTriangles:
            if triangle in triangulation:
                triangulation.pop(triangulation.index(triangle))
        for edge in polygon:
            triangulation.append((edge[0], edge[1], point))

    result_triangulation = list(filter(lambda triangle: not (outer_triangle[0] in triangle or
                                                        outer_triangle[1] in triangle or
                                                        outer_triangle[2] in triangle), triangulation))

    new_result_triangulation = [[((x * (largestX - smallestX) + smallestX),
                                  (y * (largestY - smallestY) + smallestY))
                                 for x,y in points] for points in result_triangulation]

    outer_result_triangulation = [[((x * (largestX - smallestX) + smallestX),
                                  (y * (largestY - smallestY) + smallestY))
                                 for x,y in points] for points in triangulation]


    return new_result_triangulation, (outer_result_triangulation,
                                      [((x * (largestX - smallestX) + smallestX),
                                        (y * (largestY - smallestY) + smallestY))
                                       for x,y in outer_triangle])

plist = [(x,y) for x,y,c in random_points_list(300)]
triangulation, (outer_triangulation, outer) = delaunay_triangulation(plist)

# # # 0   1
# # #  2 3
# # #  4 5
# # # 6   7  13
# # #      14
# # #      15
# # # 8   9  12
# # #  10
# # #  11
# # # n = 16
# # G = (list(range(16)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2),(7,13),(13,12),(12,9),(9,7),(7,14),(13,14),(14,15),(15,9),(15,12),(9,8),(9,10),(9,11),(11,8),(11,10),(8,10),(8,6)])

# # 0   1
# #  2 3
# #  4 5
# # 6   7
# G = (list(range(8)),[(0,1),(1,7),(7,6),(6,0),(0,2),(1,3),(7,5),(6,4),(2,3),(3,5),(5,4),(4,2)])

# # G = (list(range(4)),[(0,1),(0,2),(1,3),(2,3)])


# # V, E = triangulate_graph(G)
# V,E = G
# print ("START", E)
# (V, E), outer = tutte_from_graph((V, E))

# # spawn_in_area((V,E))

# # print_tutte_dual(G)

vertices = []
edges = []
outer_set = set()

vertices_index = dict()
index = 0

for (x0,y0),(x1,y1),(x2,y2) in outer_triangulation:
    if not (x0,y0) in vertices_index:
        vertices_index[(x0,y0)] = index
        vertices.append((x0,y0))
        index += 1
    if not (x1,y1) in vertices_index:
        vertices_index[(x1,y1)] = index
        vertices.append((x1,y1))
        index += 1
    if not (x2,y2) in vertices_index:
        vertices_index[(x2,y2)] = index
        vertices.append((x2,y2))
        index += 1

    if (x0,y0) in outer:
        outer_set.add(vertices_index[(x0, y0)])
    if (x1,y1) in outer:
        outer_set.add(vertices_index[(x1, y1)])
    if (x2,y2) in outer:
        outer_set.add(vertices_index[(x2, y2)])

    edges.append((vertices_index[(x0, y0)],vertices_index[(x1, y1)]))
    edges.append((vertices_index[(x1, y1)],vertices_index[(x2, y2)]))
    edges.append((vertices_index[(x0, y0)],vertices_index[(x2, y2)]))

outer_list = list(sorted(outer_set))

vertices = vertices # + plist

# # PLOT GRAPH:
# V, E = vertices, edges
V, E = add_color((vertices, edges))
V.pop(outer_list[2])
V.pop(outer_list[1])
V.pop(outer_list[0])
E = [(a,b) for a, b in E if not (a in outer_list or b in outer_list)]
E = [(a if a < outer_list[2] else a-1,
      b if b < outer_list[2] else b-1)
     for a, b in E]
E = [(a if a < outer_list[1] else a-1,
      b if b < outer_list[1] else b-1)
     for a, b in E]
E = [(a if a < outer_list[0] else a-1,
      b if b < outer_list[0] else b-1)
     for a, b in E]
SV = scale_graph(V)
print_graph([(x,y,c) for (x,y,c) in SV], E)

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('finished_data.png')
resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append((0,0,0))

G = (list(range(len(vertices))), edges)
# (V, E), _ = tutte_from_graph(G)
V,E = G
# (V,E), outer = add_outer_face((V,E), 3) # >= 4 break guarantee?
p, E_ = ForceDirected((V,E), 10e2, 10e4, outer_list)
(V, E) = (p,E_)
## End of tutte
V.pop(outer_list[2])
V.pop(outer_list[1])
V.pop(outer_list[0])
E = [(a,b) for a, b in E if not (a in outer_list or b in outer_list)]
E = [(a if a < outer_list[2] else a-1,
      b if b < outer_list[2] else b-1)
     for a, b in E]
E = [(a if a < outer_list[1] else a-1,
      b if b < outer_list[1] else b-1)
     for a, b in E]
E = [(a if a < outer_list[0] else a-1,
      b if b < outer_list[0] else b-1)
     for a, b in E]

V_, E_ = add_color((V, E))
print_graph_scaled((V_, E_))

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('finished_result.png')
resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append((0,0,0))

spawn_in_area((V, E))

print ("Random map")
flat_m = []
for yi in range(height):
    for xi in range(width):
        flat_m.append(resMap[xi][yi])

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('finished_spawn.png')
