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

V = list(range(16))
E = [(0,1),
     (0,2),
     (0,6),
     (1,3),
     (1,7),
     (2,3),
     (2,4),
     (4,5),
     (4,6),
     (5,7),
     (6,7),
     (6,8),
     (7,9),
     (7,13),
     (7,14),
     (8,9),
     (8,10),
     (8,11),
     (9,10),
     (9,11),
     (9,12),
     (9,15),
     (10,11),
     (12,13),
     (12,15),
     (13,14),
     (14,15)]

# double bucket sort of edges
def bucket_sort_first(E):
    buckets = [[] for i in range(len(E))]
    for (a,b) in E:
        buckets[a].append((a,b))
    total = []
    for x in buckets:
        total += x
    return total

def bucket_sort_second(E):
    buckets = [[] for i in range(len(E))]
    for (a,b) in E:
        buckets[b].append((a,b))
    total = []
    for x in buckets:
        total += x
    return total

# Remove duplicates
def remove_duplicates_and_create_p(E):
    total = [E[0]]
    duplicate = []
    for x, y in zip(E,E[1:]):
        if x == y:
            duplicate.append(x)
        else:
            total.append(y)
    return total, duplicate

adj = [[] for _ in range(16)]
for (a,b) in E:
    adj[a].append(b)
    # adj[b].append(a)


from collections import defaultdict

# This class represents an directed graph
# using adjacency list representation
class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices

        # default dictionary to store graph
        self.graph = defaultdict(list)

        # time is used to find discovery times
        self.Time = 0

        # Count is number of biconnected components
        self.count = 0

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    '''A recursive function that finds and prints strongly connected
    components using DFS traversal
    u --> The vertex to be visited next
    disc[] --> Stores discovery times of visited vertices
    low[] -- >> earliest visited vertex (the vertex with minimum
               discovery time) that can be reached from subtree
               rooted with current vertex
    st -- >> To store visited edges'''
    def BCCUtil(self, u, parent, low, disc, st):

        # Count of children in current node
        children = 0

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1


        # Recur for all the vertices adjacent to this vertex
        for v in self.graph[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if disc[v] == -1 :
                parent[v] = u
                children += 1
                st.append((u, v)) # store the edge in stack
                self.BCCUtil(v, parent, low, disc, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 -- per Strongly Connected Components Article
                low[u] = min(low[u], low[v])

                # If u is an articulation point, pop
                # all edges from stack till (u, v)
                if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
                    self.count += 1 # increment count
                    w = -1
                    while w != (u, v):
                        w = st.pop()
                        print(w,end=" ")
                    print()

            elif v != parent[u] and low[u] > disc[v]:
                '''Update low value of 'u' only of 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2
                -- per Strongly Connected Components Article'''

                low[u] = min(low [u], disc[v])

                st.append((u, v))


    # The function to do DFS traversal.
    # It uses recursive BCCUtil()
    def BCC(self):

        # Initialize disc and low, and parent arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        parent = [-1] * (self.V)
        st = []

        # Call the recursive helper function to
        # find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.BCCUtil(i, parent, low, disc, st)

            # If stack is not empty, pop all edges from stack
            if st:
                self.count = self.count + 1

                while st:
                    w = st.pop()
                    print(w,end=" ")
                print ()

g = Graph(16)
E = bucket_sort_second(bucket_sort_first(E))
print (E)
for a, b in E:
    g.addEdge(a,b)

g.BCC(); print ("Above are %d biconnected components in graph"%(g.count));

# SPQR_tree[0] = ("S", [0,1,2,3,4], [(4,(3,4)),(3,(2,3)),(2,(1,2)),(1,(0,1))], [(0,4)])

# SPQR_tree[1] = ("S", [5,6,7], [(5,(0,5)),(6,(5,6)),(7,(6,1))], [])
# SPQR_tree[5] = ("Q", [0, 5], [], [(0,5)])
# SPQR_tree[6] = ("Q", [5, 6], [], [(5,6)])
# SPQR_tree[7] = ("Q", [6, 1], [], [(6,1)])

# SPQR_tree[2] = ("S", [1,7,2], [(8,(1,7)), (9, (7, 2))], [])
# SPQR_tree[8] = ("Q", [1,7], [], [(1,7)])
# SPQR_tree[9] = ("Q", [7,2], [], [(7,2)])

# SPQR_tree[3] = ("P", [2, 3], [(10, (2,3)), (11, (2,3)), (12, (2,3))], [(2,3)])
# SPQR_tree[10] = ("S", [2, 8, 3], [(13, (2,8)), (14, (8,3))], [])
# SPQR_tree[13] = ("Q", [2, 8], [], [(2,8)])
# SPQR_tree[14] = ("Q", [8, 3], [], [(8,3)])

# SPQR_tree[11] = ("Q", [2, 3], [], [(2,3)])

# SPQR_tree[12] = ("S", [2, 9, 3], [(15, (2,9)), (16, (9,3))], [])
# SPQR_tree[15] = ("Q", [2, 9], [], [(2,9)])
# SPQR_tree[16] = ("Q", [9, 3], [], [(9,3)])

# SPQR_tree[4] = ("Q", [3, 4], [], [(3,4)])

# vertices = []
# edges = []

# vt, et = dict(), []
# def trev(v, polygon):
#     global vt
#     global et

#     t, vert, sub_tree, edge = SPQR_tree[v]

#     print_filled_polygon(polygon, (255,(v * 80 % 255),(v * 30 % 255)))

#     et += edge
#     if t == "S":
#         poly_x = list(map(lambda x: x[0], polygon))
#         poly_y = list(map(lambda x: x[1], polygon))

#         cx = sum(poly_x) // len(poly_x)
#         cy = sum(poly_y) // len(poly_y)
#         rx = 0.5 * (max(poly_x) - min(poly_x)) // 2
#         ry = 0.5 * (max(poly_y) - min(poly_y)) // 2

#         # print (cx, cy, rx, ry)

#         total_verts = sum(len(SPQR_tree[vi][1]) for vi in vert)
#         offset_verts = 0
#         for i, vi in enumerate(vert):
#             offset_verts += len(SPQR_tree[vi][1])
#             xpos = int((cx + rx * (math.cos(offset_verts/total_verts * 2 * math.pi))))
#             ypos = int((cy + ry * (math.sin(offset_verts/total_verts * 2 * math.pi))))
#             if vi not in vt:
#                 vt[vi] = (xpos, ypos)

#         sorted_sub_tree = sorted(sub_tree)

#         index = 0
#         for i, (vix, viy) in sub_tree:
#             a_x, a_y = vt[vix]
#             b_x, b_y = vt[viy]

#             # print ((a_x, a_y), (b_x, b_y))
#             # print_polygon([(cx, cy), (a_x, a_y), (b_x, b_y)], (255,(v * 80 % 255),(v * 30 % 255)))

#             v_res, e = trev(i, [(cx, cy), (a_x, a_y), (b_x, b_y)])
#             et += e
#             vt = vt | v_res
#             index += 1
#     elif t == "R":
#         for i, vi in enumerate(vert):
#             if vi not in vt:
#                 poly_x = list(map(lambda x: x[0], polygon))
#                 poly_y = list(map(lambda x: x[1], polygon))

#                 x_rand = random.randint(min(poly_x), max(poly_x))
#                 y_rand = random.randint(min(poly_y), max(poly_y))
#                 vt[vi] = (x_rand, y_rand)
#     elif t == "P":
#         if vert[0] not in vt:
#             vt[vert[0]] = min(polygon)
#         if vert[1] not in vt:
#             vt[vert[1]] = max(polygon)

#         poly_x = list(map(lambda x: x[0], polygon))
#         poly_y = list(map(lambda x: x[1], polygon))

#         for j, (i, (vix, viy)) in enumerate(sub_tree):
#             a_x, a_y = vt[vix]
#             b_x, b_y = vt[viy]

#             new_polygon = [(a_x, a_y),
#                            (int(min(polygon)[0] + j * (max(polygon)[0] - min(polygon)[0]) // (len(sub_tree) + 1)),
#                             int(min(polygon)[1] + j * (max(polygon)[1] - min(polygon)[1]) // (len(sub_tree) + 1))),
#                            (b_x, b_y),
#                            (int(min(polygon)[0] + (j+1) * (max(polygon)[0] - min(polygon)[0]) // (len(sub_tree) + 1)),
#                             int(min(polygon)[1] + (j+1) * (max(polygon)[1] - min(polygon)[1]) // (len(sub_tree) + 1))),]

#             v, e = trev(i, new_polygon)
#             vt = vt | v

#     return vt, et

# vertices, edges = trev(0, [(0, 0), (width - point_width, 0),(width - point_width,height - point_height),(0,height - point_height)])
# print ("Vert", vertices)
# vertices = [vertices[i] for i in range(len(vertices))]
# vertices = list(map(lambda x: (x[0], x[1], (x[0] % 255, x[1] % 255, 255)), vertices)) # Add color ?

# print_graph(vertices, edges)
# print_square(0,0,width - point_width, height - point_height, (255,255,255))

# print ("Random map")
# flat_m = []
# for yi in range(height):
#     for xi in range(width):
#         flat_m.append(resMap[xi][yi])

# img = Image.new('RGB', (width, height)) # width, height
# img.putdata(flat_m)
# img.save('random.png')
