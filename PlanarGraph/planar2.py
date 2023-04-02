edges = []
edges.append((1,2))
edges.append((1,13))
edges.append((1,4))
edges.append((1,8))
edges.append((1,12))

edges.append((2,3))
edges.append((2,13))
edges.append((2,13)) # Duplicate edge ! (Is removed by bucket sort)
edges.append((13,2)) # Duplicate edge ! (Is removed by bucket sort)

edges.append((3,4))
edges.append((3,13))

edges.append((4,5))
edges.append((4,6))
edges.append((4,7))

edges.append((5,6))
edges.append((5,8))
edges.append((5,7))

edges.append((6,7))

edges.append((8,9))
edges.append((8,12))
edges.append((8,11))

edges.append((9,10))
edges.append((9,11))
edges.append((9,12))

edges.append((10,11))
edges.append((10,12))

edges = [(a-1, b-1) for a,b in edges]
vertices = list(range(13))

# Bucket sort (generic for bucket fun and val fun)
def bucket_sort(low, high, values, bucket_fun, val_fun = lambda x: x):
    buckets = [list() for _ in range(low, high)]
    for v in values:
        buckets[bucket_fun(v)].append(val_fun(v))
    res = []
    for l in buckets:
        res += l
    return res

def split_off_multiple_edges(edges):
    # Sort edges such that all multiple edges comes after each other
    edges = bucket_sort(0, len(vertices), edges, lambda x: min(x[0], x[1]))
    edges = bucket_sort(0, len(vertices), edges, lambda x: max(x[0], x[1]))

    # remove duplicate edges
    deleted = []
    tmp = []
    i = 0
    while i < len(edges):
        x = edges[i]
        C = []
        for y in edges[i+1:]:
            if not (min(x) == min(y) and
                    max(x) == max(y)):
                break

            if len(C) == 0:
                C.append(x)
            C.append(y)

        if len(C) > 0:
            e_ = (x[0], x[1])
            tmp.append(e_)
            C.append(e_)
            deleted.append(C)
        else:
            tmp.append(x)

        i += len(C) + 1
    edges = tmp

    # deleted contains new connected componenet

    return edges

edges = split_off_multiple_edges(edges)

# edges = list(reversed(edges))
# print ("EDGE SORT", edges)


Adj = {i : [] for i in range(len(vertices))}
for v, w in edges:
    Adj[v].append(w)
    Adj[w].append(v) # Must connect both ways! (Undirected!)

dfs_numbering = dict()
count = 0

dfs_tree = {v: list() for v in range(len(vertices))}
dfs_fronds = {v: list() for v in range(len(vertices))}

lowpt1 = dict()
lowpt2 = dict()

paths = []
path = []
old_path = []

visited = set()

def dfs(v):
    global count
    global path
    global old_path

    dfs_numbering[v] = count
    count += 1

    lowpt1[dfs_numbering[v]] = dfs_numbering[v]
    lowpt2[dfs_numbering[v]] = dfs_numbering[v]

    path.append(v)
    old_path = list(path)
    for w in Adj[v]:
        if (v, w) in visited or (w, v) in visited:
            continue
        visited.add((v,w))

        if w in dfs_numbering:
            dfs_fronds[w].append(v)
            path.append(w) # The back edge
            paths.append(list(path))
            path.clear()
            path.append(v)
            old_path.clear()
            continue

        dfs_tree[v].append(w)
        dfs(w)
        path = list(old_path)

        if dfs_numbering[w] < lowpt1[dfs_numbering[v]]:
            lowpt2[dfs_numbering[v]] = lowpt1[dfs_numbering[v]]
            lowpt1[dfs_numbering[v]] = dfs_numbering[w]

dfs(0)
print (paths)
# print (dfs_numbering)
# print (lowpt1)
# print (lowpt2)
# print (dfs_tree)
# print (dfs_fronds)

exit(0)

# Gc = set(edges)
# Pc = dict()

# # Palm tree is represented by PARENT, TREE_ARC, and TYPE
# parent = [0 for _ in range(len(vertices))]
# tree_arc = [0 for _ in range(len(vertices))]
# arc_type = [0 for _ in edges]

# Adj = {i : [] for i in range(len(vertices))}
# for v, w in edges:
#     Adj[v].append(w)

# def deg(v):
#     # degree of v in Gc
#     pass

# def C_union(C, l):
#     global Gc

#     C_ = C.union(l)
#     Gc = Gc.difference(C_)
#     return C_

# def new_component(l):
#     return C_union(set(), l)

# def new_virtual_edge(v, w, C):
#     e_ = (v,w)
#     C = C_union(C, set([e_])) # TODO: Update C?
#     return e_

# def make_tree_edge(e, v, w):
#     Pc[v] = w

# def first_child(v):
#     for x in Adj[v]:
#         if x in Pc and v == Pc[x]:
#             return x
#     # otherwise?
#     pass

# def high(w):
#     if len(F(w)) == 0:
#         return 0
#     else:
#         return 0 # TODO!

# # lowpt1 = dict(min(set(v) + set(w for ))
# # lowpt2 = dict()
# # ND = dict()

# degree = [0 for _ in range(len(vertices))]

# fronds = [[] for _ in range(len(vertices))]

# START = {e: False for e in edges}


# def type_2_pairs():
#     global ESTACK
    
#     while (v != 1 and (any(lambda x: x[1] == v, TSTACK) or deg(w) == 2 and first_child(w) > w)):
#         if a == v and parent[b] == a:
#             TSTACK.pop()
#         else:
#             e_ab = None
#             if deg(w) == 2 and first_child(w) > w:
#                 C = new_component(set())
#                 # TODO # remove top edges (v,w) and (w,b) from ESTACK and add to C
#                 e_ = new_virtual_edge(v,x,C)
#                 if ESTACK[-1] == (v,b):
#                     e_ab = ESTACK.pop()
#             else:
#                 h,a,b = TSTACK.pop()
#                 C = new_component(set())
#                 while any(lambda x: a <= x[0] <= h and a <= x[1] <= h):
#                     if (x,y) == (a,b):
#                         e_ab = ESTACK.pop()
#                     else:
#                         C = C_union(C, set(ESTACK.pop()))
#             if not e_ab is None:
#                 C = new_component(set([e_ab, e_]))
#                 e_ = new_virtual_edge(v,b,C)
#             ESTACK.append(e_)
#             make_tree_edge(e_, (v, b))
#             w = b

# def type_1_pair():
#     global ESTACK
    
#     if lowpt2(w) >= v and lowpt1(w) < v and (parent[v] != 1 or True): # TODO, v is adjacent to a not yet visited tree arc:
#         C = new_component(set())
#         while any(lambda x: w <= x[0] <= w + ND(w) or w <= y <= w + ND(w), ESTACK):
#             C = C_union(C, set(ESTACK.pop()))
#         e_ = new_virtual_edge(v, lowpt1(w), C)
#         if ESTACK[-1] == (v, lowpt1(w)):
#             C = new_component(set([ESTACK.pop(), e_]))
#             e_ = new_virtual_edge(v, lowpt1(w), C)
#         if lowpt1(w) != parent[v]:
#             ESTACK.append(e_)
#             make_tree_edge(e_, (lowpt1(w), v))
#         else:
#             C = new_component(set([e_, (lowpt1(w), v)]))
#             e_ = new_virtual_edge(lowpt1, e, C)
#             make_tree_edge(e_, (lowpt1(w), v))

# def path_search(v):
#     global TSTACK
#     global ESTACK

#     print ("path search", v)

#     for w in Adj[v]:
#         e = (v, w)
#         if e in tree_arc:
#             if START[e]: # e starts a path
#                 deleted = []
#                 temp = []
#                 for (h,a,b) in TSTACK:
#                     if a > lowpt1(w):
#                         deleted.append((h,a,b))
#                     else:
#                         temp.append((h,a,b))
#                 TSTACK = temp

#                 if len(deleted) == 0:
#                     TSTACK.append((w + ND(w) - 1, lowpt(w), v))
#                 else:
#                     y = max(deleted)[0]
#                     h,a,b = deleted.pop() # last triple deleted
#                     TSTACK.append((max(y, w + ND(w) - 1), lowpt(w), b))
#                 path_search(w)
#                 ESTACK.append((v,w))
#                 # check for type-2 pairs
#                 type_2_pairs()
#                 # check for a type-1 pair
#                 type_1_pair()
#                 if START[e]: # e starts a path
#                     TSTACK.clear()
#                 while (len(TSTACK) > 0
#                        and TSTACK[-1][1] != v
#                        and TSTACK[-1][2] != v
#                        and high(v) > TSTACK[-1][0]):
#                     TSTACK.pop()
#         else:
#             if START[e]: # e starts a path
#                 deleted = []
#                 temp = []
#                 for (h,a,b) in TSTACK:
#                     if a > w:
#                         deleted.append((h,a,b))
#                     else:
#                         temp.append((h,a,b))
#                 TSTACK = temp
                
#                 if len(deleted) == 0:
#                     TSTACK.append((v,w,v))
#                 else:
#                     y = max(deleted)[0]
#                     h, a, b = deleted.pop()
#                     TSTACK.append((y,w,b))
#             if w == parent[v]:
#                 C = new_component(set([e, (w,v)]))
#                 e_ = new_virtual_edge(w,v,C)
#                 make_tree_edge(e_, (w, v))
#             else:
#                 ESTACK.append((v, w)) # e = v \-> w

# def find_split_components():
#     global TSTACK
#     global ESTACK
#     ESTACK = [] # TODO: Where to init?
#     TSTACK = []
#     path_search(0) # path_search(1)
#     print (ESTACK, TSTACK)
#     C = new_component(set(ESTACK)) # e1, ..., el from ESTACK.

# def build_triconnnected_components(vertices, edges):
#     # Sort edges in linear time using bucket sort twice (sort by min endpoint, and then max endpoint) O(|V| + |E|)
#     edge_sort_temp = bucket_sort(0, len(vertices), edges, lambda x: min(x[0], x[1]))
#     edge_sort_2 = bucket_sort(0, len(vertices), edge_sort_temp, lambda x: max(x[0], x[1]))
#     print (edge_sort_2)

#     C = find_split_components()
#     print (C)
#     m = 0 if C == None else len(C)
#     C_type = ["bond" for i in range(m)]
    
#     for i in range(m):
#         if len(C[i]) != 0 and (C_type[i] == "bond" or C_type[i] == "polygon"):
#             for e in C[i]:
#                 for j in range(m): # TODO! not linear:
#                     if j != i and e in C[j] and C_type[i] == C_type[j]:
#                         C[i] = (C[i] + C[j]) - set([e])
#                         C[j] = set()
#     print (C)


# build_triconnnected_components(vertices, edges)
