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

C_counter = -1
Cs = []
C_type = []

def split_off_multiple_edges(edges):
    global C_counter
    
    # Sort edges in linear time using bucket sort twice (sort by min endpoint, and then max endpoint) O(|V| + |E|)
    # Sort edges such that all multiple edges comes after each other
    edges = bucket_sort(0, len(vertices), edges, lambda x: min(x[0], x[1]))
    edges = bucket_sort(0, len(vertices), edges, lambda x: max(x[0], x[1]))

    # remove duplicate edges
    tmp = []
    i = 0
    while i < len(edges):
        x = edges[i]
        C = set()
        for y in edges[i+1:]:
            if not (min(x) == min(y) and
                    max(x) == max(y)):
                break

            if len(C) == 0:
                C.add(x)
            C.add(y)

        if len(C) > 0:
            e_ = (x[0], x[1])
            tmp.append(e_)
            C.add(e_)

            C_counter += 1
            Cs.append(C)
            C_type.append("bond")
        else:
            tmp.append(x)

        i += len(C) + 1
    edges = tmp

    return edges

edges = split_off_multiple_edges(edges)

Adj = {i : [] for i in range(len(vertices))}
for v, w in edges:
    Adj[v].append(w)
    Adj[w].append(v) # Must connect both ways! (Undirected!)

dfs_numbering = dict()
count = 0

dfs_tree = {v: list() for v in range(len(vertices))}
dfs_fronds = {v: list() for v in range(len(vertices))}
dfs_fronds_inv = {v: list() for v in range(len(vertices))}
parent = [None for v in range(len(vertices))]

ND = [0 for v in range(len(vertices))] # Number of decendants

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
        # dfs never search same edge twice!
        if (v, w) in visited or (w, v) in visited:
            continue
        visited.add((v,w))

        if w in dfs_numbering:
            dfs_fronds[v].append(w)
            dfs_fronds_inv[w].append(v)
            path.append(w) # The back edge
            paths.append(list(path))
            path.clear()
            path.append(v)
            old_path.clear()

            # what is the lowest number reachable in subtree
            if dfs_numbering[w] < lowpt1[dfs_numbering[v]]:
                lowpt2[dfs_numbering[v]] = lowpt1[dfs_numbering[v]]
                lowpt1[dfs_numbering[v]] = dfs_numbering[w]
            continue

        dfs_tree[v].append(w)
        parent[w] = v
        dfs(w)
        ND[v] += ND[w] + 1
        path = list(old_path)

        # what is the lowest number reachable in subtree
        if lowpt1[dfs_numbering[w]] < lowpt1[dfs_numbering[v]]:
            lowpt2[dfs_numbering[v]] = lowpt1[dfs_numbering[v]]
            lowpt1[dfs_numbering[v]] = lowpt1[dfs_numbering[w]]


dfs(0)

Gc = set(edges)
Pc = dict()

# Palm tree is represented by PARENT, TREE_ARC, and TYPE
# parent = [0 for _ in range(len(vertices))]
# tree_arc = [0 for _ in range(len(vertices))]
# arc_type = [0 for _ in edges]

Adj = {i : [] for i in range(len(vertices))}
for v, w in edges:
    Adj[v].append(w)

def deg(v):
    # degree of v in Gc
    # print ("DEG",v,"is",(len(Adj[v])))
    return (len(Adj[v]))
    # pass

def C_union(C, l):
    global Gc

    Cs[C] = Cs[C].union(l)
    Gc = Gc.difference(Cs[C])
    return C

def new_component(l):
    global C_counter

    C_counter += 1
    Cs.append(set())
    C_type.append("polygon")

    C = C_union(C_counter, l)

    return C_counter

def new_virtual_edge(v, w, C):
    e_ = (v,w)
    Cs[C].add(e_)
    Gc.add(e_)
    return e_

def make_tree_edge(e, vw):
    v, w = vw
    Pc[v] = w

def first_child(v):
    print ("FIRT CHIL", Adj[v])
    return Adj[v][0]
    # for x in Adj[v]:
    #     print (Pc, v, x)
    #     if x in Pc: # and v == Pc[x]
    #         return x
    # # otherwise?
    # print ("===== TODO =====")
    # print ("first child pass?")
    # print ("===== ODOT =====")
    # pass

def high(w):
    if len(dfs_fronds_inv[w]) == 0:
        return 0
    return dfs_fronds_inv[w][0] # source vertex if furst visited edge in F(w)?

# lowpt1 = dict(min(set(v) + set(w for ))
# lowpt2 = dict()
# ND = dict()

# degree = [0 for _ in range(len(vertices))]

# fronds = [[] for _ in range(len(vertices))]


START = set()
for l in paths:
    START.add((l[0], l[1]))
    START.add((l[1], l[0])) # TODO: Should reverse be included?

def type_2_pairs(v,w):
    global ESTACK

    while (v != 0 and
           (any(map(lambda stack_val: stack_val != "EOS" and
                    stack_val[1] == v, TSTACK))
            or (deg(w) == 2 and first_child(w) > w))):
        if any(map(lambda x: x[1] == v and parent[x[2]] == x[1], TSTACK)):
            TSTACK.pop()
        else:
            e_ab = None
            if deg(w) == 2 and first_child(w) > w:
                # TODO remove top edges (v,w) and (w,b) from ESTACK and add to C
                print ("======= TODO ========")
                print (ESTACK)

                ESTACK.pop() # (v, w)
                _, b = ESTACK.pop() # (w, b)
                C = new_component(set())
                # C = new_component(set([(v, w), (w,b)]))

                print (v, w, b)
                print ("======= ODOT ========")
                e_ = new_virtual_edge(v,b,C) # TYPO (based on original): b was x, but no x is defined anywhere?
                if len(ESTACK) > 0 and ESTACK[-1] == (v,b):
                    e_ab = ESTACK.pop()
            else:
                h,a,b = TSTACK.pop()
                C = new_component(set())
                while any(map(lambda x: (a <= x[0] <= h and
                                    a <= x[1] <= h), ESTACK)):
                    if any(map(lambda x: x == (a,b), ESTACK)): # (x,y) == (a,b)
                        e_ab = ESTACK.pop()
                    else:
                        C = C_union(C, set([ESTACK.pop()]))
                e_ = new_virtual_edge(a,b,C)
            if not e_ab is None:
                C = new_component(set([e_ab, e_]))
                e_ = new_virtual_edge(v,b,C)
            ESTACK.append(e_)
            make_tree_edge(e_, (v, b))
            w = b

def type_1_pair(v, w):
    global ESTACK

    if (lowpt2[w] >= v and
        lowpt1[w] < v and
        (parent[v] != 0 or len([adjecent for adjecent in Adj[v]]) > 0)):
        # TODO: should `parent[v] != 1` ?
        # TODO, v is adjacent to a not yet visited tree arc:
        print ("======= TODO ========")
        print (v, "adjacent to a not yet visited tree arc")
        print ([adjecent for adjecent in Adj[v]])
        print ("good enough?")
        print ("======= ODOT ========")

        C = new_component(set())
        while any(map(lambda x: w <= x[0] <= w + ND[w] or w <= x[1] <= w + ND[w], ESTACK)):
            C = C_union(C, set([ESTACK.pop()]))
        e_ = new_virtual_edge(v, lowpt1[w], C)
        if len(ESTACK) > 0 and ESTACK[-1] == (v, lowpt1[w]):
            C = new_component(set([ESTACK.pop(), e_]))
            e_ = new_virtual_edge(v, lowpt1[w], C)
        if lowpt1[w] != parent[v]:
            ESTACK.append(e_)
            make_tree_edge(e_, (lowpt1[w], v))
        else:
            C = new_component(set([e_, (lowpt1[w], v)]))
            e_ = new_virtual_edge(lowpt1[w], v, C)
            make_tree_edge(e_, (lowpt1[w], v))

def path_search(v):
    global TSTACK
    global ESTACK

    for w in Adj[v]:
        e = (v, w)
        if w in dfs_tree[v]: # e in tree_arc
            if e in START: # e starts a path
                deleted = []
                temp = []
                for stack_val in TSTACK:
                    if stack_val == "EOS":
                        temp.append("EOS")
                        continue

                    (h,a,b) = stack_val
                    if a > lowpt1[w]:
                        deleted.append((h,a,b))
                    else:
                        temp.append((h,a,b))
                TSTACK = temp

                if len(deleted) == 0:
                    TSTACK.append((w + ND[w] - 1, lowpt1[w], v))
                else:
                    y, _, _ = max(deleted)
                    h,a,b = deleted[0] # .pop() # last triple deleted
                    TSTACK.append((max(y, w + ND[w] - 1), lowpt1[w], b))
                TSTACK.append("EOS")
            path_search(w)
            ESTACK.append((v,w))
            # check for type-2 pairs
            type_2_pairs(v,w)
            # check for a type-1 pair
            type_1_pair(v,w)
            if e in START: # e starts a path
                while TSTACK[-1] != "EOS":
                    TSTACK.pop()
                TSTACK.pop()
                # TSTACK.clear()
            while any(map(lambda x: x != "EOS" and (x[1] != v and x[2] != v and high(v) > x[0]), TSTACK)):
                TSTACK.pop()
        else:
            if e in START: # e starts a path
                deleted = []
                temp = []
                for p in TSTACK:
                    if p == "EOS":
                        temp.append(p)
                        continue
                    (h,a,b) = p
                    if a > w:
                        deleted.append((h,a,b))
                    else:
                        temp.append((h,a,b))
                TSTACK = temp

                if len(deleted) == 0:
                    TSTACK.append((v,w,v))
                else:
                    y = max(deleted)[0]
                    h, a, b = deleted[0] # TODO: Last triple?
                    TSTACK.append((y,w,b))
            if w == parent[v]:
                C = new_component(set([e, (w,v)]))
                e_ = new_virtual_edge(w,v,C)
                make_tree_edge(e_, (w, v))
            else:
                ESTACK.append((v, w)) # e = v \-> w

def find_split_components():
    global TSTACK
    global ESTACK
    ESTACK = [] # TODO: Where to init?
    TSTACK = ["EOS"]
    path_search(0) # path_search(1)
    C = new_component(set(ESTACK)) # e1, ..., el from ESTACK.

find_split_components()

print ("PC",Pc)

def build_triconnnected_components():
    print ("CS", Cs)
    for i in range(len(Cs)):
        if len(Cs[i]) != 0 and (C_type[i] == "bond" or C_type[i] == "polygon"):
            for e in Cs[i]:
                for j in range(len(Cs)): # TODO! not linear:
                    if j != i and (e in Cs[j] or (e[1], e[0]) in Cs[j]) and C_type[i] == C_type[j]:
                        Cs[i] = (Cs[i].union(Cs[j])).difference(set([e]))
                        Cs[j] = set()

    print ()
    for x in list(filter(lambda x: len(x) > 0, Cs)):
        print ("CS", x)
    print ()


build_triconnnected_components()
