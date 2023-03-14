import random
import os
import colorsys
import math
from PIL import Image

width = 1000
height = 1000

# Paper: Drawing Graphs by Eigenvectors: Theory and Practice*

def make_random_node(dims):
    return [random.randint(0, d-1) for d in dims]

def make_random_edge(num_edges):
    return (random.randint(0,num_edges-1), random.randint(0,num_edges-1))

def Ax(A, x):
    return [dot([A[i][j] for i in range(len(A))], x) for j in range(len(x))]

def dot(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def xTAx(x, A):
    return dot(x, Ax(A, x))

node_dims = [width, height]
p = len(node_dims)

n = 6 # R^n
nodes = [make_random_node(node_dims) for _ in range(n)]

num_edges = 2 * n
E = set()
for _ in range(num_edges):
    (a,b) = make_random_edge(n)
    E.add((a,b))
    E.add((b,a))
for i in range(num_edges):
    if (i,i) in E:
        E.remove((i,i))

N = []
for i in range(n):
    N.append(set())
    for j in range(n):
        if (i,j) in E:
            N[i].add(j)

# most cases p <= 3
# p < n

x = [[nodes[i][j] for i in range(n)] for j in range(p)] # Transpose?

d = [[math.sqrt(sum([(x[k][i] - x[k][j]) ** 2 for k in range(p)])) for i in range(n)] for j in range(n)]

A = [[1 if (i,j) in E else 0 for i in range(n)] for j in range(n)]

def deg(i):
    return sum([A[i][j] for j in N[i]])

L = [[deg(i) if i == j else (-A[i][j] if (i,j) in E else 0) for i in range(n)] for j in range(n)]

D = [[deg(i) if i == j else 0 for i in range(n)] for j in range(n)]

# u1 = [alpha * 1 for _ in range(n)]

print (sum([xTAx(x[k], L) for k in range(p)]) / sum([xTAx(x[k], D) for k in range(p)]))

print (sum([xTAx(x[k], L) for k in range(p)]) / sum([xTAx(x[k], D) for k in range(p)]))

# print (sum([A[i][j] * d[i][j]**2 for (i,j) in E]) // 2)

# print (sum([sum([A[i][j] * (x[k][i] - x[k][j]) ** 2 for (i,j) in E]) for k in range(n)]))

# matrix{[1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]}

# n = area_total
# p = 2 # Dimension

# epsilon = 10**(-8) # Tolerance
# u = [[] for _ in range(0,p)]
# u[0] = [1 / math.sqrt(n) * 1 for i in range(n)]

# u_hat = [[] for _ in range(0,p+1)]
# for k in range(1, p):
#     u_hat[k] = [random.randint(0,100000) for j in range(n)]
#     u_hat_k_norm = math.sqrt(sum([u_hat[k][j] ** 2 for j in range(n)]))
#     u_hat[k] = [u_hat[k][j] / u_hat_k_norm for j in range(n)]

#     while len(u[k]) == 0 or sum([u_hat[k][j] * u[k][j] for j in range(n)]) < 1 - epsilon:
#         u[k] = list(u_hat[k])
#         for l in range(k-1):
#             factor = (sum([u[k][j] * sum([D[i][j] * u[l][i] for i in range(n)]) for j in range(n)])) / (sum([u[l][j] * sum([D[i][j] * u[k][i] for i in range(n)]) for j in range(n)]))
#             u[k] = [u[k][j] - factor * u[l][j] for j in range(n)]

#         for i in range(n):
#             u_hat[k][i] = 1/2 * (u[k][i] + sum([A[i][j] * u[k][j] for j in range(n)]))

#         u_hat_k_norm = math.sqrt(sum([u_hat[k][j] ** 2 for j in range(n)]))
#         u_hat[k] = [u_hat[k][j] / u_hat_k_norm for j in range(n)]
#     u[k] = list(u_hat[k])

# print ([(u[0][i], u[1][i]) for i in range(n)])

# print (u[0])
# print ([xTAx([u[0][k], u[1][k]], D) for k in range(n)])
