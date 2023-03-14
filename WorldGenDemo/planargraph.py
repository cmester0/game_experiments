import random
import os
import colorsys
import math
from PIL import Image

mapList = []

width = 1000
height = 1000

# https://en.wikipedia.org/wiki/Force-directed_graph_drawing
# https://en.wikipedia.org/wiki/Graph_drawing

resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append(0)

area_size = 30
area_total = 20
areas = [(random.randint(area_size, width-1-area_size), random.randint(area_size, height-1-area_size)) for _ in range(area_total)]

edges = set()
for x in range(area_total):
    for _ in range(3):
        edges.add((x, random.randint(0, area_total-1)))

for k, (ax, ay) in enumerate(areas):
    for i in range(area_size):
        for j in range(area_size):
            resMap[ax+i][ay+j] = (k+1) / area_total

for a,b in edges:
    (ax, ay) = areas[a]
    (bx, by) = areas[b]

    resolution = width * height + width + height
    for i in range(resolution):
        resMap[ax + area_size // 2 + (bx - ax) * i // resolution][ay + area_size // 2 + (by - ay) * i // resolution] = 1

flat_m = []
for yi in range(height):
    for xi in range(width):
        h = 0
        s = 0
        v = 0

        if resMap[xi][yi] != 0:
            h = resMap[xi][yi]
            s = 1.0
            v = 1.0

        r, g, b = colorsys.hsv_to_rgb(h,s,v)
        flat_m.append((int(255 * r),int(255 * g),int(255 * b)))

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('planar_pre.png')

def distance(a, b):
    return math.sqrt((a[0]  - b[0])**2 + (a[1] - b[1])**2)

# Adjecency matrix
A = [[0 for _ in range(area_total)] for _ in range(area_total)]
for a,b in edges:
    A[a][b] = 1 # distance(areas[a],areas[b])
    A[b][a] = 1 # distance(areas[b],areas[a])

D = [[0 for _ in range(area_total)] for _ in range(area_total)]
for i in range(area_total):
    D[i][i] = sum(A[i])

# Laplacian matrix
L = [[0 for _ in range(area_total)] for _ in range(area_total)]
for i in range(area_total):
    for j in range(area_total):
        if i == j:
            L[i][j] = sum(A[i]) # deg(i)
        elif (i,j) in edges:
            L[i][j] = -A[i][j] # -distance(areas[i],areas[j])
        else:
            L[i][j] = 0

n = area_total
p = 2 # Dimension
epsilon = 10**(-8) # Tolerance
u = [[] for _ in range(0,p)]
u[0] = [1 for i in range(n)]

u_hat = [[] for _ in range(0,p+1)]
for k in range(1, p):
    u_hat[k] = [random.randint(0,100000) for j in range(n)]
    u_hat_k_norm = math.sqrt(sum([u_hat[k][j] ** 2 for j in range(n)]))
    u_hat[k] = [u_hat[k][j] / u_hat_k_norm for j in range(n)]

    while len(u[k]) == 0 or sum([u_hat[k][j] * u[k][j] for j in range(n)]) < 1 - epsilon:
        u[k] = list(u_hat[k])
        for l in range(k-1):
            factor = (sum([u[k][j] * sum([D[i][j] * u[l][i] for i in range(n)]) for j in range(n)])) / (sum([u[l][j] * sum([D[i][j] * u[k][i] for i in range(n)]) for j in range(n)]))
            u[k] = [u[k][j] - factor * u[l][j] for j in range(n)]

        for i in range(n):
            u_hat[k][i] = 1/2 * (u[k][i] + sum([A[i][j] * u[k][j] for j in range(n)]))

        u_hat_k_norm = math.sqrt(sum([u_hat[k][j] ** 2 for j in range(n)]))
        u_hat[k] = [u_hat[k][j] / u_hat_k_norm for j in range(n)]
    u[k] = list(u_hat[k])

print ([(u[0][i], u[1][i], u[2][i]) for i in range(n)])

#############################################################################

for i in range(1,p):
    for j in range(0,i-1):
        u[i] = [u[i][k] - ((sum([u[i] * u[j] for z in range(n)])) * u[j][k]) for k in range(n)]

#############################################################################

width = 1000
height = 1000

for xi in range(width):
    for yi in range(height):
        resMap[xi][yi] = 0

area_size = 30
areas = [(int((areas[i][0] -  width / 2 - sum([u[1][j] for j in range(n)]) / sum(A[i]) * 10000) / 10 +  width / 2),
          int((areas[i][1] - height / 2 - sum([u[2][j] for j in range(n)]) / sum(A[i]) * 10000) / 10 + height / 2)) for i in range(n)]

print (areas)

for k, (ax, ay) in enumerate(areas):
    for i in range(area_size):
        for j in range(area_size):
            resMap[ax+i][ay+j] = (k+1) / area_total

for a,b in edges:
    print (a,b)
    (ax, ay) = areas[a]
    (bx, by) = areas[b]

    resolution = width * height + width + height
    for i in range(resolution):
        resMap[ax + area_size // 2 + (bx - ax) * i // resolution][ay + area_size // 2 + (by - ay) * i // resolution] = 1

flat_m = []
for yi in range(height):
    for xi in range(width):
        h = 0
        s = 0
        v = 0

        if resMap[xi][yi] != 0:
            h = resMap[xi][yi]
            s = 1.0
            v = 1.0

        r, g, b = colorsys.hsv_to_rgb(h,s,v)
        flat_m.append((int(255 * r),int(255 * g),int(255 * b)))

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('planar_post.png')
