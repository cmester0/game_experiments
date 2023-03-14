import random
import os
import colorsys
import math
from PIL import Image

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

def planarfy_graph(vertices, edges):
    for i,j in edges:
        x0, y0, c0 = vertices[i]
        x1, y1, c1 = vertices[j]

        vec_size = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        scale_size = vec_size / 10

        if vec_size < 100:
            continue

        x0_ = int(x0 + (x1 - x0) / scale_size)
        x1_ = int(x1 + (x0 - x1) / scale_size)

        y0_ = int(y0 + (y1 - y0) / scale_size)
        y1_ = int(y1 + (y0 - y1) / scale_size)

        vertices[i] = (x0_,y0_,c0)
        vertices[j] = (x1_,y1_,c1)

    return vertices

def zoom_graph(vertices, edges):
    # Find scale
    x_min, x_max = width, 0
    y_min, y_max = height, 0

    for i in range(len(vertices)):
        x0, y0, c0 = vertices[i]

        if x_max < x0:
            x_max = x0
        if x0 < x_min:
            x_min = x0

        if y_max < y0:
            y_max = y0
        if y0 < y_min:
            y_min = y0

    x_scale = (width-point_width) / (x_max - x_min)
    y_scale = (height-point_height) / (y_max - y_min)

    print ("SCALE", x_scale, y_scale)

    for i in range(len(vertices)):
        x0, y0, c0 = vertices[i]

        x0_ = int((x0 - x_min) * x_scale) # int((x0 + width/2) * x_scale - width / 2)
        y0_ = int((y0 - y_min) * y_scale) # int((y0 + height/2) * y_scale - height / 2)

        print (0, x0_, width, "vs", 0, y0_, height)
        print (x_min, x0, x_max, "vs", y_min, y0, y_max)

        vertices[i] = (x0_,y0_,c0)

    return vertices

for i in range(100):
    vertices = planarfy_graph(vertices, edges)
    vertices = zoom_graph(vertices, edges)

    resMap = []
    for xi in range(width):
        resMap.append([])
        for yi in range(height):
            resMap[xi].append((0,0,0))

    print_graph(vertices, edges)

    print ("Collapsed map")
    flat_m = []
    for yi in range(height):
        for xi in range(width):
            flat_m.append(resMap[xi][yi])

    img = Image.new('RGB', (width, height)) # width, height
    img.putdata(flat_m)
    img.save('random_collapse' + "{number:02}".format(number=i) + '.png')
