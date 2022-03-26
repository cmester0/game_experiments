import random
import os
import colorsys
import math
from PIL import Image

def random_walk (x, y, length, result, steps_size = 1):
    if length == 0:
        return result

    rem_length = length - (1 if not (x,y) in result else 0)

    result.append((x,y))

    direction = random.randint(0, 4)
    if direction == 0:
        return random_walk(x + steps_size, y, rem_length, result, steps_size)
    elif direction == 1:
        return random_walk(x, y + steps_size, rem_length, result, steps_size)
    elif direction == 2:
        return random_walk(x - steps_size, y, rem_length, result, steps_size)
    else: # direction == 3:
        return random_walk(x, y - steps_size, rem_length, result, steps_size)

width = 2 ** 12 - 1
height = 2 ** 12 - 1

resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append(0)

class bin_tree:
    def __init__(self, depth = 0):
        self.round_up = True
        self.values = []
        self.subtrees = dict()
        self.value_size = 10 # 2
        self.depth = depth

    def size (self):
        return len(self.values) + ((self.subtrees["above"].size() +
                                    self.subtrees["below"].size())
                                   if ("above" in self.subtrees and
                                       "below" in self.subtrees) else 0)

    # def sections (self, x0, x1, y0, y1):
    #     if ("above" in self.subtrees and
    #         "below" in self.subtrees):
    #         l = []

    #         for d, (xd, yd) in self.dirs:
    #             xa, xb = (x0, chunk_x(self.x)) if xd == 1 else (chunk_x(self.x), x1)
    #             ya, yb = (y0, chunk_y(self.y)) if yd == 1 else (chunk_y(self.y), y1)

    #             for v in self.subtrees[d].sections(xa, xb, ya, yb):
    #                 l.append(v)
    #         return l
    #     else:
    #         return [(self.depth, (x0, x1, y0, y1))]

    def value_center (self):
        angle = random.uniform(0,2 * math.pi)
        vx = math.sin(angle)
        vy = math.cos(angle)

        x_mean = sum([x for (x,y) in self.values]) / len(self.values)
        y_mean = sum([y for (x,y) in self.values]) / len(self.values)
        
        scalars = [((x - x_mean) * vx + (y - y_mean) * vy) / (vx * vx + vy * vy)  for (x,y) in self.values]
        scalars = sorted(scalars)
        
        points = [(x_mean + vx * s, y_mean + vy * s) for s in scalars]
        
        px0, py0 = points[len(scalars) // 2]
        px1, py1 = points[len(scalars) // 2 + 1]

        a = 10 if px0 == px1 else (py0 - py1) / (px0 - px1)
        b = y_mean - x_mean * a

        px = px0 # (px0 + px1) / 2
        py = py0 # (py0 + py1) / 2

        return (a, b, px, py)

    def insert (self, x, y):
        if len(self.subtrees) == 0:
            if len(self.values) < self.value_size:
                self.values.append((x, y))
                return True
            else:
                self.values.append((x, y))
                (la, lb, px, py) = self.value_center()
                self.a = la
                self.b = lb

                self.px = px
                self.py = py

                self.subtrees["above"] = bin_tree(self.depth + 1)
                self.subtrees["below"] = bin_tree(self.depth + 1)

                for (vx, vy) in self.values:
                    direction = "below"

                    if (self.a * vx + self.b > vy):
                        direction = "above"

                    if not self.subtrees[direction].insert(vx, vy):
                        return False

                self.values = []
                return True
                # return self.insert(x, y)

        direction = "below"

        if (self.a * x + self.b > y):
            direction = "above"

        if not self.subtrees[direction].insert(x, y):
            return False

        return True

qt = bin_tree ()

visited = set()

chunk_width = 4 ** 3 - 1
chunk_height = 4 ** 3 - 1

step_diff = chunk_width // 4 - 1 # chunk_height // 4

def chunk_x(x):
    return int(chunk_width * x + width // 2)
def chunk_x_float(x):
    return chunk_width * x + width // 2

def chunk_y(y):
    return int(chunk_height * y + height // 2)
def chunk_y_float(y):
    return chunk_height * y + height // 2

number_of_points = 100
# coordinate_list = [
#     (math.sqrt(number_of_points) * math.cos(x * 2 * math.pi / number_of_points),
#      math.sqrt(number_of_points) * math.sin(x * 2 * math.pi / number_of_points))
#     for x in range(number_of_points)]

# coordinate_list = [(random.randint(-width / chunk_width // 2, width / chunk_width // 2), random.randint(-height / chunk_height // 2, height / chunk_height // 2)) for _ in range(number_of_points)]

coordinate_list = []
for (x,y) in random_walk(0, 0, number_of_points, []):
    if (x,y) in visited:
        continue
    visited.add((x,y))
    coordinate_list.append((x,y))

for (x,y) in coordinate_list:
    qt.insert(x, y)

    
def line(a,b,color,equation):
    step = math.sqrt(0.0001 / (1 + a**2))

    i = -width // 2
    even = True
    while (i < width // 2):
        i += step
        xi = i
        yi = a * xi + b

        even = not even
        
        color_tip = color

        for (under, (a2,b2)) in equation:
            if (a - a2 == 0):
                continue

            if (under == "below"):
                if (a2 * xi + b2 > yi):
                    color_tip = 1
                    break

            elif (under == "above"):
                if (a2 * xi + b2 < yi):
                    color_tip = 1
                    break

        if color_tip == 1:
            continue
        
        yi = chunk_y_float(yi)
        xi = chunk_x_float(xi)
        if (xi >= 2 and xi < width - 2 and
            yi >= 2 and yi < height - 2):
            resMap[int(xi)+1][int(yi)] = color_tip
            resMap[int(xi)-1][int(yi)] = color_tip
            resMap[int(xi)][int(yi)] = color_tip            
            resMap[int(xi)][int(yi)+1] = color_tip
            resMap[int(xi)][int(yi)-1] = color_tip
            
def recursive_and_points_lines(qt, color, color2, equations):
    if ("above" in qt.subtrees and
        "below" in qt.subtrees):
        line(qt.a, qt.b, color, equations)

        # for i in range(-(step_diff-2), (step_diff-2) + 1):
        #     for j in range(-(step_diff-2), (step_diff-2) + 1):
        #         resMap[chunk_x(qt.px)+i][chunk_y(qt.py)+j] = color2

        new_equations = equations.copy()
        new_equations.append(("above", (qt.a, qt.b)))
        recursive_and_points_lines(qt.subtrees["above"], 0.9 * color, 0.9 * color2, new_equations)
                             
        new_equations = equations.copy()
        new_equations.append(("below", (qt.a, qt.b)))
        recursive_and_points_lines(qt.subtrees["below"], 0.9 * color, 0.9 * color2, new_equations)

    else:
        for (x,y) in qt.values:
            for i in range(-(step_diff-1), (step_diff-1) + 1):
                for j in range(-(step_diff-1), (step_diff-1) + 1):
                    resMap[chunk_x(x)+i][chunk_y(y)+j] = color2

recursive_and_points_lines(qt, 0.9 * 1, 1, [])

# for i in range(-(step_diff-1), (step_diff-1) + 1):
#     for j in range(-(step_diff-1), (step_diff-1) + 1):
#         resMap[chunk_x(0)+i][chunk_y(0)+j] = 0

print ("Map pre remove")
flat_m = []
for yi in range(height):
    for xi in range(width):
        h = 0
        s = 0
        v = 0

        if resMap[xi][yi] == 0:
            h = 0
            s = 0
            v = 0
        else:
            h = resMap[xi][yi]
            s = 1.0
            v = 1.0

        r, g, b = colorsys.hsv_to_rgb(h,s,v)
        flat_m.append((int(255 * r),int(255 * g),int(255 * b)))

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('bin_tree.png')
