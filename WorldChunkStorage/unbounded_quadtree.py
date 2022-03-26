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

width = 2 ** 10 - 1
height = 2 ** 10 - 1

resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append(0)

class quad_tree:
    def __init__(self, depth = 0):
        self.round_up = True
        self.values = []
        self.subtrees = dict()
        self.value_size = 16
        self.dirs = set ({("nw",(-1, 1)),
                          ("ne",( 1, 1)),
                          ("sw",(-1,-1)),
                          ("se",( 1,-1))})
        self.depth = depth

    def size (self):
        return len(self.values) + ((self.subtrees["nw"].size() +
                                    self.subtrees["ne"].size() +
                                    self.subtrees["sw"].size() +
                                    self.subtrees["se"].size())
                                   if ("nw" in self.subtrees and
                                       "ne" in self.subtrees and
                                       "sw" in self.subtrees and
                                       "se" in self.subtrees) else 0)

    def sections (self, x0, x1, y0, y1):
        if ("nw" in self.subtrees and
            "ne" in self.subtrees and
            "sw" in self.subtrees and
            "se" in self.subtrees):
            
            l = []

            for d, (xd, yd) in self.dirs:
                xa, xb = (x0, chunk_x(self.x)) if xd == 1 else (chunk_x(self.x), x1)
                ya, yb = (y0, chunk_y(self.y)) if yd == 1 else (chunk_y(self.y), y1)

                for v in self.subtrees[d].sections(xa, xb, ya, yb):
                    l.append(v)
            return l
        else:
            return [(self.depth, (x0, x1, y0, y1))]

    def value_center (self):
        slx = sorted([x for (x,y) in self.values])
        sly = sorted([y for (x,y) in self.values])

        slx0 = slx[(len(self.values)-1)//2]
        sly0 = sly[(len(self.values)-1)//2]

        slx1 = slx[(len(self.values)-1)//2+1]
        sly1 = sly[(len(self.values)-1)//2+1]

        px = (slx0 + slx1) / 2
        py = (sly0 + sly1) / 2

        px = px + 0.5 if (px / 0.5) % 2 == 0 else px
        py = py + 0.5 if (py / 0.5) % 2 == 0 else py

        return (px, py)

    def insert (self, x, y):
        if len(self.subtrees) == 0:
            if len(self.values) < self.value_size:
                self.values.append((x, y))
                return True
            else:
                (vcx, vcy) = self.value_center()
                self.x = vcx
                self.y = vcy

                for d, (xd, yd) in self.dirs:
                    self.subtrees[d] = quad_tree(self.depth + 1)

                for (vx, vy) in self.values:
                    for d, (xd, yd) in self.dirs:
                        if (((xd ==  1 and vx <= self.x or
                              xd == -1 and vx  > self.x)) and
                            ((yd ==  1 and vy <= self.y or
                              yd == -1 and vy  > self.y))):
                            if self.subtrees[d].insert(vx, vy):
                                break

                            print ("F 0", d)
                            return False
                    else:
                        print ("F N")
                self.values = []
                # return self.insert(x, y)

        for d, (xd, yd) in self.dirs:
            if (((xd ==  1 and x <= self.x or
                  xd == -1 and x  > self.x)) and
                ((yd ==  1 and y <= self.y or
                  yd == -1 and y  > self.y))):
                if self.subtrees[d].insert(x, y):
                    break

                print ("F 1", d)
                return False
        else:
            print ("F None")
            return False

        return True

qt = quad_tree ()

visited = set()

chunk_width = 4 ** 2 - 1
chunk_height = 4 ** 2 - 1

print ((chunk_width + 1) // 4 - 1)

step_diff = chunk_width // 4 - 1 # chunk_height // 4

def chunk_x(x):
    return int(chunk_width * x + width // 2)

def chunk_y(y):
    return int(chunk_height * y + height // 2)

# coordinate_list = [(random.randint(-width / chunk_width // 2, width / chunk_width // 2), random.randint(-height / chunk_height // 2, height / chunk_height // 2)) for _ in range(20)]
coordinate_list = []
for (x,y) in random_walk(0, 0, 40, []):
    if (x,y) in visited:
        continue
    visited.add((x,y))
    coordinate_list.append((x,y))

for (x,y) in coordinate_list:
    qt.insert(x, y)

section_list = qt.sections(0, width-1, 0, height-1)

for iteration, (depth, (x0,x1,y0,y1)) in enumerate(section_list):
    color = (iteration * 3) / len(section_list) % 1.0
    color_a = 0.3 + 0.7 * color
    color_b = 0.2 + 0.7 * color

    x0 += 1
    x1 -= 1
    
    y0 += 1
    y1 -= 1
    
    # for i in range(x0, x1+1):
    #     for j in range(y0, y1+1):
    #         resMap[i][j] = color_b

    for i in range(x0, x1+1):
        resMap[i][y0] = color_b
        resMap[i][y1] = color_b
    for j in range(y0, y1+1):
        resMap[x0][j] = color_b
        resMap[x1][j] = color_b

            
    x0 += depth
    x1 -= depth
    
    y0 += depth
    y1 -= depth

    for i in range(x0, x1+1):
        for j in range(y0, y1+1):
            resMap[i][j] = color_a

    
    # x0 = x0 + (depth + 1)
    # x1 = x1 - (depth + 1)
    
    # y0 = y0 + (depth + 1)
    # y1 = y1 - (depth + 1)
            
    # for i in range(x0, x1+1):
    #     resMap[i][y0] = color_b
    #     resMap[i][y1] = color_b

    # for j in range(y0, y1+1):
    #     resMap[x0][j] = color_b
    #     resMap[x1][j] = color_b
        
def insert_boundary(qt, x0, x1, y0, y1, resMap):
    if ("nw" in qt.subtrees and
        "ne" in qt.subtrees and
        "sw" in qt.subtrees and
        "se" in qt.subtrees):

        for d, (xd, yd) in qt.dirs:
            xa, xb = (x0, chunk_x(qt.x)) if xd == 1 else (chunk_x(qt.x), x1)
            ya, yb = (y0, chunk_y(qt.y)) if yd == 1 else (chunk_y(qt.y), y1)

            insert_boundary(qt.subtrees[d], xa, xb, ya, yb, resMap)
    else:
        for (x, y) in qt.values:
            for i in range(- step_diff, step_diff + 1):
                for j in range(- step_diff, step_diff + 1):
                    resMap[chunk_x(x)+i][chunk_y(y)+j] = 0.1

insert_boundary(qt, 0, width-1, 0, height-1, resMap)

for i in range(-(step_diff-1), (step_diff-1) + 1):
    for j in range(-(step_diff-1), (step_diff-1) + 1):
        resMap[chunk_x(0)+i][chunk_y(0)+j] = 0

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
img.save('quad_tree.png')
