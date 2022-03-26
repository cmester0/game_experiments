import random
import os
import colorsys
import math
from PIL import Image

def random_walk (x, y, steps_size, length, result):
    if length == 0:
        return result

    result.append((x,y))

    direction = random.randint(0, 4)
    if direction == 0:
        return random_walk(x + steps_size, y, steps_size, length - 1, result)
    elif direction == 1:
        return random_walk(x, y + steps_size, steps_size, length - 1, result)
    elif direction == 2:
        return random_walk(x - steps_size, y, steps_size, length - 1, result)
    else: # direction == 3:
        return random_walk(x, y - steps_size, steps_size, length - 1, result)

width = 400 # int(input("Initial width: "))
height = 400 # int(input("Initial height: "))

resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append(0)

class quad_tree:
    def __init__(self, x , y , step_size):
        self.x = x
        self.y = y
        self.step_size = step_size
        self.round_up = True
        self.values = []
        self.subtrees = dict()
        self.value_size = 4

    def size (self):
        return len(self.values) + ((self.subtrees["nw"].size() +
                                    self.subtrees["ne"].size() +
                                    self.subtrees["sw"].size() +
                                    self.subtrees["se"].size())
                                   if ("nw" in self.subtrees and
                                       "ne" in self.subtrees and
                                       "sw" in self.subtrees and
                                       "se" in self.subtrees) else 0)

    def all_values (self):
        return self.values + ((self.subtrees["nw"].all_values() +
                               self.subtrees["ne"].all_values() +
                               self.subtrees["sw"].all_values() +
                               self.subtrees["se"].all_values())
                                   if ("nw" in self.subtrees and
                                       "ne" in self.subtrees and
                                       "sw" in self.subtrees and
                                       "se" in self.subtrees) else [])

    def insert (self, x, y):
        print (x,y, self.x, self.y)
        
        if self.step_size == 0 and len(self.values) >= self.value_size:
            print ("F0", x, y)
            return False
        
        if len(self.subtrees) == 0:
            if len(self.values) < self.value_size:
                self.values.append((x, y))
                return True
            else:
                next_step_size = self.step_size - 1
                diff = 2 ** self.step_size
                self.subtrees["nw"] = quad_tree(self.x - diff, self.y + diff, next_step_size)
                self.subtrees["ne"] = quad_tree(self.x + diff, self.y + diff, next_step_size)
                self.subtrees["sw"] = quad_tree(self.x - diff, self.y - diff, next_step_size)
                self.subtrees["se"] = quad_tree(self.x + diff, self.y - diff, next_step_size)

                print ("SUB VALUES")
                for (vx, vy) in self.values:
                    if vx <= self.x and y >= self.y:
                        if not self.subtrees["nw"].insert(vx, vy):
                            print ("F1")
                            return False                            
                    elif vx >= self.x and y >= self.y:
                        if not self.subtrees["ne"].insert(vx, vy):
                            print ("F2")
                            return False
                    elif vx <= self.x and y <= self.y:
                        if not self.subtrees["sw"].insert(vx, vy):
                            print ("F3")
                            return False
                    elif vx >= self.x and y <= self.y:
                        if not self.subtrees["se"].insert(vx, vy):
                            print ("F4")
                            return False
                    else:
                        print("boundary_error")
                print ("SUB VALUES DONE")
                self.values = []
                # return self.insert(x, y)

        if x <= self.x and y >= self.y:
            if not self.subtrees["nw"].insert(x, y):
                print ("F5")
                return False
        elif x >= self.x and y >= self.y:
            if not self.subtrees["ne"].insert(x, y):
                print ("F6")
                return False
        elif x <= self.x and y <= self.y:
            if not self.subtrees["sw"].insert(x, y):
                print ("F7")
                return False
        elif x >= self.x and y <= self.y:
            if not self.subtrees["se"].insert(x, y):
                print ("F8")
                return False
        else:
            print("!! Boundary error !!")
            print ("F9")
            return False

        return True

qt = quad_tree (0.5, 0.5, 3)

visited = set()

old_x = 0
old_y = 0

chunk_width = 10
chunk_height = 10

def chunk_x(x):
    return int(chunk_width * x + width // 2)

def chunk_y(y):
    return int(chunk_height * y + height // 2)

for (x,y) in random_walk(0, 0, 1, 50, []):
    for vy in range(chunk_y(y), chunk_y(old_y)):
        resMap[chunk_x(x)+1][vy] = 2
    for vx in range(chunk_x(x), chunk_x(old_x)):
        resMap[vx][chunk_y(y)+1] = 2.2
    for vy in range(chunk_y(y), chunk_y(old_y), -1):
        resMap[chunk_x(x)-1][vy] = 2.3
    for vx in range(chunk_x(x), chunk_x(old_x), -1):
        resMap[vx][chunk_y(y)-1] = 2.4
    old_x = x
    old_y = y
    
    if (x,y) in visited:
        print ("SKIP", x , y)
        continue

    visited.add((x,y))
    
    print ("INSERT:" , x, y, qt.insert(x, y))

print ("SIZE:", qt.size())
print ("VALUES:", qt.all_values())

def insert_boundary(qt, x0, x1, y0, y1, resMap, color, iteration):
    for j in range(y0, y1):
        resMap[chunk_x(qt.x)][j] = 0.5 + color / 2
        
    for i in range(x0, x1):
        resMap[i][chunk_y(qt.y)] = 0.5 + color / 2

    for (x, y) in qt.values:
        for i in range(-chunk_width // 4, chunk_width // 4 + 1):
            for j in range(-chunk_height // 4, chunk_height // 4 + 1):
                resMap[chunk_x(x)+i][chunk_y(y)+j] = 0.1
                
    for (x, y) in qt.all_values():
        for i in range(-chunk_width // 4, chunk_width // 4 + 1):
            for j in range(-chunk_height // 4, chunk_height // 4 + 1):
                resMap[chunk_x(x)+i][chunk_y(y)+j] = 0.1

    if ("nw" in qt.subtrees and
        "ne" in qt.subtrees and
        "sw" in qt.subtrees and
        "se" in qt.subtrees):

        print (iteration)
        print("Split:", x0, chunk_x(qt.x), x1)
        print("Split:", y0, chunk_y(qt.y), y1)
        print ()
        
        insert_boundary(qt.subtrees["nw"], x0, chunk_x(qt.x), chunk_y(qt.y), y1, resMap, color + (0.25 / 2 ** iteration), iteration + 1)
        insert_boundary(qt.subtrees["ne"], chunk_x(qt.x), x1, chunk_y(qt.y), y1, resMap, color + (0.5  / 2 ** iteration), iteration + 1)
        insert_boundary(qt.subtrees["sw"], x0, chunk_x(qt.x), y0, chunk_y(qt.y), resMap, color - (0.25 / 2 ** iteration), iteration + 1)
        insert_boundary(qt.subtrees["se"], chunk_x(qt.x), x1, y0, chunk_y(qt.y), resMap, color - (0.5  / 2 ** iteration), iteration + 1)
        
insert_boundary(qt, 0, width, 0, height, resMap, 0.5, 0)

# for (x,y) in random_walk(0, 0, 400, set()):
#     for i in range(10):
#         for j in range(10):
#             resMap[width // 2 + x * 10 + i][height // 2 + y * 10 + j] = 1

for i in range(-chunk_width // 4 + 1, chunk_width // 4 - 1 + 1):
    for j in range(-chunk_height // 4 + 1, chunk_height // 4 - 1 + 1):
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
img.save('voronoi.png')
