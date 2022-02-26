import random
import os
import colorsys
import math
from PIL import Image

mapList = []

grow_max = 200

width = 2 * grow_max + 200 # int(input("Initial width: "))
height = 2 * grow_max + 200 # int(input("Initial height: "))

spawnSize = 4 # round(width * height / (8 ** 4))
emptySpaceSpawn = spawnSize * 0

def distance(xa, ya, xb, yb):
    if abs(xa - xb) < width / 2:
        xminsq = (xa  - xb)**2
    else:
        xminsq = (width - max(xa, xb) + min(xa,xb))**2

    if abs(ya - yb) < height / 2:
        yminsq = (ya  - yb)**2
    else:
        yminsq = (height - max(ya, yb) + min(ya,yb))**2
    
    return math.sqrt(xminsq + yminsq)    

resMap = []
for xi in range(width):
    resMap.append([])
    for yi in range(height):
        resMap[xi].append(0)

print ("Spawning Seeds")
for i in range(spawnSize + emptySpaceSpawn):
    resMap[random.randint(grow_max, width-grow_max-1)][random.randint(grow_max, height-grow_max-1)] = i+1

print ("### Filling Map area ###")
fill_iter = 0
filled = False
while not filled:
    print ("FILL ITERATION", fill_iter)
    fill_iter += 1

    if fill_iter > 50:
        break
    
    tempMap = resMap
    filled = True
    for xi in range(width):
        for yi in range(height):
            if tempMap[xi][yi] != 0:
                xir = xi
                yir = yi
                
                if bool(random.getrandbits(1)):
                    if bool(random.getrandbits(1)):
                        xir = (xi+1) % width
                    else:
                        xir = (xi-1) % width
                else:
                    if bool(random.getrandbits(1)):
                        yir = (yi+1) % height
                    else:
                        yir = (yi-1) % height

                if resMap[xir][yir] == 0:
                    resMap[xir][yir] = tempMap[xi][yi]
            else:
                filled = False
                
print ("Map pre remove")
flat_m = []
for yi in range(height):
    for xi in range(width):
        h = 0
        s = 0
        v = 0
        
        if resMap[xi][yi] > 0 and resMap[xi][yi] <= spawnSize:
            h = resMap[xi][yi] / spawnSize
            s = 1.0
            v = 1.0
        elif resMap[xi][yi] > spawnSize and resMap[xi][yi] <= spawnSize+emptySpaceSpawn:
            h = (resMap[xi][yi]-spawnSize) / (emptySpaceSpawn)
            s = 1.0
            v = 0.5
            
        r, g, b = colorsys.hsv_to_rgb(h,s,v)
        flat_m.append((int(255 * r),int(255 * g),int(255 * b)))

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('voronoi.png')
