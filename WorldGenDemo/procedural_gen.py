from PIL import Image
import opensimplex
import uuid

width = 4000
height = 4000

noise_map = []
# (Height, Humididty, Temperature)
noise_scale = [0.05, 0.02, 0.01]
general_scale = 1 / 2
for k in range(3):
    opensimplex.seed(uuid.uuid1().int >> 64) # 1234 + k
    noise_map.append([[10 / 2 * (opensimplex.noise2(x=i * general_scale * noise_scale[k], y=j * general_scale * noise_scale[k]) + 1) for j in range(height)] for i in range(width)])

biome_map = [[0 for j in range(height)] for i in range(width)]
for j in range(height):
    for i in range(width):
        print (noise_map[0][i][j], noise_map[1][i][j], noise_map[2][i][j])

        if noise_map[0][i][j] > 6:
            biome_map[i][j] = 1 # mountain

        elif noise_map[0][i][j] < 4:
            biome_map[i][j] = 2 # ocean

        else:
            if noise_map[1][i][j] < 4:
                biome_map[i][j] = 3 # desert

            elif noise_map[1][i][j] > 6:
                biome_map[i][j] = 4 # rainforest

            else:
                if noise_map[2][i][j] < 4:
                    biome_map[i][j] = 5 # forest

                if noise_map[2][i][j] > 6:
                    biome_map[i][j] = 6 # savanna

                else:
                    biome_map[i][j] = 7 # plains

#                NONE,   mountain,    ocean,      desert,   rainforest,  forest,     svanna,     plains
biome_color = [(0,0,0), (255,0,0), (0,0,255), (255,255,0), (0,255,0),   (0,50,0), (0,100,100), (100,200,0)]

m = [[biome_color[biome_map[i][j]] for j in range(height)] for i in range(width)]

flat_m = [m[i][j] for j in range(height) for i in range(width)]

img = Image.new('RGB', (width, height)) # width, height
img.putdata(flat_m)
img.save('procedural.png')
