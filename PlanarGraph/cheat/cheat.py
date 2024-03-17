import random
import os
import colorsys
import math
from PIL import Image
import heapq
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

n, m = map(int,input().split())

colors = ["red", "green", "orange", "yellow", "blue"]

G = nx.Graph()
for v in range(n):
    G.add_node(str(v))

for i in range(m):
    f,t = map(int, input().split())
    G.add_edge(str(f),str(t),weight=2)

# nx.draw_planar(G, with_labels=True, font_color='white')

layout = nx.planar_layout(G)
for v in range(n):
    print (" ".join(map(str,layout[str(v)])))

# 0   1
#  2 3
#  4 5
# 6   7  13
#      14
#      15
# 8   9  12
#  10
#  11
# n = 16

# 16 27
# 0 1
# 0 2
# 0 6
# 1 3
# 1 5
# 1 7
# 2 3
# 2 4
# 3 5
# 4 5
# 4 6
# 5 7
# 6 8
# 8 9
# 8 10
# 8 11
# 9 10
# 9 11
# 10 11
# 7 13
# 7 14
# 9 12
# 9 15
# 12 13
# 12 15
# 13 14
# 14 15

# for a,b,g_id in self.edges:
#     if g_id == -1:
#         G.add_edge(str(a),str(b),color=color,weight=2)

