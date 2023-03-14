# First networkx library is imported 
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt

import pydot
from networkx.drawing.nx_pydot import graphviz_layout

import random

# Defining a Class
class GraphVisualization:
   
    def __init__(self):
          
        # visual is a list which stores all 
        # the set of edges that constitutes a
        # graph
        self.visual = []
          
    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)
          
    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph().to_directed()
        G.add_edges_from(self.visual)
    
        pos = graphviz_layout(G, prog="dot") # circo # twopi
        # nx.draw(G, pos)

        nx.draw_networkx(G, pos)
        # plt.show()
        plt.savefig("graph.png")
  
# Driver code
G = GraphVisualization()

depth = 10

nodes = {0: 0}

random.seed()
def generate_nodes(curr_node):
    while nodes[curr_node] < depth:
        old_node = curr_node
        if 0 == random.randint(0,5):
            curr_node = random.randint(0,len(nodes)-1)
            if 0 == random.randint(0,3):
                G.addEdge(old_node, curr_node)
                continue

        curr_node = curr_node + 1
        nodes[curr_node] = nodes[old_node] + 1
        G.addEdge(old_node, curr_node)
        
generate_nodes(0)
while 0 == random.randint(0,2):
    generate_nodes(random.randint(0,len(nodes)-1))

G.visualize()
