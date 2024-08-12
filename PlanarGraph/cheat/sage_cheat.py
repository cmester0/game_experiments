import sage.all
from sage import graphs
from sage.graphs.graph import Graph
from sage.graphs.connectivity import TriconnectivitySPQR
from sage.graphs.connectivity import spqr_tree, spqr_tree_to_graph, connected_components_subgraphs

from sage.graphs.generators.smallgraphs import PetersenGraph

# G = PetersenGraph()
G = Graph([(1, 2), (1, 4), (1, 8), (1, 12), (3, 4), (2, 3), (2, 13), (3, 13), (4, 5), (4, 7), (5, 6), (5, 8), (5, 7), (6, 7), (8, 11), (8, 9), (8, 12), (9, 10), (9, 11), (9, 12), (10, 12)])
G.plot(layout="spring").save("pre_example.png")

T = spqr_tree(G, algorithm="Hopcroft_Tarjan", solver=None, verbose=0, integrality_tolerance=0.001)
H = spqr_tree_to_graph(T)
# emb = H.get_embedding()

# H.layout(layout='planar', save_pos=True, test=True)
H.plot(layout="planar", set_embedding=True).save("spqr_example.png")

# L = connected_components_subgraphs(G)
# print (L)


# graphs_list.show_graphs(L)

# print (spqr_tree(G, algorithm="Hopcroft_Tarjan", solver=None, verbose=0, integrlity_tolerance=0.001))
