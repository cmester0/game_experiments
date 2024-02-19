from spqr import *

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

wiki_example = SPQR()
wiki_example_R_node_top = wiki_example.make_graph()
wiki_example_S_node = wiki_example.make_graph()
wiki_example_R_node_right = wiki_example.make_graph()
wiki_example_P_node = wiki_example.make_graph()
wiki_example_R_node_down = wiki_example.make_graph()

wiki_example_R_node_top.add_edge(0, 1)
wiki_example_R_node_top.add_edge(0, 2)
wiki_example_R_node_top.add_edge(0, 6)
wiki_example_R_node_top.add_edge(1, 3)
wiki_example_R_node_top.add_edge(1, 5)
wiki_example_R_node_top.add_edge(1, 7)
wiki_example_R_node_top.add_edge(2, 3)
wiki_example_R_node_top.add_edge(2, 4)
wiki_example_R_node_top.add_edge(3, 5)
wiki_example_R_node_top.add_edge(4, 5)
wiki_example_R_node_top.add_edge(4, 6)
wiki_example_R_node_top.add_edge(5, 7)
wiki_example_R_node_top.add_virtual_edge(6, 7, wiki_example_S_node.g_id)

print ("\n".join(map(str, wiki_example_R_node_top.edges)))
print ()

wiki_example_S_node.add_virtual_edge(6, 7, wiki_example_R_node_top.g_id)
wiki_example_S_node.add_edge(6, 8)
wiki_example_S_node.add_virtual_edge(7, 9, wiki_example_R_node_right.g_id)
wiki_example_S_node.add_virtual_edge(8, 9, wiki_example_R_node_down.g_id)

print ("\n".join(map(str, wiki_example_S_node.edges)))
print ()

wiki_example_P_node.add_virtual_edge(8, 9, wiki_example_S_node.g_id)
wiki_example_P_node.add_edge(8, 9)
wiki_example_P_node.add_virtual_edge(8, 9, wiki_example_R_node_down.g_id)

print ("\n".join(map(str, wiki_example_S_node.edges)))
print ()

wiki_example_R_node_down.add_virtual_edge(8, 9, wiki_example_P_node.g_id)
wiki_example_R_node_down.add_edge(8, 10)
wiki_example_R_node_down.add_edge(8, 11)
wiki_example_R_node_down.add_edge(9, 10)
wiki_example_R_node_down.add_edge(9, 11)
wiki_example_R_node_down.add_edge(10, 11)

print ("\n".join(map(str, wiki_example_R_node_down.edges)))
print ()

wiki_example_R_node_right.add_virtual_edge(7, 9, wiki_example_S_node.g_id)
wiki_example_R_node_right.add_edge(7, 13)
wiki_example_R_node_right.add_edge(7, 14)
wiki_example_R_node_right.add_edge(9, 12)
wiki_example_R_node_right.add_edge(9, 15)
wiki_example_R_node_right.add_edge(12, 13)
wiki_example_R_node_right.add_edge(12, 15)
wiki_example_R_node_right.add_edge(13, 14)
wiki_example_R_node_right.add_edge(14, 15)

print ("\n".join(map(str, wiki_example_R_node_right.edges)))

wiki_example.draw()
