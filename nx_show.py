import networkx as nx
import re
import matplotlib.pyplot as plt

text = "a quick brown fox jumps over the lazy dog"
dep_list = [
"det(fox-4, a-1)",
"amod(fox-4, quick-2)",
"amod(fox-4, brown-3)",
"root(ROOT-0, fox-4)",
"dep(fox-4, jumps-5)",
"case(dog-9, over-6)",
"det(dog-9, the-7)",
"amod(dog-9, lazy-8)",
"nmod(jumps-5, dog-9)",
]

# dep_list = [
# 'det(three-2, the-1)',
# 'root(ROOT-0, three-2)',
# 'case(states-5, from-3)',
# 'amod(states-5, interior-4)',
# 'nmod:from(three-2, states-5)',
# 'dep(three-2, chicago-7)',
# 'dep(three-2, dallas-9)',
# 'conj:and(chicago-7, dallas-9)',
# 'cc(chicago-7, and-10)',
# 'dep(three-2, houston-11)',
# 'conj:and(chicago-7, houston-11)',
# ]

pattern = re.compile(r'.+?\((.+?), (.+?)\)')
edges = []
for p in dep_list:
  m = pattern.search(p)
  a, b = m.groups()
  # print(a, b)
  edges.append( (a, b))
  
# graph = nx.Graph(edges)
# path = nx.shortest_path(graph, source='fox-4', target='dog-9')
# # path = nx.shortest_path(graph, source='dallas-9', target='houston-11')
# print('{0}'.format(path))

# # plt.subplot(121)
# nx.draw(graph, with_labels=True, font_weight='bold')

dgraph = nx.DiGraph([(1,2), (2,3), (3,4)])
for node in dgraph.successors(1):
  print(node)

# nx.draw(dgraph, with_labels=True, font_weight='bold')
# # plt.subplot(122)
# plt.show()

