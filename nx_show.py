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

'''
          ROOT
           |
          fox
       /   |   \     \\
      a  quick brown jumps
                      |
                      dogs
                    /  |  \\
                 over the lazy
'''


dep_list = [
"det(occasion-2, the-1)",
"nsubj(exceptional-5, occasion-2)",
"cop(exceptional-5, was-3)",
"advmod(exceptional-5, suitably-4)",
"root(ROOT-0, exceptional-5)",
"det(reunion-8, a-7)",
"dep(exceptional-5, reunion-8)",
"case(trio-14, of-9)",
"det(trio-14, the-10)",
"amod(trio-14, 1970s-era-11)",
"compound(trio-14, sam-12)",
"compound(trio-14, rivers-13)",
"nmod(reunion-8, trio-14)",
"case(dave_holland-17, with-16)",
"advcl(exceptional-5, dave_holland-17)",
"case(bass-19, on-18)",
"nmod(dave_holland-17, bass-19)",
"cc(dave_holland-17, and-20)",
"conj(dave_holland-17, barry_altschul-21)",
"case(drums-23, on-22)",
"nmod(barry_altschul-21, drums-23)",
]

pattern = re.compile(r'.+?\((.+?), (.+?)\)')
edges = []
for p in dep_list:
  m = pattern.search(p)
  a, b = m.groups()
  # print(a, b)
  edges.append( (a, b))
  
graph = nx.Graph(edges)
path = nx.shortest_path(graph, source='dave_holland-17', target='barry_altschul-21')
print(type(path), len(path))
print('{0}'.format(path))
n = nx.shortest_path_length(graph, source='dave_holland-17', target='barry_altschul-21')
print(n)

# plt.subplot(121)
nx.draw(graph, with_labels=True, font_weight='bold')
# plt.subplot(122)
plt.show()
