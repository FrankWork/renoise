import argparse
import os
import re
import networkx as nx

pattern = re.compile(r'.+?\((.+?), (.+?)\)')
regex = re.compile("([^\()]+)\((.+)-(\d+)'*, (.+)-(\d+)'*\)")

def clean(line):
    line = line.lower()
    line = re.sub("[&'-\._]", ' ', line)
    line = re.sub(' {2,}', ' ', line)
    line = line.strip()
    return line

def brute_match_meta(text, pattern, meta_char='<META>'):
  find = None
  for i in range(len(text)):
    find = True
    for j in range(len(pattern)):
      if text[i+j] != pattern[j] and text[i+j]!=meta_char:
        find = False
        break
    if find:
      return i
  return -1

def brute_match(text, pattern):
  find = None
  for i in range(len(text)):
    find = True
    for j in range(len(pattern)):
      if text[i+j] != pattern[j]:
        find = False
        break
    if find:
      return i
  return -1

def find_entity_index_from_tree(dep_edges, e1_tokens, e2_tokens):
  # dep_edges: a list of (tag, w1, idx1, w2, idx2)
  # restore words from tree and find entity index
  tree_toks = ['<META>' for _ in range(len(dep_edges)*3)] 
  max_idx = 0
  for dep in dep_edges:
    tag, w1, idx1, w2, idx2 = dep
    # if w1=='ROOT':
    #   continue
  
    tree_toks[idx1] = w1
    tree_toks[idx2] = w2

    max_idx = max(idx1, idx2, max_idx)
  
  tree_toks = tree_toks[:max_idx+1]

  e1_idx = brute_match(tree_toks, e1_tokens)
  e2_idx = brute_match(tree_toks, e2_tokens)

  # some tokens are ignored by stanford lex parser
  if e1_idx == -1:
    e1_idx = brute_match_meta(tree_toks, e1_tokens)
    # print(e1_tokens, tree_toks)
  if e2_idx == -1:
    e2_idx = brute_match_meta(tree_toks, e2_tokens)
    # print(e2_tokens, tree_toks)

  if e1_idx==-1 or e2_idx==-1:
    raise Exception('can not find entity in the tree')
  return e1_idx, e2_idx, max_idx

def get_shortest_path(dep_text, entity_pair):
    if dep_text[0] == 'None\n':
        return None

    dep_edges = []
    edges = []
    nodes_set = set()
    for dep_str in dep_text:
        m=regex.search(dep_str)
        if not m:
            print(dep_text)
            raise Exception('can not parse %s' % dep_str)
        tag, w1, idx1, w2, idx2 = m.groups()
        idx1 = int(idx1)
        idx2 = int(idx2)
        dep_edges.append( (tag, w1, idx1, w2, idx2) )
        node1 = '{0}-{1}'.format(w1, idx1)
        node2 = '{0}-{1}'.format(w2, idx2)
        nodes_set.add(node1)
        nodes_set.add(node2)
        edges.append( (node1, node2) )
    
    e1_toks = entity_pair[0].split()
    e2_toks = entity_pair[1].split()
    e1_idx, e2_idx, _ = find_entity_index_from_tree(dep_edges, e1_toks, e2_toks)
    e1_nodes = ['{0}-{1}'.format(w, i+e1_idx) for i, w in enumerate(e1_toks)]
    e2_nodes = ['{0}-{1}'.format(w, i+e2_idx) for i, w in enumerate(e2_toks)]

    # # WARNING: some tokens of the entity are ignored by the lex parser
    # for node in e1_nodes + e2_nodes:
    #     assert node in nodes_set
    e1_nodes = [node for node in e1_nodes if node in nodes_set]
    e2_nodes = [node for node in e2_nodes if node in nodes_set]

    if len(e1_nodes) == 0 or len(e2_nodes) == 0:
        print(entity_pair)
# ['houston', 'chicago']
# ['chicago', 'houston']
# ['dallas', 'chicago']
# ['chicago', 'dallas']
# ['giants stadium', 'buffalo']
# ['buffalo', 'doug jolley']


    # # print('-'*80)
    # # print(entity_pair)
    # graph = nx.Graph(edges)
    # min_n = 1000
    # s_path = None
    # for s in e1_nodes:
    #     for t in e2_nodes:
    #         path = nx.shortest_path(graph, source=s, target=t)
    #         n = len(path)
    #         if min_n > n:
    #             min_n = n
    #             s_path = path
    # # return s_path
    # # print(' '.join(s_path))
    # # print('-'*80)
    # assert len(s_path) != 0


def get_sdp_for_all(parser_file, entity_file, shortest_path_file):
    # load entity pairs
    entity_pairs = []
    with open(entity_file) as f:
        for line in f:
            line = clean(line)
            pair = line.split('\t')
            entity_pairs.append(pair)
            # print(pair[0], '\t', pair[1])

    dep_text = []
    n_tree = 0
    fw = open(shortest_path_file, 'w')
    with open(parser_file) as fr:
        for line in fr:
            if line != '\n':
                dep_text.append(line)
            else:
                get_shortest_path(dep_text, entity_pairs[n_tree])
                n_tree += 1
                # if n_tree > 20:
                #     break
                # if n_tree % 100*100 == 0:
                #     print(n_tree)
                dep_text.clear() # python3
    fw.close()
    assert n_tree == len(entity_pairs)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--fin', default="", help='', required=True)
    # parser.add_argument('--fout', default="", help='', required=True)

    # args = parser.parse_args()
    if not os.path.exists('preprocess'):
        os.makedirs('preprocess')

    get_sdp_for_all("data/test.stp", "data/test.entity_pair", "preprocess/test.tree")
    get_sdp_for_all("data/train.stp", "data/train.entity_pair", "preprocess/train.tree")

# the occasion was suitably exceptional : a reunion of the 1970s-era sam rivers trio , with dave_holland on bass and barry_altschul on drums .
# det(occasion-2, the-1)
# nsubj(exceptional-5, occasion-2)
# cop(exceptional-5, was-3)
# advmod(exceptional-5, suitably-4)
# root(ROOT-0, exceptional-5)
# det(reunion-8, a-7)
# dep(exceptional-5, reunion-8)
# case(trio-14, of-9)
# det(trio-14, the-10)
# amod(trio-14, 1970s-era-11)
# compound(trio-14, sam-12)
# compound(trio-14, rivers-13)
# nmod(reunion-8, trio-14)
# case(dave_holland-17, with-16)
# advcl(exceptional-5, dave_holland-17)
# case(bass-19, on-18)
# nmod(dave_holland-17, bass-19)
# cc(dave_holland-17, and-20)
# conj(dave_holland-17, barry_altschul-21)
# case(drums-23, on-22)
# nmod(barry_altschul-21, drums-23)

# det(occasion-2, the-1)
# nsubj(exceptional-5, occasion-2)
# cop(exceptional-5, was-3)
# advmod(exceptional-5, suitably-4)
# root(ROOT-0, exceptional-5)
# det(reunion-8, a-7)
# parataxis(exceptional-5, reunion-8)
# case(rivers-14, of-9)
# det(rivers-14, the-10)
# nummod(rivers-14, 1970s-11)
# compound(rivers-14, era-12)
# compound(rivers-14, sam-13)
# nmod:of(reunion-8, rivers-14)
# advmod(reunion-8, trio-15)
# case(dave-17, with-16)
# nmod:with(trio-15, dave-17)
# dep(dave-17, holland-18)
# case(bass-20, on-19)
# nmod:on(holland-18, bass-20)
# cc(holland-18, and-21)
# advmod(altschul-23, barry-22)
# dep(dave-17, altschul-23)
# conj:and(holland-18, altschul-23)
# case(drums-25, on-24)
# nmod:on(altschul-23, drums-25)


# awk -v FS='\t' -v OFS='\t' '{print $3, $4}' test.txt > test.entity_pair