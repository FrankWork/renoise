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

def find_entity_index_from_tree(edge_list, e1_tokens, e2_tokens):
  # edge_list: a list of (tag, w1, idx1, w2, idx2)
  # restore words from tree and find entity index
  tree_toks = ['<META>' for _ in range(len(edge_list)*3)] 
  max_idx = 0
  for dep in edge_list:
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
  return e1_idx, e2_idx, tree_toks

def get_shortest_path(dep_text, entity_pair):
    if dep_text[0] == 'None\n':
        return None

    edge_list = []
    edges_map = {}
    nodes_set = set()
    for dep_str in dep_text:
        m=regex.search(dep_str)
        if not m:
            print(dep_text)
            raise Exception('can not parse %s' % dep_str)
        tag, w1, idx1, w2, idx2 = m.groups()
        idx1 = int(idx1)
        idx2 = int(idx2)
        edge_list.append( (tag, w1, idx1, w2, idx2) )
        node1 = '{0}-{1}'.format(w1, idx1)
        node2 = '{0}-{1}'.format(w2, idx2)
        nodes_set.add(node1)
        nodes_set.add(node2)
        edges_map[(node1, node2)] = tag
    
    e1_toks = entity_pair[0].split()
    e2_toks = entity_pair[1].split()
    e1_idx, e2_idx, _ = find_entity_index_from_tree(edge_list, e1_toks, e2_toks)
    e1_nodes = ['{0}-{1}'.format(w, i+e1_idx) for i, w in enumerate(e1_toks)]
    e2_nodes = ['{0}-{1}'.format(w, i+e2_idx) for i, w in enumerate(e2_toks)]

    # # WARNING: some tokens of the entity are ignored by the lex parser
    e1_nodes = [node for node in e1_nodes if node in nodes_set]
    e2_nodes = [node for node in e2_nodes if node in nodes_set]

    assert len(e1_nodes) != 0 and len(e2_nodes) != 0

    # print('-'*80)
    # print(entity_pair)
    graph = nx.Graph(list(edges_map.keys()))
    min_n = 1000
    s_path = None
    for s in e1_nodes:
        for t in e2_nodes:
            path = nx.shortest_path(graph, source=s, target=t)
            n = len(path)
            if min_n > n:
                min_n = n
                s_path = path
    # return s_path
    # print(' '.join(s_path))
    # print('-'*80)
    assert len(s_path) != 0
    n = len(path)
    ret_path = [path[0].split('-')[0] ]
    for i in range(n-1):
        node1 = path[i]
        node2 = path[i+1]
        tag = None
        if (node1, node2) in edges_map:
            tag = edges_map[(node1, node2)]
        else:
            tag = edges_map[(node2, node1)]
            tag = tag + '-'
        ret_path.append(tag)
        ret_path.append(node2.split('-')[0])
    return ret_path[1:-1] # ignore first and last element

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
                path = get_shortest_path(dep_text, entity_pairs[n_tree])
                fw.write(' '.join(path) + '\n')
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
    # get_sdp_for_all("data/train.stp", "data/train.entity_pair", "preprocess/train.tree")
