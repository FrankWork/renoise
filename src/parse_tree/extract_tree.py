import re

regex = re.compile("([^\()]+)\((.+)-(\d+)'*, (.+)-(\d+)'*\)")

def get_lex_tree_from_str(tree_str):
  nodes = []
  for line in tree_str.split('\n'):
    if line != '\n' and line != "":
      m=regex.search(line)
      if not m:
        raise Exception('can not parse %s' % line)
      tag, w1, idx1, w2, idx2 = m.groups()
      idx1 = int(idx1)-1
      idx2 = int(idx2)-1
      nodes.append( (tag, w1, idx1, w2, idx2) )
  return nodes

def restore_sentence_from_tree(nodes):
  words = ['<unk>']*300
  n_words = 0
  for dep in nodes:
    tag, w1, idx1, w2, idx2 = dep
    n_words = max(idx1, idx2, n_words)
    words[idx1]=w1
    words[idx2]=w2
  
  # tmp = []
  # for w in words[:n_words+1]:
  #   if w != 0:
  #     tmp.append(w)
  # s = " ".join(tmp)
  return words[:n_words+1]

def get_children(tokens, nodes):
  children = [[] for _ in range(len(tokens))]
  for dep in nodes:
    tag, w1, idx1, w2, idx2 = dep
    if w1=='ROOT':
      continue

    for w in w1.split('_'):
      aligned = False
      for idx, tok in enumerate(tokens):
        if tok == w and abs(idx-idx1)<10:
          children[idx].extend(w2.split('_'))
          aligned=True
      # if not aligned:
      #   # print(w1, w2)
      #   if w1 not in  ['no']:
      #     raise Exception('not aligned %s: %s' % (w1, " ".join(tokens)))

    # if not children[idx1]:
    #   children[idx1] = (w1, [])
    # words[idx1]=w1
    # words[idx2]=w2
  for i in range(len(tokens)):
    print("%s :\t %s" % (tokens[i], " ".join(children[i])))
  
  print()

def clean_str(line):
  line = line.lower()
  line = re.sub('-lrb-', ' ', line)
  line = re.sub('-rrb-', ' ', line)
  line = re.sub("''", ' ', line)
  line = re.sub("_", ' ', line)
  line = re.sub('###end###', ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' in line
  line = re.sub(r'\\\*', ' ', line) # replace '\*'
  line = re.sub(' {2,}', ' ', line)
  line = line.strip()
  return line

def clean_tree(lines):
  s = " ".join(tmp)
  s = re.sub(" _ ", "_", s)
  s = re.sub("[':&`_\.,!?=]", " ", s)
  s = re.sub("-", " ", s)
  s = re.sub(" ", " ", s)#'\xa0'
  s = re.sub("LRB|RRB", " ", s)

  s = re.sub(r'\\\*', ' ', s) # replace '\*'
  s = re.sub(' {2,}', ' ', s)
  s = s.strip()

  return s+'\n'


stp_f = open("test.stp.align")
with open("tmp_test.txt") as f:
  for i, line in enumerate(f):
    # if i > 20:
    #   break
    parts = line.strip().split('\t')
    
    e1_kb, e2_kb = parts[0], parts[1]
    e1_str, e2_str, relation = parts[2], parts[3], parts[4]
    tokens = clean_str(parts[5]).split()

    tree_str = ""
    line = stp_f.readline()
    tree_str += line
    while True:
      line = stp_f.readline()
      tree_str += line
      if line == '\n':
        break
    
    nodes = get_lex_tree_from_str(tree_str)
    get_children(tokens, nodes)



# words = restore_sentence_from_tree(nodes)
# print(" ".join(words))



# s = restore_sentence_from_tree(lex.split('\n'))
# print(s)
# print(txt)

# stp_lex = []
# tmp = []
# with open("test.stp.align") as f:
#   for line in f:
#     tmp.append(line)
#     if line == '\n':
#       stp_lex.append("".join(tmp))
#       tmp.clear()
# print(len(stp_lex))