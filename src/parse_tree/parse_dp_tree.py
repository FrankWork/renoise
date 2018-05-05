import os
import pickle
import argparse
import multiprocessing
import re
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--max_sent', default=50, help='max sent per parse')
args = parser.parse_args()

home_dir = os.environ['HOME']
project_dir = home_dir+"/work/relation-extraction/renoise"

parser_model_path= home_dir+"/bin/stanford-parser-full-2018-02-27/"
os.environ['STANFORD_PARSER'] = parser_model_path+'/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = parser_model_path+'/stanford-parser-3.9.1-models.jar'

# Dependency Tree
from nltk.parse.stanford import StanfordDependencyParser

origin_train_file = project_dir+"/data/RE/train.txt"
origin_test_file = project_dir+"/data/RE/test.txt"

train_sents_file = project_dir+"/preprocess/train.sents"
test_sents_file = project_dir+"/preprocess/test.sents"

train_dp_tree_file = project_dir+"/preprocess/train.tree.pkl"
test_dp_tree_file = project_dir+"/preprocess/test.tree.pkl"

max_sent = args.max_sent # max sentences per parse
num_threads = 10

def clean_str(line):
  line = re.sub('-lrb-', '(', line)
  line = re.sub('-rrb-', ')', line)
  line = re.sub("''", ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' in line
  line = re.sub(' {2,}', ' ', line)
  return line

def get_sentence_from_data(in_file, out_file):
  print("extract sentences from %s" % os.path.basename(in_file))
  of = open(out_file, 'w')
  with open(in_file) as f:
    for line in f:
      parts = line.strip().split('\t')
      sentence = parts[5].strip('###END###').lower().strip()
      sentence = clean_str(sentence)
      of.write('%s\n' % sentence)
  of.close()

def parse_and_save_fn(args):
  sentences, out_filename = args
  sentences = [sent.strip() for sent in sentences] # strip '\n'
  chunks = [sentences[x:x+max_sent] for x in range(0, len(sentences), max_sent)]
  del sentences

  dep_parser=StanfordDependencyParser()
  f = open(out_filename, 'wb')
  for i in range(len(chunks)):
    start_time = time.time()
    parse_trees = dep_parser.raw_parse_sents(chunks[i])
    graphs = [next(tree_iter) for tree_iter in parse_trees]
    graphs = [dict(graph.nodes) for graph in graphs]
    for graph in graphs:
      pickle.dump(graph, f)

    duration = time.time() - start_time
    n_sents = len(chunks[i])
    print('parse %d sentences, %.1f sec/sent' % (n_sents, duration/n_sents))
    sys.stdout.flush()
  
  f.close()

def parse_and_save_async(in_filename, out_filename):
  print('parse %s' % os.path.basename(out_filename))
  f = open(in_filename)
  sentences = f.readlines()
  f.close()

  n = len(sentences)
  m = n // num_threads
  shards = [sentences[x:x+m] for x in range(0, n, m)]
  if len(shards) > num_threads:
    tmp_arr = shards.pop()
    shards[-1].extend(tmp_arr)
  
  del sentences

  out_filenames = [out_filename+'.%d'%i for i in range(num_threads)]
  pool = multiprocessing.Pool(num_threads)
  try:
    pool.map_async(parse_and_save_fn, zip(shards, out_filenames)).get(999999)
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    pool.terminate()


  

def load_graphs(pickle_file):
  with open(pickle_file, 'rb') as f:
    graphs = pickle.load(f)
  return graphs


# get_sentence_from_data(origin_train_file, train_sents_file)
# get_sentence_from_data(origin_test_file, test_sents_file)

# parse_and_save_async(train_sents_file, train_dp_tree_file)
parse_and_save_async(test_sents_file, test_dp_tree_file)

# graphs = load_graphs('test.head.np')

# for graph in graphs:
#   for i in sorted(graph.keys()):
#     cur_node=graph[i]
#     deps = cur_node['deps']
#     for k in deps.keys():
#       for idx in deps[k]:
#         child = graph[idx]
#         w = cur_node['word']
#         print(w if w is not None else '<ROOT>', k, child['word'], sep='\t')
    
# <ROOT>  root    jumps
# fox     det     the
# fox     appos   brown
# brown   amod    quick
# jumps   nsubj   fox
# jumps   nmod    dog
# dog     case    over
# dog     det     the
# dog     amod    lazy

# import pickle

# arr = list(range(10))
# f = open('tmp.txt', 'wb')
# for i in arr:
#   pickle.dump(i, f)
# f.close()

# f = open('tmp.txt', 'rb')
# try:
#   while True:
#     i = pickle.load(f)
#     print(i)
# except EOFError:
#   pass
# f.close()






