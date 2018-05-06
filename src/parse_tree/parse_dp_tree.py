import os
import pickle
import argparse
import multiprocessing
import re
import time
import sys

home_dir = os.environ['HOME']
default_dir = "."#home_dir+"/renoise"#"/work/relation-extraction/renoise"

parser = argparse.ArgumentParser()
parser.add_argument('--max_sent', default=100, type=int, help='max sent per parse')
parser.add_argument('--job_idx', default=0, type=int, help='job_idx')
parser.add_argument('--job_type', default="train", help='train or test')
parser.add_argument('--project_dir', default=default_dir, help='project dir')
args = parser.parse_args()



# parser_model_path= home_dir+"/bin/stanford-parser-full-2018-02-27/"
# os.environ['STANFORD_PARSER'] = parser_model_path+'/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = parser_model_path+'/stanford-parser-3.9.1-models.jar'

# # Dependency Tree
# from nltk.parse.stanford import StanfordDependencyParser

origin_train_file = args.project_dir+"/train.txt"
origin_test_file = args.project_dir+"/test.txt"

sents_dir = args.project_dir + "/tmp_parser/sents"
lex_dir = args.project_dir + "/tmp_parser/lex"

train_sents_file = sents_dir+"/train"
test_sents_file = sents_dir+"/test"

train_lex_file = lex_dir+"/train"
test_lex_file = lex_dir+"/test"

if not os.path.exists(sents_dir):
  os.makedirs(sents_dir)
if not os.path.exists(lex_dir):
  os.makedirs(lex_dir)


def clean_str(line):
  line = re.sub('-lrb-', '(', line)
  line = re.sub('-rrb-', ')', line)
  line = re.sub("''", ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' in line
  line = re.sub(' {2,}', ' ', line)
  return line

def get_sentence_from_data(in_file, out_file):
  print("extract sentences from %s" % os.path.basename(in_file))
  sentences = []
  with open(in_file) as f:
    for line in f:
      parts = line.strip().split('\t')
      sentence = parts[5].strip('###END###').lower().strip()
      sentence = clean_str(sentence)
      sentences.append(sentence)
  
  n_shards = 0
  for i in range(0, len(sentences), args.max_sent):
    with open("%s.%d" %(out_file, n_shards), 'w') as f:
      for sent in sentences[i:i+args.max_sent]:
        f.write(sent+'\n')
    n_shards+=1
  print("write %d files" % n_shards)
  sys.stdout.flush()


def parse_with_states(sents_dir):
  if args.job_idx == 0 and args.job_type=='train':
    state_file = args.project_dir + "/tmp_parser/state.pkl"
  else:
    state_file = args.project_dir + "/tmp_parser/state.%s.pkl.%d" %(args.job_type, args.job_idx)

  if os.path.exists(state_file):
    with open(state_file, 'rb') as f:
      state = pickle.load(f)
    print(state)
  else:
    state = set()

  # parser_sh = args.project_dir+"/src/parse_tree/parser.sh"
  parser_sh = args.project_dir+"/parser.sh"  

  for filename in os.listdir(sents_dir):
    type, idx = filename.split('.')
    if type != args.job_type:
      continue
    idx = int(idx)
    if idx>= args.job_idx and idx < args.job_idx + 500:
      if filename not in state:
        in_path = os.path.join(sents_dir, filename)
        out_path = os.path.join(lex_dir, filename)
        if os.system('bash %s %s %s'%(parser_sh, in_path, out_path)) == 0:
          state.add(filename)
          with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        else:
          exit()
        
    
  

get_sentence_from_data(origin_train_file, train_sents_file)
get_sentence_from_data(origin_test_file, test_sents_file)

parse_with_states(sents_dir)

# nohup python parse_dp_tree.py --job_type train --job_idx 1000 &
# [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

# extract sentences from train.txt
# write 5701 files
# extract sentences from test.txt
# write 1725 files










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



# parse_and_save_async(train_sents_file, train_dp_tree_file)
# parse_and_save_async(test_sents_file, test_dp_tree_file)

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






