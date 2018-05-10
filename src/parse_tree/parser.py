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


def clean_str(line):
  line = re.sub('-lrb-', '(', line)
  line = re.sub('-rrb-', ')', line)
  line = re.sub("''", ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' in line
  line = re.sub(' {2,}', ' ', line)
  return line
  # re.sub(r'\\\*', ' ', 'i will \*') # replace '\*'
 
def get_sentence_from_data(in_file, out_dir):
  print("extract  sentences from %s"   % in_file)
  basename, _ =  in_file.split('.'  ) 

  len60, len100,  len200, len3000 = set(), set(), set(), set()
  with oplen200_ file) as f:
    for llen3000n f:
      parts = line.strip().split('\t')
      sentence = parts[5].strip('###END###').lower().strip()
      sentence = clean_str(sentence)
      tokens = sentence.split()
      n = len(tokens)
      if n <= 60:
        len60.add(sentence)
      elif n>60 and n<=100:
        len100.add(sentence)
      elif n>100 and n<=200:
        len200.add(sentence)
      elif n>200 and n<=220:
        len3000.add(sentence)
  
  print(len(len60), len(len100), len(len200), len(len3000))

  def write_shrads(sentences, dir_name):
    sentences = list(sentences)
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)

    n_shards = 0
    for i in range(0, len(sentences), args.max_sent):
      filename = os.path.join(dir_name, "%s.%d" %(basename, n_shards))
      with open(filename, 'w') as f:
        for sent in sentences[i:i+args.max_sent]:
          f.write(sent+'\n')
      n_shards+=1
    print("write %d files" % n_shards)
  
  write_shrads(len60, out_dir+"/len60")
  write_shrads(len100, out_dir+"/len100")
  write_shrads(len200, out_dir+"/len200")
  write_shrads(len3000, out_dir+"/len3000")

  sys.stdout.flush()

get_sentence_from_data("train.txt", "tmp_dir_train")
get_sentence_from_data("test.txt", "tmp_dir_test")

# extract sentences from train.txt
# len: 60 100 200 3000
# 334338 31632 2027 16 sentences
# len60   3344 files 
#           range(0,500) range(2000,3344) (386.06 wds/sec; 10.93 sents/sec). 12g 20th
#           range(500,2000)               (287.55 wds/sec; 8.17 sents/sec).  24g 20th
# len100  317 files 35g 20th    (149.06 wds/sec; 2.08 sents/sec)
# len200  21 files  50g 10th    (25.93 wds/sec; 0.21 sents/sec)
# len3000 1 files   50g 5th     (3.72 wds/sec; 0.02 sents/sec)     

arr = ["$data_dir/train.%d" %i for i in range(0,21)]

# extract sentences from test.txt
# 55840 5487 365 15
# len60   559 files range(0, 559) 6g 4th
# len100  55 files                6g 4th  (80.93 wds/sec; 1.13 sents/sec)
# len200  4 files                 35g 10th; 50g 10th (21.25 wds/sec; 0.17 sents/sec)
# len3000 1 files  max len 225    50g 5th (5.06 wds/sec; 0.02 sents/sec)
arr = ["$data_dir/test.%d" %i for i in range(0,4)]
# " ".join(arr)







def parse_with_states(sents_dir):
  default_state_file = args.project_dir + "/tmp_parser/state.pkl"
  with open(default_state_file, 'rb') as f:
    state0 = pickle.load(f)
  
  state_file = args.project_dir + "/tmp_parser/state.%s.pkl.%d" %(args.job_type, args.job_idx)
  if os.path.exists(state_file):
    with open(state_file, 'rb') as f:
      state = pickle.load(f)
  else:
    state = set()
  
  state.update(state0)
  print(state)

  # parser_sh = args.project_dir+"/src/parse_tree/parser.sh"
  parser_path = os.environ['PWD']+"/"

  parser_cmd = """java -mx8g \
  -cp stanford-parser-full-2018-02-27/stanford-parser.jar:stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar \
  edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -nthreads 8 \
   -maxLength 200\
  -sentences newline \
  -outputFormat "typedDependencies" \
  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
  %s > %s
  """

  for filename in os.listdir(sents_dir):
    type, idx = filename.split('.')
    if type != args.job_type:
      continue
    idx = int(idx)
    if idx>= args.job_idx and idx < args.job_idx + 500:
      if filename not in state:
        in_path = os.path.join(sents_dir, filename)
        out_path = os.path.join(lex_dir, filename)
        if os.system(parser_cmd%(in_path, out_path)) == 0:
          state.add(filename)
          with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        else:
          exit()
  

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






