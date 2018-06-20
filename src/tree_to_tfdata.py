import os
import numpy as np
import tensorflow as tf
import re
import multiprocessing
import pickle

flags = tf.app.flags

flags.DEFINE_string("data_dir", "data", "data directory")
flags.DEFINE_string("pre_train_word_embed_file", "vec.txt", "")
flags.DEFINE_string("relation_file", "RE/relation2id.txt", "")
flags.DEFINE_string("kb_entities_file", "RE/entity2id.txt", "")
flags.DEFINE_string("pre_train_kb_entity_embed_file", "pretrain/entity2vec.txt", "")
flags.DEFINE_string("txt_train_file", "RE/train.txt", "")
flags.DEFINE_string("txt_test_file", "RE/test.txt", "")
flags.DEFINE_string("tree_train_file", "train.stp", "")
flags.DEFINE_string("tree_test_file", "test.stp", "")

flags.DEFINE_string("out_dir", "preprocess", "")
flags.DEFINE_string("word_embed_file", "word_embed.npy", "")
flags.DEFINE_string("vocab_file", "vocab.txt", "")
flags.DEFINE_string("kb_entity_embed_file", "kb_entity_embed.npy", "")
flags.DEFINE_string("train_records", "train.records","")
flags.DEFINE_string("test_records", "test.records","")

flags.DEFINE_integer("max_bag_size", 100, "")
flags.DEFINE_integer("num_threads", 10, "")
# flags.DEFINE_integer("max_len", 220, "")
flags.DEFINE_integer("max_children", 5, "")
FLAGS = flags.FLAGS

feature = tf.train.Feature
sequence_example = tf.train.SequenceExample

regex = re.compile("([^\()]+)\((.+)-(\d+)'*, (.+)-(\d+)'*\)")

def features(d): return tf.train.Features(feature=d)
def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))
def bytes_feature(v): return feature(bytes_list=tf.train.BytesList(value=v))
def feature_list(l): return tf.train.FeatureList(feature=l)
def feature_lists(d): return tf.train.FeatureLists(feature_list=d)

def distance_feature(x):
	if x < -60:
		return 0
	if x >= -60 and x <= 60:
		return x+61
	if x > 60:
		return 122

def convert_word_embedding(word_size=50):
  pre_train_path = os.path.join(FLAGS.data_dir, FLAGS.pre_train_word_embed_file)
  embed_file = os.path.join(FLAGS.out_dir, FLAGS.word_embed_file)
  vocab_file = os.path.join(FLAGS.out_dir, FLAGS.vocab_file)

  if not os.path.exists(embed_file):
    word_embed = [np.zeros([word_size]), 
                  0.2*np.random.random([word_size])-0.1]# (-0.1, 0.1) uniform
    vocab = ['PAD', 'UNK']

    with open(pre_train_path) as f:
      line = f.readline()
      parts = line.strip().split()
      n_word, word_size = int(parts[0]), int(parts[1])
      print("convert pre-trained word embed, shape (%d, %d)" % (n_word, word_size))

      for line in f:
        parts = line.strip().split()
        token = parts[0]
        vec = [float(x) for x in parts[1:]]

        assert len(vec)==word_size
        word_embed.append(vec)
        vocab.append(token)

    np.save(embed_file, np.asarray(word_embed, dtype=np.float32))
  
  
    with open(vocab_file, 'w') as f:
      for token in vocab:
        f.write("%s\n" % token)


def load_vocab():
  
  vocab_file = os.path.join(FLAGS.out_dir, FLAGS.vocab_file)
  vocab = []
  vocab2id = {}
  with open(vocab_file) as f:
    for id, line in enumerate(f):
      token = line.strip()
      vocab.append(token)
      vocab2id[token] = id

  print("load vocab, size: %d" % len(vocab))
  return vocab, vocab2id

def load_relation():
  path = os.path.join(FLAGS.data_dir, FLAGS.relation_file)
  relations = []
  relation2id = {}
  with open(path) as f:
    for line in f:
      parts = line.strip().split()
      rel, id = parts[0], int(parts[1])
      relations.append(rel)
      relation2id[rel] = id
  print("load relation, relation size %d" % len(relations))
  return relations, relation2id

def convert_entities_embedding(embed_size=50):
  pre_train_path = os.path.join(FLAGS.data_dir, FLAGS.pre_train_kb_entity_embed_file)
  embed_path = os.path.join(FLAGS.out_dir, FLAGS.kb_entity_embed_file)

  if not os.path.exists(embed_path):
    ent_embed = [np.zeros([embed_size]), 
                  0.2*np.random.random([embed_size])-0.1]# (-0.1, 0.1) uniform
    
    with open(pre_train_path) as f:
      for line in f:
        parts = line.strip().split()
        vec = [float(x) for x in parts]
        ent_embed.append(vec)

    np_array = np.asarray(ent_embed, dtype=np.float32)
    print("convert pre-trained entity embed to npy, shape: ", np_array.shape)
    np.save(embed_path, np_array)

def load_kb_entities():
  path = os.path.join(FLAGS.data_dir, FLAGS.kb_entities_file)
  entities = ['PAD', 'UNK']
  entity2id = {'PAD':0, 'UNK':1}
  with open(path) as f:
    for line in f:
      parts = line.strip().split()
      ent, id = parts[0], int(parts[1])
      entities.append(ent)
      entity2id[ent] = id+2
  print("load entities, size %d" %len(entities))

  return entities, entity2id

def clean_str(line):
  line = line.lower()
  line = re.sub('-lrb-', ' ', line)
  line = re.sub('-rrb-', ' ', line)
  line = re.sub("[&'-\._]", ' ', line)
  line = re.sub('###end###', ' ', line)
  line = re.sub('\\\/', ' ', line)  # remove '\/'
  line = re.sub(r'\\\*', ' ', line) # remove '\*'
  line = re.sub(' {2,}', ' ', line)
  line = line.strip()
  return line

def get_lex_tree_from_str(tree_str):
  nodes = []
  for line in tree_str.split('\n'):
    if line=='None':
      return None

    if line != '\n' and line != "":
      m=regex.search(line)
      if not m:
        raise Exception('can not parse %s' % line)
      tag, w1, idx1, w2, idx2 = m.groups()
      idx1 = int(idx1)-1
      idx2 = int(idx2)-1
      nodes.append( (tag, w1, idx1, w2, idx2) )
  return nodes

def split_tree(nodes, e1_tokens, e2_tokens):
  # nodes: a list of (tag, w1, idx1, w2, idx2)
  tmp_toks = [0 for _ in range(len(nodes)*3)]
  max_idx = 0
  for dep in nodes:
    tag, w1, idx1, w2, idx2 = dep
    if w1=='ROOT':
      continue
    
    w1 = [x for x in w1.split('_') if x]
    w2 = [x for x in w2.split('_') if x]

    tmp_toks[idx1] = w1
    tmp_toks[idx2] = w2

    max_idx = max(idx1, idx2, max_idx)
  
  tree_toks = []
  for w in tmp_toks[:max_idx+1]:
    if isinstance(w, list):
      tree_toks.extend(w)
  
  # FIXME tmp_toks is not aligned with tree_toks
  e1_idx = find_entity_idx(tree_toks, e1_tokens, -1)
  e2_idx = find_entity_idx(tree_toks, e2_tokens, -1)

  if e1_idx==-1 or e2_idx==-1:
    # FIXME, ignore 30 instance
    # print(' '.join(e1_tokens), '||', ' '.join(e2_tokens), '||', ' '.join(tree_toks))
    return None

  # w1 - tag -> w2
  for dep in nodes:
    tag, w1, idx1, w2, idx2 = dep
    if w1=='ROOT':
      continue
  
  
    # if '_' in w1 or '_' in w2:
    #   print(w1, w2)

    # for w in w1.split('_'):
    #   aligned = False
    #   for idx, tok in enumerate(tokens):
    #     if tok == w and abs(idx-idx1)<10:
    #       children[idx].extend(w2.split('_'))
    #       aligned=True
  return None, None, None

def find_entity_idx(txt_toks, ent_toks, default_idx):
  txt = ' '.join(txt_toks)
  ent = ' '.join(ent_toks)
  idx = txt.find(ent)

  if idx == -1:
    return default_idx

  prefix = txt[:idx].split()
  word_idx = len(prefix)
  return word_idx

def write_records(bag_key, bag_value, writer):
  kb_e1_id, kb_e2_id, label = bag_key
  bag_size = len(bag_value)
  tokens_list = []
  dist1_list = []
  dist2_list = []
  children_list = []
  seq_len_list = []
  for value in bag_value:
    tokens, children, dist1, dist2, length = value
    
    tokens_list.append(int64_feature(tokens))
    dist1_list.append(int64_feature(dist1))
    dist2_list.append(int64_feature(dist2))
    seq_len_list.append(int64_feature([length]))
    children = np.reshape(children, [-1])
    children_list.append(int64_feature(children))


  example = sequence_example(
                  context=features({
                      'e1': int64_feature([0]),
                      'e2': int64_feature([0]),
                      'label': int64_feature([label]),
                      'bag_size': int64_feature([bag_size])
                  }),
                  feature_lists=feature_lists({
                      "tokens": feature_list(tokens_list),
                      "children": feature_list(children_list),
                      "e1_dist": feature_list(dist1_list),
                      "e2_dist": feature_list(dist2_list),
                      "seq_len": feature_list(seq_len_list),
                  }))
  writer.write(example.SerializeToString())

def convert_data(txt_file, tree_file, record_file, 
                  kb_entity2id, relation2id, vocab2id, shard=True):
  data = {}
  print("convert data from %s to %s"%(os.path.basename(txt_file), 
                                      os.path.basename(record_file)))
  # read data from txt file
  stp_f = open(tree_file)
  with open(txt_file) as f:
    for i, line in enumerate(f):
      # if i > 20:
      #   break
      parts = line.strip().split('\t')
      
      e1_kb, e2_kb = parts[0], parts[1]
      e1_str, e2_str, relation = parts[2], parts[3], parts[4]
      tokens = clean_str(parts[5]).split()
      label = relation2id[relation] if relation in relation2id else relation2id['NA']
      length = len(tokens)

      # distance features
      e1_tokens = clean_str(e1_str).split()
      e2_tokens = clean_str(e2_str).split() 
      e1_idx = find_entity_idx(tokens, e1_tokens, -1)
      e2_idx = find_entity_idx(tokens, e2_tokens, -1)
      if e1_idx == -1 or e2_idx == -1:
        raise Exception('can not find entity in the text')
      
      dist1 = [distance_feature(i-e1_idx) for i in range(length)]
      dist2 = [distance_feature(i-e2_idx) for i in range(length)]

      # parse lex tree to get children tokens
      tree_str = ""
      line = stp_f.readline()
      tree_str += line
      while True:
        line = stp_f.readline()
        tree_str += line
        if line == '\n':
          break
      
      nodes = get_lex_tree_from_str(tree_str)
      if nodes is None:
        continue
      # children = get_children(tokens, nodes)
      ret_val = split_tree(nodes, e1_tokens, e2_tokens)
      if not ret_val:
        continue
      sdp_tree, e1_tree, e2_tree = ret_val

      # # word to idx
      # tokens = [vocab2id[x] if x in vocab2id else vocab2id['UNK'] 
      #                for x in tokens]
      # for i in range(len(children)):
      #   children[i] = [vocab2id[x] if x in vocab2id else vocab2id['UNK'] 
      #                for x in children[i]]

      # bag_key = (e1_str, e2_str, label) #e1_str+'||'+e2_str+'||'+relation
      # bag_value = (tokens, children, dist1, dist2, length)
      # if bag_key not in data:
      #   data[bag_key]=[]
      # data[bag_key].append(bag_value)
  stp_f.close()

  # shard data and convert to tf records data
  # print("write records file ")
  # with tf.python_io.TFRecordWriter(record_file) as writer:
  #   if shard:
  #     for key in data:
  #       # slice a big bag into small bags
  #       arr = data[key]
  #       for i in range(0, len(arr), FLAGS.max_bag_size):
  #         write_records(key, arr[i:i+FLAGS.max_bag_size], writer)
  #   else:
  #     for key in data:
  #       write_records(key, data[key], writer)

def main(_):
  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

  convert_word_embedding()
  convert_entities_embedding()
  # embed_file = os.path.join(FLAGS.out_dir, FLAGS.word_embed_file)
  # embed = np.load(embed_file)
  relations, relation2id = load_relation()
  vocab,     vocab2id    = load_vocab()
  entities,  entity2id   = load_kb_entities()
  convert_data(os.path.join(FLAGS.data_dir, FLAGS.txt_train_file),
               os.path.join(FLAGS.data_dir, FLAGS.tree_train_file),
               os.path.join(FLAGS.out_dir, FLAGS.train_records),
               entity2id, relation2id, vocab2id)
  convert_data(os.path.join(FLAGS.data_dir, FLAGS.txt_test_file),
               os.path.join(FLAGS.data_dir, FLAGS.tree_test_file),
               os.path.join(FLAGS.out_dir, FLAGS.test_records),
               entity2id, relation2id, vocab2id, shard=False)

if __name__=='__main__':
  tf.app.run()
