import os
import numpy as np
import tensorflow as tf
import re
import multiprocessing
from functools import partial

flags = tf.app.flags

flags.DEFINE_string("data_dir", "data", "data directory")
flags.DEFINE_string("relation_file", "RE/relation2id.txt", "")

flags.DEFINE_string("out_dir", "preprocess", "")
flags.DEFINE_string("word_embed_file", "word_embed.npy", "")
flags.DEFINE_string("vocab_file", "vocab.txt", "")
flags.DEFINE_string("kb_entity_embed_file", "kb_entity_embed.npy", "")
flags.DEFINE_string("train_records", "train.records","")
flags.DEFINE_string("test_records", "test.records","")

flags.DEFINE_integer("max_bag_size", 100, "")
flags.DEFINE_integer("num_threads", 10, "")
flags.DEFINE_integer("batch_size", 100, "")
FLAGS = flags.FLAGS

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

def parse_example(example_proto):
  context_features = {
            'e1': tf.FixedLenFeature([], tf.int64),
            'e2': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'bag_size': tf.FixedLenFeature([], tf.int64),}
  sequence_features = {
            # "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            # "e1_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            # "e2_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            # "seq_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "tokens": tf.VarLenFeature(dtype=tf.int64),
            "e1_dist": tf.VarLenFeature(dtype=tf.int64),
            "e2_dist": tf.VarLenFeature(dtype=tf.int64),
            "seq_len": tf.VarLenFeature(dtype=tf.int64),
            }
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                                  serialized=example_proto,
                                  context_features=context_features,
                                  sequence_features=sequence_features)

  e1 = context_parsed['e1']
  e2 = context_parsed['e2']
  label = context_parsed['label']
  bag_size = context_parsed['bag_size']

  # tokens = tf.sparse_tensor_to_dense(sequence_parsed['tokens'])
  # e1_dist = tf.sparse_tensor_to_dense(sequence_parsed['e1_dist'])
  # e2_dist = tf.sparse_tensor_to_dense(sequence_parsed['e2_dist'])
  tokens = sequence_parsed['tokens']
  e1_dist = sequence_parsed['e1_dist']
  e2_dist = sequence_parsed['e2_dist']
  seq_len = sequence_parsed['seq_len']
  
  return seq_len


def parse_batch_sparse(seq_len):
  return seq_len.values

def length_statistics(length):
    '''get maximum, mean, quantile info for length'''
    # length = sorted(length)
    # length = np.asarray(length)

    # p7 = np.percentile(length, 70)
    # Probability{length < p7} = 0.7
    percent = [50, 70, 80, 90, 95, 98, 99, 100]
    quantile = [np.percentile(length, p) for p in percent]
    
    print('(percent, quantile) %s' % str(list(zip(percent, quantile))))

def show_long_line():
  with open('/home/lzh/work/relation-extraction/renoise/data/RE/train.txt') as f:
    for line in f:
      parts = line.strip().split('\t')
      sentence = parts[5].strip('###END###').split()
      n = len(sentence)
      if n> 200:
        print(line)

def main(_):
  with open('/home/lzh/work/relation-extraction/renoise/data/RE/test.txt') as f:
    for line in f:
      line = re.sub('-lrb-', ' ', line)
      line = re.sub('-rrb-', ' ', line)
      line = re.sub("''", ' ', line)
      line = re.sub('\/', ' ', line)
      line = re.sub('-rrb-', ' ', line)
      line = re.sub(' {2,}', ' ', line)

      parts = line.strip().split('\t')
      sentence = parts[5].strip('###END###').split()
      n = len(sentence)
      if n> 200:
        print(n)
  exit()

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)
  
  
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.map(parse_example)  # Parse the record into tensors.
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(1)  
  dataset = dataset.batch(FLAGS.batch_size)
  dataset = dataset.map(parse_batch_sparse)
  iterator = dataset.make_initializable_iterator()
  seq_len = iterator.get_next()

  # Initialize `iterator` with training data.
  train_filenames = [os.path.join(FLAGS.out_dir, FLAGS.train_records)+'.%d'%i 
                     for i in range(FLAGS.num_threads)]

  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
    # x = sess.run(batch_data)
    # print(x)
    lengths = np.array([],dtype=np.int64)
    while not sess.should_stop():
      length = sess.run(seq_len)
      lengths = np.concatenate([lengths, length])
      # print(lengths.shape)
  print(lengths.shape)
  length_statistics(lengths)
  # (570088,)
  # (percent, quantile) [(50, 38.0), (70, 46.0), (80, 52.0), (90, 61.0), (95, 70.0), (98, 83.0), (100, 9619.0)]
  lengths.sort()
  print(lengths[-100:])
#   [ 279  279  285  285  285  285  285  285  285  285  285  285  285  285
#   285  285  285  285  285  285  285  285  285  285  285  285  285  285
#   285  334  334  335  335  335  335  335  336  336  336  336  336  336
#   337  337  337  350  350  350  350  350  351  351  351  351  351  351
#   351  352  352  352  352  352  372  372  372  372  372  372  372  372
#   372  372  372  382  383  383  383  383  383  384  384  386  386  386
#   400  400  400  412  412  412  440  440  440  440  440  440  794  794
#  9619 9619]


      
if __name__=='__main__':
  tf.app.run()
