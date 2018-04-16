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
            "seq_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "tokens": tf.VarLenFeature(dtype=tf.int64),
            "e1_dist": tf.VarLenFeature(dtype=tf.int64),
            "e2_dist": tf.VarLenFeature(dtype=tf.int64),
            # "seq_len": tf.VarLenFeature(dtype=tf.int64),
            }
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                                  serialized=example_proto,
                                  context_features=context_features,
                                  sequence_features=sequence_features)

  e1 = context_parsed['e1']
  e2 = context_parsed['e2']
  label = context_parsed['label']
  bag_size = context_parsed['bag_size']

  tokens = tf.sparse_tensor_to_dense(sequence_parsed['tokens'])
  e1_dist = tf.sparse_tensor_to_dense(sequence_parsed['e1_dist'])
  e2_dist = tf.sparse_tensor_to_dense(sequence_parsed['e2_dist'])
  seq_len = sequence_parsed['seq_len']
  
  return e1, e2, label, bag_size, tokens, e1_dist, e2_dist, seq_len


def main(_):
  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)
  
  vocab, vocab2id = load_vocab()
  relations, relation2id = load_relation()
  
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TFRecordDataset(filenames)

  dataset = dataset.map(parse_example)  # Parse the record into tensors.
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(1)  
  dataset = dataset.padded_batch(5, ([], [], [], [], [None,None], [None,None], [None,None], [None]))
  iterator = dataset.make_initializable_iterator()
  batch_data = iterator.get_next()

  # Initialize `iterator` with training data.
  train_filenames = [os.path.join(FLAGS.out_dir, FLAGS.train_records)+'.%d'%i 
                     for i in range(FLAGS.num_threads)]
  
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
    e1, e2, label, bag_size, tokens, e1_dist, e2_dist, seq_len = sess.run(batch_data)
    print('bag_size:', bag_size.shape, bag_size)
    print('seq_len:', seq_len.shape, seq_len)
    print('tokens:', tokens.shape, tokens)
    

if __name__=='__main__':
  tf.app.run()
