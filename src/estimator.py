import os
import numpy as np
import tensorflow as tf
from model import CNNModel
import logging

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
flags.DEFINE_integer("max_len", 220, "")
flags.DEFINE_integer("epochs", 1, "")

FLAGS = flags.FLAGS

def get_params():
  return {
        "learning_rate": 0.01, 
        "pos_dim" : 5,
        "num_filters" : 230,
        "kernel_size" : 3,
        "max_len" : FLAGS.max_len,
        "num_rels" : 53,
        "batch_size" : FLAGS.batch_size,
        "l2_coef" : 1e-3,
  }


def load_vocab():
  
  vocab_file = os.path.join(FLAGS.out_dir, FLAGS.vocab_file)
  vocab = []
  vocab2id = {}
  with open(vocab_file) as f:
    for id, line in enumerate(f):
      token = line.strip()
      vocab.append(token)
      vocab2id[token] = id

  tf.logging.info("load vocab, size: %d" % len(vocab))
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
  tf.logging.info("load relation, relation size %d" % len(relations))
  return relations, relation2id

def _parse_example(example_proto):
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
  
  return e1, e2, label, bag_size, tokens, e1_dist, e2_dist, seq_len

def _parse_batch_sparse(*args):
  e1, e2, labels, bag_size, tokens, e1_dist, e2_dist, seq_len=args

  n_sent = tf.reduce_sum(bag_size)
  idx0 = tf.constant([], dtype=tf.int64)
  idx1 = tf.constant([], dtype=tf.int64)
  i = tf.constant(0, dtype=tf.int64)
  shape_invariants=[i.get_shape(), tf.TensorShape([None]),tf.TensorShape([None])]
  def body(i, a, b):
    a = tf.concat([a, i*tf.ones(seq_len.values[i], dtype=tf.int64)], axis=0)
    b = tf.concat([b, tf.range(seq_len.values[i], dtype=tf.int64)], axis=0)
    return i+1, a, b
  _, idx0, idx1 = tf.while_loop(lambda i, a, b: i<n_sent, 
                                body, [i, idx0, idx1], shape_invariants)
  idx = tf.stack([idx0,idx1], axis=-1)

  max_len = tf.reduce_max(seq_len.values)

  dense_shape = [n_sent, max_len]
  tokens = tf.SparseTensor(idx, tokens.values, dense_shape)
  e1_dist = tf.SparseTensor(idx, e1_dist.values, dense_shape)
  e2_dist = tf.SparseTensor(idx, e2_dist.values, dense_shape)

  tokens = tf.sparse_tensor_to_dense(tokens)
  e1_dist = tf.sparse_tensor_to_dense(e1_dist)
  e2_dist = tf.sparse_tensor_to_dense(e2_dist)

  bag_idx = tf.scan(lambda a, x: a+x, tf.pad(bag_size, [[1,0]]))
  bag_idx = tf.cast(bag_idx, tf.int32)

  features = e1,e2, bag_size, bag_idx, seq_len.values, tokens, e1_dist, e2_dist
  return features, labels

def _input_fn(filenames, epochs, batch_size, shuffle=False):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_example)  # Parse the record into tensors.
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(epochs)  
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(_parse_batch_sparse)
  #iterator = dataset.make_initializable_iterator()
  #batch_data = iterator.get_next()
  return dataset

def train_input_fn():
  """An input function for training"""
  # Initialize `iterator` with training data.
  train_filenames = [os.path.join(FLAGS.out_dir, FLAGS.train_records)+'.%d'%i 
                     for i in range(FLAGS.num_threads)]
  return _input_fn(train_filenames, FLAGS.epochs, FLAGS.batch_size, shuffle=True)

def test_input_fn():
  test_filenames = [os.path.join(FLAGS.out_dir, FLAGS.test_records)+'.%d'%i 
                     for i in range(FLAGS.num_threads)]
  return _input_fn(test_filenames, 1, FLAGS.batch_size, shuffle=False)

def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  vocab, vocab2id = load_vocab()
  relations, relation2id = load_relation()
  word_embed = np.load(os.path.join(FLAGS.out_dir, FLAGS.word_embed_file))

  training = mode == tf.estimator.ModeKeys.TRAIN
  m = CNNModel(params, word_embed, features, labels, training)
  
  # Compute evaluation metrics.
  metrics = {'accuracy': m.accuracy}
  tf.summary.scalar('accuracy', m.accuracy[1])

  if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=m.total_loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  logging_hook = tf.train.LoggingTensorHook({"loss" : m.total_loss, 
  "accuracy" : m.accuracy[1]}, every_n_iter=10)

  return tf.estimator.EstimatorSpec(mode, loss=m.total_loss, train_op=m.train_op, 
                training_hooks = [logging_hook])

def main(_):
  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)
  
  params = get_params()
  classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params)
  classifier.train(input_fn=train_input_fn)

  eval_result = classifier.evaluate(input_fn=test_input_fn)
  tf.logging.info('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

  # with tf.train.MonitoredTrainingSession() as sess:
  #  sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
  #  while not sess.should_stop():
  #    s = sess.run(m.bag_score)
    

if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  log = logging.getLogger('tensorflow')
  fh = logging.FileHandler('tmp.log')
  log.addHandler(fh)
  tf.app.run()
