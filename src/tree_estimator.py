import os
import time
import numpy as np
import tensorflow as tf
from tree_model import CNNModel
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

flags.DEFINE_integer("num_threads", 10, "")
flags.DEFINE_integer("batch_size", 100, "")
flags.DEFINE_integer("max_len", 220, "")
flags.DEFINE_integer("epochs", 1, "")
flags.DEFINE_integer("log_freq", 50, "")
flags.DEFINE_integer("max_children", 5, "")

FLAGS = flags.FLAGS

def get_params():
  return {
        "learning_rate": 0.001, 
        "pos_dim" : 5,
        "num_filters" : 230,
        "kernel_size" : 3,
        "max_len" : FLAGS.max_len,
        "num_rels" : 53,
        "batch_size" : FLAGS.batch_size,
        "l2_coef" : 1e-4,
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
            "children": tf.VarLenFeature(dtype=tf.int64),
            "e1_dist": tf.VarLenFeature(dtype=tf.int64),
            "e2_dist": tf.VarLenFeature(dtype=tf.int64),
            "seq_len": tf.VarLenFeature(dtype=tf.int64),
            }
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                                  serialized=example_proto,
                                  context_features=context_features,
                                  sequence_features=sequence_features)

  # e1 = context_parsed['e1']
  # e2 = context_parsed['e2']
  label = context_parsed['label']
  bag_size = context_parsed['bag_size']

  tokens = sequence_parsed['tokens']
  children = sequence_parsed['children']
  e1_dist = sequence_parsed['e1_dist']
  e2_dist = sequence_parsed['e2_dist']
  seq_len = sequence_parsed['seq_len']

  # tokens = tf.sparse_tensor_to_dense(tokens)
  # children = tf.sparse_tensor_to_dense(children)
  # e1_dist = tf.sparse_tensor_to_dense(e1_dist)
  # e2_dist = tf.sparse_tensor_to_dense(e2_dist)
  # seq_len = tf.sparse_tensor_to_dense(seq_len)

  
  
  return label, bag_size, tokens, e1_dist, e2_dist, seq_len, children

def batch_sparse_idx(n_sent, seq_len, n_channel=1):
  '''
  [ [ 0  0] [ 0  1] [ 0  2] [ 0  3] [ 0  4] [ 0  5]
    [ 1  0] [ 1  1] [ 1  2] [ 1  3] [ 1  4] [ 1  5] [ 1  6] [ 1  7] 
  ]
  '''
  idx0 = tf.constant([], dtype=tf.int64)
  idx1 = tf.constant([], dtype=tf.int64)
  i = tf.constant(0, dtype=tf.int64)
  shape_invariants=[i.get_shape(), tf.TensorShape([None]),tf.TensorShape([None])]
  def body(i, a, b):
    length = seq_len.values[i]
    a = tf.concat([a, i*tf.ones([tf.cast(length*n_channel, tf.int32)], dtype=tf.int64)], axis=0)
    b = tf.concat([b, tf.range(length*n_channel, dtype=tf.int64)], axis=0)
    return i+1, a, b
  _, idx0, idx1 = tf.while_loop(lambda i, a, b: i<n_sent, 
                                body, [i, idx0, idx1], shape_invariants)
  idx = tf.stack([idx0,idx1], axis=-1)
  return idx

def _parse_batch_sparse(*args):
  labels, bag_size, tokens, e1_dist, e2_dist, seq_len, children=args

  n_sent = tf.reduce_sum(bag_size)
  max_len = tf.reduce_max(seq_len.values)

  # reshape 2d tensor: tokens, e1_dist, e2_dist
  idx2d = batch_sparse_idx(n_sent, seq_len)
  dense_shape_2d = [n_sent, max_len]

  tokens = tf.SparseTensor(idx2d, tokens.values, dense_shape_2d)
  e1_dist = tf.SparseTensor(idx2d, e1_dist.values, dense_shape_2d)
  e2_dist = tf.SparseTensor(idx2d, e2_dist.values, dense_shape_2d)

  # map sparse tensor to 2d dense tensor
  tokens = tf.sparse_tensor_to_dense(tokens)   # [n_sent, len]
  e1_dist = tf.sparse_tensor_to_dense(e1_dist) # [n_sent, len]
  e2_dist = tf.sparse_tensor_to_dense(e2_dist) # [n_sent, len]

  # reshape 3d tensor: children
  idx3d = batch_sparse_idx(n_sent, seq_len, n_channel=FLAGS.max_children)
  dense_shape_3d = [n_sent, max_len*FLAGS.max_children]

  children = tf.SparseTensor(idx3d, children.values, dense_shape_3d)

  # map sparse tensor to 3d dense tensor
  children = tf.sparse_tensor_to_dense(children)   # [n_sent, len*n_channel]
  shape2d = tf.shape(tokens)
  children = tf.reshape(children, [shape2d[0], shape2d[1], FLAGS.max_children])

  # idx to restore bag
  bag_idx = tf.scan(lambda a, x: a+x, tf.pad(bag_size, [[1,0]]))
  bag_idx = tf.cast(bag_idx, tf.int32)

  features = bag_size, bag_idx, seq_len.values, tokens, e1_dist, e2_dist, children
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
  train_filenames = [os.path.join(FLAGS.out_dir, FLAGS.train_records)]
  return _input_fn(train_filenames, FLAGS.epochs, FLAGS.batch_size, shuffle=True)

def test_input_fn():
  test_filenames = [os.path.join(FLAGS.out_dir, FLAGS.test_records) ]
  return _input_fn(test_filenames, 1, FLAGS.batch_size, shuffle=False)



class PatTopKHook(tf.train.SessionRunHook):
  def __init__(self, prob_tensor, labels_tensor):
    self.prob_tensor = prob_tensor
    self.labels_tensor = labels_tensor
    self.all_prob=[]
    self.all_labels = []
  
  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self.prob_tensor, self.labels_tensor])
  
  def after_run(self, run_context, run_values):
    prob, label = run_values.results
    self.all_prob.append(prob)
    self.all_labels.append(label)

  def end(self, session):
    all_prob = np.concatenate(self.all_prob, axis=0)
    all_labels = np.concatenate(self.all_labels,axis=0)

    np.save('prob.npy', all_prob)
    np.save('labels.npy', all_labels)
    tf.logging.info('save results to .npy file')
    
    bag_size, num_class = all_prob.shape
    mask = np.ones([num_class])
    mask[0]=0
    mask_prob = np.reshape(all_prob*mask, [-1])
    idx_prob = mask_prob.argsort()

    one_hot_labels = np.zeros([bag_size, num_class])
    one_hot_labels[np.arange(bag_size), all_labels] = 1
    one_hot_labels = np.reshape(one_hot_labels, [-1])

    idx = idx_prob[-100:][::-1]
    p100 = np.mean(one_hot_labels[idx])
    idx = idx_prob[-200:][::-1]
    p200 = np.mean(one_hot_labels[idx])
    idx = idx_prob[-500:][::-1]
    p500 = np.mean(one_hot_labels[idx])

    tf.logging.info("p@100: %.3f p@200: %.3f p@500: %.3f" % (p100, p200, p500))
    tf.logging.info(all_prob[-1][:5])


def my_model(features, labels, mode, params):
  """DNN with three hidden layers, and dropout of 0.1 probability."""
  vocab, vocab2id = load_vocab()
  relations, relation2id = load_relation()
  word_embed = np.load(os.path.join(FLAGS.out_dir, FLAGS.word_embed_file))

  training = mode == tf.estimator.ModeKeys.TRAIN
  m = CNNModel(params, word_embed, features, labels, training)
  
  # Compute evaluation metrics.
  metrics = {'accuracy': m.accuracy, 'mask_accuracy': m.mask_accuracy}
  tf.summary.scalar('accuracy', m.accuracy[1])

  if mode == tf.estimator.ModeKeys.EVAL:
        p_hook = PatTopKHook(m.prob, labels)
        return tf.estimator.EstimatorSpec(
            mode, loss=m.total_loss, eval_metric_ops=metrics, evaluation_hooks=[p_hook])

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  logging_hook = tf.train.LoggingTensorHook({"loss" : m.total_loss, 
              "accuracy" : m.accuracy[0], 'mask_accuracy': m.mask_accuracy[0]}, 
               every_n_iter=FLAGS.log_freq)
  

  return tf.estimator.EstimatorSpec(mode, loss=m.total_loss, train_op=m.train_op, 
                training_hooks = [logging_hook])

def main(_):
  start_time = time.time()
  params = get_params()
  classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir="saved_models/model-tree/",
        params=params)
  classifier.train(input_fn=train_input_fn)

  eval_result = classifier.evaluate(input_fn=test_input_fn)
  tf.logging.info('\nTest set accuracy: {accuracy:0.3f} {mask_accuracy:0.3f}\n'.format(**eval_result))
  duration = time.time() - start_time
  tf.logging.info("duration: %.2f hours" % (duration/3600))

  # test_records = os.path.join(FLAGS.out_dir, FLAGS.test_records)
  # dataset = _input_fn(test_records, 1, 3)
  # batch_data = dataset.make_one_shot_iterator().get_next()

  # with tf.train.MonitoredTrainingSession() as sess:
  #   features, labels = batch_data

  #   for t in  sess.run(features):
  #     print(t.shape)
  #     print(t)
  #     print()
  #  while not sess.should_stop():
  #    s = sess.run(m.bag_score)
    

if __name__=='__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  log = logging.getLogger('tensorflow')
  fh = logging.FileHandler('tmp.log')
  log.addHandler(fh)
  tf.app.run()
