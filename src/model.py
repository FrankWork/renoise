import tensorflow as tf

class CNNModel(object):
  def __init__(self, hparams, word2vec, batch_data, training):
    e1, e2, label, bag_size, seq_len, tokens, e1_dist, e2_dist = batch_data
    xavier = xavier
    self.hparams = hparams
    pos_dim = self.hparams.pos_dim
    filters = self.hparams.filters
    kernel_size = self.hparams.kernel_size
    max_len = self.hparams.max_len

    with tf.device('/cpu:0'):
      embedding = tf.Variable(word2vec, dtype=tf.float32)
      self.inputs = tf.nn.embedding_lookup(embedding, tokens)
    
    with tf.name_scope('joint'):
      wpe1 = tf.get_variable("wpe1", shape=[123, pos_dim], initializer=xavier)
      wpe2 = tf.get_variable("wpe2", shape=[123, pos_dim], initializer=xavier)
      pos_left = tf.nn.embedding_lookup(wpe1, self.wps1)
      pos_right = tf.nn.embedding_lookup(wpe2, self.wps2)
      self.pos_embed = tf.concat([pos_left, pos_right], 2)

    with tf.name_scope('conv'):
      self._input = tf.concat([self.inputs, self.pos_embed], 2)
      self.conv = tf.layers.conv1d(self._input, filters, kernel_size,
              activation=tf.nn.relu, kernel_initializer=xavier, padding='same')
      pool_max = tf.layers.max_pooling1d(self.conv, pool_size=max_len, strides=1)
      pooled = tf.layers.dropout(pool_max, training=training)
    
    with tf.name_scope("map"):
      W = tf.get_variable(
          "W",
          shape=[filters, self.num_rels],
          initializer=xavier)
      b = tf.get_variable("b", shape=[self.num_rels], initializer=xavier)
      l2_loss += tf.nn.l2_loss(W)
      l2_loss += tf.nn.l2_loss(b)

      # the implementation of Lin et al 2016 comes from https://github.com/thunlp/TensorFlow-NRE/blob/master/network.py
      sen_a = tf.get_variable("attention_A", [filters], initializer=xavier)
      sen_q = tf.get_variable("query", [filters, 1], initializer=xavier)
      sen_r = []
      sen_s = []
      sen_out = []
      sen_alpha = []
      self.bag_score = []
      self.predictions = []
      self.losses = []
      self.accuracy = []
      self.total_loss = 0.0
      # selective attention model, use the weighted sum of all related the sentence vectors as bag representation
      for i in range(batch_num):
        sen_r.append(pooled[self.bag_num[i]:self.bag_num[i+1]])
        bag_size = self.bag_num[i+1] - self.bag_num[i]
        sen_alpha.append(
          tf.reshape(
            tf.nn.softmax(
              tf.reshape(
                tf.matmul(
                  tf.multiply(sen_r[i], sen_a), sen_q), 
              [bag_size])), 
          [1, bag_size]))
        sen_s.append(
          tf.reshape(
            tf.matmul(sen_alpha[i], sen_r[i]), 
          [1, filters]))
        sen_out.append(tf.reshape(tf.nn.xw_plus_b(sen_s[i], W, b), [self.num_rels]))
        self.bag_score.append(tf.nn.softmax(sen_out[i]))

        with tf.name_scope("output"):
          self.predictions.append(tf.argmax(self.bag_score[i], 0, name="predictions"))

        with tf.name_scope("loss"):
          nscor = self.soft_label_flag[i] * self.bag_score[i] + joint_p * tf.reduce_max(self.bag_score[i])* tf.cast(self.preds[i], tf.float32)
          self.nlabel = tf.reshape(tf.one_hot(indices=[tf.argmax(nscor, 0)], depth=self.num_rels, dtype=tf.int32), [self.num_rels])
          self.ccc = self.preds[i]
          self.losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.nlabel)))

          if i == 0:
              self.total_loss = self.losses[i]
          else:
              self.total_loss += self.losses[i]

        with tf.name_scope("accuracy"):
          self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.preds[i], 0)), "float"), name="accuracy"))

        with tf.name_scope("update"):
          self.global_step = tf.Variable(0, name="global_step", trainable=False)
          optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
          self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

      # https://github.com/tyliupku/soft-label-RE.git