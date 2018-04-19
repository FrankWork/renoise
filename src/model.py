import tensorflow as tf

class CNNModel(object):
  def __init__(self, hparams, word2vec, batch_data, training=False):
    e1, e2, label, bag_size, bag_idx, seq_len, tokens, e1_dist, e2_dist = batch_data
    xavier = tf.contrib.layers.xavier_initializer()
    self.hparams = hparams
    pos_dim = self.hparams.pos_dim
    num_filters = self.hparams.num_filters
    kernel_size = self.hparams.kernel_size
    max_len = self.hparams.max_len
    num_rels = self.hparams.num_rels
    batch_size = self.hparams.batch_size

    with tf.device('/cpu:0'):
      embedding = tf.Variable(word2vec, dtype=tf.float32)
      self.inputs = tf.nn.embedding_lookup(embedding, tokens)
    
    with tf.name_scope('joint'):
      wpe1 = tf.get_variable("wpe1", shape=[123, pos_dim], initializer=xavier)
      wpe2 = tf.get_variable("wpe2", shape=[123, pos_dim], initializer=xavier)
      pos_left = tf.nn.embedding_lookup(wpe1, e1_dist)
      pos_right = tf.nn.embedding_lookup(wpe2, e2_dist)
      self.pos_embed = tf.concat([pos_left, pos_right], 2)

    with tf.name_scope('conv'):
      self._input = tf.concat([self.inputs, self.pos_embed], 2)
      self.conv = tf.layers.conv1d(self._input, num_filters, kernel_size,
              activation=tf.nn.relu, kernel_initializer=xavier, padding='same')
      # pool_max = tf.layers.max_pooling1d(self.conv, pool_size=max_len, strides=1)
      # pool_max = tf.squeeze(pool_max)
      pool_max = tf.reduce_max(self.conv, axis=1)
      pooled = tf.layers.dropout(pool_max, training=training)
      
    with tf.name_scope("att"):
      W = tf.get_variable(
          "W",
          shape=[num_filters, num_rels],
          initializer=xavier)
      b = tf.get_variable("b", shape=[num_rels], initializer=xavier)
      self.total_loss = 0.0
      self.total_loss += tf.nn.l2_loss(W)
      self.total_loss += tf.nn.l2_loss(b)

      # the implementation of Lin et al 2016 comes from https://github.com/thunlp/TensorFlow-NRE/blob/master/network.py
      sen_a = tf.get_variable("attention_A", [num_filters], initializer=xavier)
      sen_q = tf.get_variable("query", [num_filters, 1], initializer=xavier)
      sen_r = []
      sen_s = []
      sen_out = []
      sen_alpha = []
      self.bag_score = []
      self.predictions = []
      self.losses = []
      self.accuracy = []

      
      # selective attention model, use the weighted sum of all related the sentence vectors as bag representation
      n_bags = tf.shape(label)[0]
      ini_score_arr = tf.TensorArray(tf.float32, size=n_bags)
      def body(i, score_arr):
        sen_r = pooled[bag_idx[i]:bag_idx[i+1]] # shape (n_sent,feat_size)
        sen_alpha = tf.nn.softmax(
                          tf.matmul(
                            tf.multiply(sen_r, sen_a), sen_q, name='alpha'), 
                          axis=0
                        ) # (n_sent,1)
        sen_s = tf.matmul(sen_alpha, sen_r, transpose_a=True, name='att_sum') #(1,feat_size)
        bag_vec = tf.squeeze(tf.nn.xw_plus_b(sen_s, W, b)) # (num_rels)
        # bag_vec = tf.reshape(tf.nn.xw_plus_b(sen_s, W, b), [num_rels])
        return i+1, score_arr.write(i, tf.nn.softmax(bag_vec))
      _, bag_score_arr = tf.while_loop(lambda i, ta: i<n_bags, body, [0, ini_score_arr])
      self.bag_score = bag_score_arr.stack()

        # with tf.name_scope("output"):
        #   self.predictions.append(tf.argmax(self.bag_score[i], 0, name="predictions"))

        # with tf.name_scope("loss"):
        #   nscor = self.soft_label_flag[i] * self.bag_score[i] + joint_p * tf.reduce_max(self.bag_score[i])* tf.cast(self.preds[i], tf.float32)
        #   self.nlabel = tf.reshape(tf.one_hot(indices=[tf.argmax(nscor, 0)], depth=num_rels, dtype=tf.int32), [num_rels])
        #   self.ccc = self.preds[i]
        #   self.losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.nlabel)))

        #   if i == 0:
        #       self.total_loss = self.losses[i]
        #   else:
        #       self.total_loss += self.losses[i]

        # with tf.name_scope("accuracy"):
        #   self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.preds[i], 0)), "float"), name="accuracy"))

        # with tf.name_scope("update"):
        #   self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #   optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        #   self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

