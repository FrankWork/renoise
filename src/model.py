import tensorflow as tf

class CNNModel(object):
  def __init__(self, params, word2vec, features, labels, training=False):
    e1, e2, bag_size, bag_idx, seq_len, pcnn_mask, tokens, e1_dist, e2_dist = features
    
    xavier = tf.contrib.layers.xavier_initializer()
    
    self.params = params
    pos_dim = self.params["pos_dim"]
    num_filters = self.params["num_filters"]
    gru_size = num_filters
    rnn_layers = 1
    kernel_size = self.params["kernel_size"]
    max_len = self.params["max_len"]
    num_rels = self.params["num_rels"]
    batch_size = self.params["batch_size"]
    learning_rate = self.params["learning_rate"]
    keep_prob = 0.5
    l2_coef = self.params["l2_coef"]
    self.training = training


    with tf.name_scope('embed'):
      embedding = tf.get_variable("word2vec", initializer=word2vec)
      pos1_embed = tf.get_variable("pos1_embed", shape=[123, pos_dim], initializer=xavier)
      pos2_embed = tf.get_variable("pos2_embed", shape=[123, pos_dim], initializer=xavier)

      with tf.device('/cpu:0'):
        tokens = tf.nn.embedding_lookup(embedding, tokens)
        pos1 = tf.nn.embedding_lookup(pos1_embed, e1_dist)
        pos2 = tf.nn.embedding_lookup(pos2_embed, e2_dist)
      inputs = tf.concat([tokens, pos1, pos2], 2)
      # inputs = tf.layers.dropout(inputs, rate=0.2, training=self.training)

    with tf.name_scope('cnn-encoder'):
      conv = tf.layers.conv1d(inputs, num_filters, kernel_size,
              activation=tf.nn.relu, kernel_initializer=xavier, padding='same')
      sentence_vec = tf.reduce_max(conv, axis=1)
      # pcnn_mask = tf.expand_dims(pcnn_mask, axis=1, name='pcnn_mask')  # [batch, 1, L, 3]
      # conv4d =  tf.expand_dims(tf.transpose(conv, [0, 2, 1]), axis=-1, name='conv4d')  # [batch, feat, L, 1]
      # mask_conv = tf.multiply(conv4d, pcnn_mask, name='mask_conv')
      # pcnn_pool = tf.reduce_max(mask_conv, axis=2)  # [batch, feat, 3]
      # sentence_vec = tf.reshape(pcnn_pool, [-1, num_filters * 3])
      sentence_vec = tf.layers.dropout(sentence_vec, training=self.training)

    # with tf.name_scope('rnn-encoder'):
    #   cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
    #   cell_backward = tf.nn.rnn_cell.GRUCell(gru_size)

    #   if training:
    #     cell_forward = tf.nn.rnn_cell.DropoutWrapper(
    #           cell_forward, output_keep_prob=keep_prob)
    #     cell_backward = tf.nn.rnn_cell.DropoutWrapper(
    #           cell_backward, output_keep_prob=keep_prob)
      
    #   outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, 
    #                 inputs, sequence_length=seq_len, dtype=tf.float32)
    #   output_h = tf.add(outputs[0],outputs[1]) # [b', n, d]

    #   with tf.name_scope('rnn_attention'):
    #     alpha = tf.layers.dense(tf.nn.tanh(output_h), 1, use_bias=False) # b,n,1
    #     alpha = tf.nn.softmax(alpha, axis=1)
    #     sentence_vec = tf.matmul(alpha, output_h, transpose_a=True)
    #     sentence_vec = tf.squeeze(sentence_vec, axis=1)

    with tf.name_scope("attention"):
      # the implementation of Lin et al 2016 comes from 
      # https://github.com/thunlp/TensorFlow-NRE/blob/master/network.py
      num_units = sentence_vec.get_shape().as_list()[-1]
      sen_a = tf.get_variable("attention_A", [num_units], initializer=xavier)
      sen_q = tf.get_variable("query", [num_units, 1], initializer=xavier)

      # selective attention model, use the weighted sum of all related the sentence vectors as bag representation
      n_bags = tf.shape(labels)[0]
      def fn(i):
        with tf.name_scope('debug'):
          with tf.name_scope('dynamic'):
            sen_r = tf.tanh(sentence_vec[bag_idx[i]:bag_idx[i+1]]) # shape (n_sent,feat_size)
            # sen_r = sentence_vec[bag_idx[i]:bag_idx[i+1]] # shape (n_sent,feat_size)
          sen_alpha = tf.nn.softmax(
                            tf.matmul(
                              tf.multiply(sen_r, sen_a), sen_q, name='alpha_mul'), 
                            axis=0,
                            name='alpha') # (n_sent,1)
          sen_s = tf.matmul(sen_alpha, sen_r, transpose_a=True, name='att_sum') #(1,feat_size)
          return tf.squeeze(sen_s)
      bag_vec = tf.map_fn(fn, tf.range(n_bags), dtype=tf.float32, 
                    parallel_iterations=50)

      # bag_vec = tf.layers.dropout(bag_vec, training=self.training) # (n_bag, feat_size)
      regularizer = None# tf.contrib.layers.l2_regularizer(scale=0.0001)
      self.bag_score = tf.layers.dense(bag_vec, num_rels, kernel_regularizer=regularizer)
      
      
    with tf.name_scope("output"):
      self.total_loss = tf.reduce_mean(
                              tf.nn.softmax_cross_entropy_with_logits_v2(
                                  labels=tf.one_hot(labels, num_rels), 
                                  logits=self.bag_score))
      # self.total_loss += tf.losses.get_regularization_loss()

      self.prob = tf.nn.softmax(self.bag_score, axis=1)
      self.predictions = tf.argmax(self.bag_score, axis=1, name="predictions")
      self.accuracy = tf.metrics.accuracy(labels=labels,
                                    predictions=self.predictions)

      # ignore label 0
      # mask = tf.range(num_rels)
      # mask = tf.cast(mask>0, tf.float32)
      # self.mask_prob = self.prob * mask
      # self.mask_predictions = tf.argmax(self.mask_prob, axis=1, name="mask_predictions")
      
      label_mask = labels > 0
      mask_labels = tf.boolean_mask(labels, label_mask)
      mask_predictions = tf.boolean_mask(self.predictions, label_mask)
      
      self.mask_accuracy = tf.metrics.accuracy(labels=mask_labels,
                                    predictions=mask_predictions)

    if self.training:
      with tf.name_scope("training"):
        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)

