import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

def train_input_fn():
  """An input function for training"""
  # Convert the inputs to a Dataset.
  inputs = tf.random_normal([100, 3])
  labels = tf.random_uniform([100],
    minval=0,
    maxval=2,
    dtype=tf.int32,)

  dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000)

  # Return the dataset.
  return dataset

def test_input_fn():
  return train_input_fn()

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.expand_dims(features, axis=0)
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    # Compute loss.
    labels = tf.expand_dims(labels, axis=0)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    predicted_classes = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
    "accuracy" : accuracy[1]}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, 
              training_hooks = [logging_hook])

classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 11],
            # The model must choose between 3 classes.
            'n_classes': 2,
        })
classifier.train(
        input_fn=train_input_fn,
        steps=100)

eval_result = classifier.evaluate(input_fn=test_input_fn)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
