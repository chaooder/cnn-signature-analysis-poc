import numpy as np
import tensorflow as tf

customers = ['002', '046']

def cnn_model_fn(features, labels, mode):
	"""
	Proof of Concept CNN Topology,

	input --> conv1 --> maxpool1 --> conv2 --> maxpool2 --> 
	dense1 --> dropout1 --> dense2 --> dropout2 --> logits --> predictions 
	"""
	input_layer = tf.reshape(tf.cast(features, tf.float32),[-1,30,40,1])
	
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 3,
		kernel_size = [3,3],
		padding = "same",
		activation = tf.nn.relu
	)
		
	pool1 = tf.layers.max_pooling2d(
		inputs = conv1,
		pool_size = [2,2],
		strides = 2
	)
	
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 5,
		kernel_size = [3,3],
		padding = "same",
		activation = tf.nn.relu
	)
	
	pool2 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size = [2,2],
		strides = 2
	)
	
	pool2_flat = tf.reshape(pool2, [-1, 7*10*5])
	
	dense1 = tf.layers.dense(
		inputs = pool2_flat,
		units = 300,
		activation = tf.nn.relu
	)
	
	dropout1 = tf.layers.dropout(
		inputs = dense1,
		rate = 0.4,
		training = mode == tf.estimator.ModeKeys.TRAIN
	)
	
	dense2 = tf.layers.dense(
		inputs = dropout1,
		units = 150,
		activation = tf.nn.relu
	)
	
	dropout2 = tf.layers.dropout(
		inputs = dense2,
		rate = 0.4,
		training = mode == tf.estimator.ModeKeys.TRAIN
	)
	
	logits = tf.layers.dense(
		inputs = dropout2,
		units = 2*len(customers)
	)
	
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode, 
			predictions=predictions
		)
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2*len(customers))
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
	}
	return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)