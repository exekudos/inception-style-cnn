import tensorflow as tf
import numpy as np
import data

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 40, 40, 1])
	conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[7,7], padding="valid", activation=tf.nn.relu, kernel_initializer=None)
	conv1_norm = tf.layers.batch_normalization(inputs=conv1)
	conv2 = tf.layers.conv2d(inputs=conv1_norm, filters=48, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	conv2_norm = tf.layers.batch_normalization(inputs=conv2)
	#inception block1
	conv3a = tf.layers.conv2d(inputs=conv2_norm, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	conv3a_norm = tf.layers.batch_normalization(inputs=conv3a)
	pool3b = tf.layers.max_pooling2d(inputs=conv3a_norm, pool_size=[2,2], strides=2)
	conv4a = tf.layers.conv2d(inputs=conv2_norm, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
	pool4b = tf.layers.max_pooling2d(inputs=conv4a, pool_size=[2,2], strides=2)
	conv5a = tf.layers.conv2d(inputs=conv2_norm, filters=64, kernel_size=[3,1], padding="same", activation=tf.nn.relu)
	conv5b = tf.layers.conv2d(inputs=conv5a, filters=64, kernel_size=[1,3], padding="same", activation=tf.nn.relu)
	conv5b_norm = tf.layers.batch_normalization(inputs=conv5b)
	conv5c = tf.layers.conv2d(inputs=conv5b_norm, filters=64, kernel_size=[1,1], strides=2, padding="same", activation=tf.nn.relu)
	#block1 ends
	conc6 = tf.concat(values=[pool3b, pool4b, conv5c], axis=3)
	conv7 = tf.layers.conv2d(inputs=conc6, filters=256, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	conv7_norm = tf.layers.batch_normalization(inputs=conv7)
	pool8 = tf.layers.max_pooling2d(inputs=conv7_norm, pool_size=[3,3], strides=2)
	#inception block2
	conv9a = tf.layers.conv2d(inputs=pool8, filters=256, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	conv9a_norm = tf.layers.batch_normalization(inputs=conv9a)
	pool9b = tf.layers.max_pooling2d(inputs=conv9a_norm, pool_size=[2,2], strides=2)
	conv10a = tf.layers.conv2d(inputs=pool8, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
	conv10a_norm = tf.layers.batch_normalization(inputs=conv10a)
	pool10b = tf.layers.max_pooling2d(inputs=conv10a_norm, pool_size=[2,2], strides=2)
	conv11a = tf.layers.conv2d(inputs=pool8, filters=256, kernel_size=[3,1], padding="same", activation=tf.nn.relu)
	conv11b = tf.layers.conv2d(inputs=conv11a, filters=256, kernel_size=[1,3], padding="same", activation=tf.nn.relu)
	conv11c = tf.layers.conv2d(inputs=conv11b, filters=256, kernel_size=[1,1], strides=2, padding="same", activation=tf.nn.relu)
	#block2 ends
	conc12 = tf.concat(values=[pool9b, pool10b, conv11c], axis=3)
	conc12_flat = tf.reshape(conc12, [-1, 4*4*768])
	dense1 = tf.layers.dense(inputs=conc12_flat, units=4096, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
	dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout2, units=7)
	
	predictions = {"classes":tf.argmax(input=logits, axis=1), "probabilities":tf.nn.softmax(logits, name="softmax_tensor")}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=7)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op=optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		#print("Training accuracy:",tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name="accuracy"))
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	train_data = np.asarray(data.train_val, dtype=np.float32)
	train_labels = np.asarray(data.train_label, dtype=np.int32)
	eval_data = np.asarray(data.test_val, dtype=np.float32)
	eval_labels = np.asarray(data.test_label, dtype=np.int32)
	classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="drive/project/store")
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
	#train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=32, num_epochs=None, shuffle=True)
	#classifier.train(input_fn=train_input_fn, steps=None)
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__=="__main__":
	tf.app.run()