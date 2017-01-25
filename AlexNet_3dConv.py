
import numpy as np
import tensorflow as tf


# Path to 3d tensor. Tensor.shape is (111,111,111)
tensor_path = 'path/to/tensor/'


X = np.load(tensor_path + 'x.npz')['x'].reshape((-1, 111, 111, 111, 1))
Y = np.load(tensor_path + 'y.npz')['y']


# Accuracy function
def get_accuracy(predictions, labels):
  return 100 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1)), tf.float32))


# Graph
batch_size = 1000
num_labels = 10


graph = tf.Graph()

with graph.as_default():

    predict = tf.Variable(False)
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 111, 111, 111, 1))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [9, 9, 9, 1, 96], stddev=0.1))

    layer1_biases = tf.Variable(tf.zeros([96]))

    layer2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 5, 96, 256], stddev=0.1))
     
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[256]))

    layer3_weights = tf.Variable(tf.truncated_normal(
      [3, 3, 3, 256, 384], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[384]))

    layer4_weights = tf.Variable(tf.truncated_normal(
      [3, 3, 3, 384, 384], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[384]))

    layer5_weights = tf.Variable(tf.truncated_normal(
      [3, 3, 3, 384, 256], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[256]))

    layer6_weights = tf.Variable(tf.truncated_normal(
      [49*49*256, 4096], stddev=0.1))
    layer6_biases = tf.Variable(tf.constant(1.0, shape=[4096]))

    layer7_weights = tf.Variable(tf.truncated_normal(
      [4096, 4096], stddev=0.1))
    layer7_biases = tf.Variable(tf.constant(1.0, shape=[4096]))

    layer8_weights = tf.Variable(tf.truncated_normal(
      [4096, num_labels], stddev=0.1))
    layer8_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    #MODEL     
    def model(data):
        # Conv1         
        conv1 = tf.nn.conv3d(data, layer1_weights, [1, 4, 4, 4, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + layer1_biases)

        #Pool1
        pool1 = tf.nn.max_pool3d(hidden1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        # Conv2
        conv2 = tf.nn.conv3d(pool1, layer2_weights, [1, 1, 1, 1, 1],padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)

        # Conv3
        conv3 = tf.nn.conv3d(hidden2, layer3_weights, [1, 1, 1, 1, 1],padding='SAME')

        # Conv4
        conv4 = tf.nn.conv3d(conv3, layer4_weights, [1, 1, 1, 1, 1], padding='SAME')

        # Conv5
        conv5 = tf.nn.conv3d(conv4, layer5_weights, [1, 1, 1, 1, 1], padding='SAME')

        #Pool2
        pool2 = tf.nn.max_pool3d(conv5, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        
        normalize3_flat = tf.reshape(pool2, [-1, 49*49*256])

        #FC1
        fc1 = tf.tanh(tf.add(tf.matmul(normalize3_flat, layer6_weights), layer6_biases))
        dropout1 = tf.nn.dropout(fc1, 0.5)

        #FC2
        fc2 = tf.tanh(tf.add(tf.matmul(dropout1, layer7_weights), layer7_biases))
        dropout2 = tf.nn.dropout(fc2, 0.5)

        #FC3
        res = tf.nn.softmax(tf.add(tf.matmul(dropout2, layer8_weights), layer8_biases))
        return res

    
     # Training computation
    local_res = model(tf_train_dataset)

    with tf.name_scope("cost_function") as scope:
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(local_res), reduction_indices=[1]))
        tf.scalar_summary("cost_function", cross_entropy)
    
    # Optimizer
    train_step = tf.train.MomentumOptimizer(0.0014, 0.9).minimize(cross_entropy)

    # Predictions for the training, validation, and test data
    with tf.name_scope("accuracy") as scope:
        accuracy = get_accuracy(local_res, tf_train_labels)
        tf.scalar_summary("accuracy", accuracy)
    

    valid_prediction = tf.nn.softmax(model(tf_train_dataset))
    print ('Graph was built')
    
    merged_summary_op = tf.merge_all_summaries()


# Session
epochs = 100
steps_per_epoch = int(Y.shape[0]/batch_size) + 1
print ('STEPS %d' % steps_per_epoch)

with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())
    
    for epch in xrange(0, epochs):
		print ('EPOCH %d' % epch)
   
        for step in range(steps_per_epoch):
			offset = (step * batch_size) % (Y.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = X[np.arange(offset,(offset + batch_size))].astype('float32')
            batch_labels = Y[offset:(offset + batch_size), :]

            train_step.run(feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels})

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()


            summary_str, _ =  session.run([merged_summary_op, train_step], 
                                       feed_dict={tf_train_dataset: batch_data, tf_train_labels: batch_labels},
                                       options=run_options,
                                       run_metadata=run_metadata)

            train_writer.add_run_metadata(run_metadata, 'step%03d' % (int(step)+(steps_per_epoch * (epch+1))))
            train_writer.add_summary(summary_str, step)

            train_accuracy = accuracy.eval(feed_dict={
                tf_train_dataset:batch_data, tf_train_labels: batch_labels})

            print("Step %d" % step)
            print("Minibatch accuracy: %.1f%%" % train_accuracy)