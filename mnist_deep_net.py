#!/usr/bin/env python
# coding: utf-8

# # MNIST with Tensorflow

# In[36]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[37]:


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nodes_hidden_layer1 = 500
nodes_hidden_layer2 = 500
nodes_hidden_layer3 = 500

number_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def nn_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, nodes_hidden_layer1])),
                     'biases': tf.Variable(tf.random_normal([nodes_hidden_layer1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layer1, nodes_hidden_layer2])),
                     'biases': tf.Variable(tf.random_normal([nodes_hidden_layer2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layer2, nodes_hidden_layer3])),
                     'biases': tf.Variable(tf.random_normal([nodes_hidden_layer3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hidden_layer3, number_classes])),
                     'biases': tf.Variable(tf.random_normal([number_classes]))}
    
    # input * weights + bias
    
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_1)
    
    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)
    
    output = tf.add(tf.matmul(layer_3, output_layer['weights']), output_layer['biases'])
    
    return output
    
def train_nn(x):
    predict = nn_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    number_epochs = 20

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for n in range(number_epochs + 1):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('Epoch ', n, '/', number_epochs, ' loss: ', epoch_loss, end=' ')

            correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
            
train_nn(x)


# In[ ]:





# In[ ]:




