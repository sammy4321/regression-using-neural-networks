import numpy as np
import pandas as pd
import tensorflow as tf
n_nodes_hl1=5
n_nodes_hl2=5
n_nodes_hl3=5

n_classes=1
xx=np.arange(1000,dtype=np.float).reshape([-1,1])

yy=np.arange(1000,dtype=np.float).reshape([-1,1])

x=tf.placeholder('float')
y=tf.placeholder('float')

def neural_network_model(data):
	hidden_layer_1={'weights':tf.Variable(tf.random_normal([1,n_nodes_hl1])),
	'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_layer_2={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
	'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_layer_3={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
	'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,1])),
	'biases':tf.Variable(tf.random_normal([1]))}

	l1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	#l1=tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	#l2=tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	#l3=tf.nn.relu(l3)

	output=tf.matmul(l3,output_layer['weights'])+output_layer['biases']

	return output

def train_neural_network(x):
	prediction=neural_network_model(x)

	cost=tf.losses.mean_squared_error(labels=y,predictions=prediction)
	optimizer=tf.train.AdamOptimizer().minimize(cost)

	hm_epochs=200
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			epoch_loss=0;epoch_x=xx;epoch_y=yy
			_,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
			epoch_loss+=c
			print('Epoch',epoch,'Completed out of',hm_epochs,'loss:',epoch_loss)
		pred=sess.run(prediction,feed_dict={x:epoch_x})
		print(pred)


train_neural_network(x)
