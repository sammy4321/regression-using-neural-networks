import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')

cdcd=0
n_nodes_hl1=5
n_nodes_hl2=5
n_nodes_hl3=5

n_classes=1
xx=np.arange(0,10,0.1,dtype=np.float).reshape([-1,1])

#yy=np.arange(1000,dtype=np.float).reshape([-1,1])
yy=xx*xx

x=tf.placeholder('float')
y=tf.placeholder('float')

def neural_network_model(data):
	global cdcd
	hidden_layer_1={'weights':tf.get_variable('w1',shape=[1,n_nodes_hl1],initializer=tf.contrib.layers.xavier_initializer()),
	'biases':tf.get_variable('b1',shape=[n_nodes_hl1],initializer=tf.contrib.layers.xavier_initializer())}
	hidden_layer_2={'weights':tf.get_variable('w2',shape=[n_nodes_hl1,n_nodes_hl2],initializer=tf.contrib.layers.xavier_initializer()),
	'biases':tf.get_variable('b2',shape=[n_nodes_hl2],initializer=tf.contrib.layers.xavier_initializer())}
	hidden_layer_3={'weights':tf.get_variable('w3',shape=[n_nodes_hl2,n_nodes_hl3],initializer=tf.contrib.layers.xavier_initializer()),
	'biases':tf.get_variable('b3',shape=[n_nodes_hl3],initializer=tf.contrib.layers.xavier_initializer())}
	output_layer={'weights':tf.get_variable('outw',shape=[n_nodes_hl3,1],initializer=tf.contrib.layers.xavier_initializer()),
	'biases':tf.get_variable('outb',shape=[1],initializer=tf.contrib.layers.xavier_initializer())}
	cdcd=hidden_layer_1

	l1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	l1=tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	l2=tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	l3=tf.nn.relu(l3)

	output=tf.matmul(l3,output_layer['weights'])+output_layer['biases']

	return output

def train_neural_network(x):
	prediction=neural_network_model(x)

	cost=tf.losses.mean_squared_error(labels=y,predictions=prediction)
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	cost_list=[]
	epoch_list=[]
	cost_list_scatter=[]
	epoch_list_scatter=[]
	weight_list_0=[]

	hm_epochs=20000
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss=0;epoch_x=xx;epoch_y=yy
			_,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
			epoch_loss+=c
			print('Epoch',epoch,'Completed out of',hm_epochs,'loss:',epoch_loss)
			cost_list.append(c)
			cd_cd=sess.run(cdcd['weights'])
			#print(cd_cd[0][0])
			weigt=cd_cd[0][0]
			weight_list_0.append(weigt)
			epoch_list.append(epoch)
			if epoch % 100 == 0:
				cost_list_scatter.append(c)
				epoch_list_scatter.append(epoch)
			
		
		pred=sess.run(prediction,feed_dict={x:epoch_x})
		#print(pred)
		x_x=xx.reshape([-1])
		y_y=pred.reshape([-1])
		#axess=plt.gca()
		#axess.set_xlim([0,100])
		#axess.set_ylim([0,100])
		#plt.title('Cost Graph')
		#plt.xlabel('Number of epochs')
		#plt.ylabel('Cost')
		#plt.plot(epoch_list,cost_list)
		plt.subplot(1,3,1,)
		plt.plot(epoch_list,cost_list)
		#plt.scatter(epoch_list_scatter,cost_list_scatter)
		plt.title('Cost Graph')
		plt.xlabel('Epochs')
		plt.ylabel('Cost Values')

		plt.subplot(1,3,3)


		plt.plot(x_x,y_y)
		
		plt.title('Prediction')
		plt.xlabel('X->')
		plt.ylabel('Y->')
		plt.show()



train_neural_network(x)
