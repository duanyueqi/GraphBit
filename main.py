import numpy as np
import copy
import scipy.io as sio
from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Activation
from keras.optimizers import Adam,RMSprop
from keras.layers.convolutional import Convolution2D
from keras.applications.vgg16 import VGG16
from RL_network import PGAgent
from US_network import USNet
from keras import backend as K
from keras.datasets import cifar10
import pickle
import matplotlib.pyplot as plt
#GPU to run
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
	'''
	    Parameters:
			dim_feature: the dimension of hashed feature, eg 16, 32, 64
			rate: a factor that balance two rewards, set to be 1
			num_epoch_total: the total epoch of two stage trainning
			num_epoch_us: epoch of unsupervised network
			num_epoch_rl: epoch of deep Q network
			batchsize: used in both US network and deep Q network, 32 is a very common option.
			alpha: Lamda_2 when assuming Lamda_1 = 1, 0.2 is a proper value
			beta: Lamda_3 when assuming Lamda_1 = 1, 0.4 is a proper value
			max_connection: the max capacity of connection between Nodes. In previous test,
							it would be at most 300 for 64 dimensions, so 10000 indicates an
							unlimited connection.
			x_train: it's the Cifar10 feature, extracted by a vgg16 pretrained on ImageNet.

	'''
	dim_feature = 16
	batchsize  = 32
	rate = 1
	learning_rate = 0.0004
	num_epoch_total = 1
	num_epoch_us = 5
	num_epoch_rl = 3
	alpha = 0.2
	beta = 0.4
	max_connection = 10000
	x_train = np.transpose(np.load('feat16_train.npy'))
	print(np.mean(x_train, axis=(0,1)))
	'''
	    Variables In Training:
			loss_bf: the loss of unsupervised network BEFORE connection
			xx: the node that is very reliable, which will 'guide' the non-reliable one.
			yy: the node that is not reliable, which will 'follow the guide' from reliable one.
			temp_xx: it indicates to the xx, but it's still in training process, so might not be the final one.
			temp_yy: similar to temp_xx
			finalrewards: sum of rewards after each action
			env: the unsupervised network
			rl: deep Q network
			state: a square metrix to document the state of connection
			score: rewards in one episode of actions
			episode: the same defination in RL

	'''

	loss_bf = 0
	xx = []
	yy = []
	temp_xx = []
	temp_yy = []
	finalrewards = []
	env = USNet(dim_feature, batchsize, alpha, beta)

	rl = PGAgent(dim_feature*dim_feature, dim_feature*dim_feature,dim_feature, batchsize)
	state = np.zeros((dim_feature, dim_feature))
	prev_x = None
	score = 0
	episode = 0
	act_times = 0
	rmsprop = RMSprop(lr=0.01)
	opt=RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

	for total_epoch in range(num_epoch_total):
		#pre-train for unsupervised network
		env.model.compile(loss=env.TOLoss(xx,yy),optimizer=opt)
		env.model.fit(x_train, x_train, batch_size=batchsize, epochs=num_epoch_us)
		num_layer = 0

		# deep Q Learning Process
		for epoch in range(num_epoch_rl):
			for num_minibatch in range(x_train.shape[0]/batchsize):
				x_train_batch = x_train[num_minibatch*batchsize:min((num_minibatch+1)*batchsize,x_train.shape[0]),:]
				loss_bf = 0
				# a loop that will break when either not connecting or disconnecting.
				while True:
					cur_x = state
					x = cur_x  if np.sum(cur_x) != 0 else np.zeros((1,dim_feature*dim_feature))
					x = np.reshape(x, (1, dim_feature*dim_feature))
					prev_x = copy.deepcopy(cur_x)

					act_times += 1
					loss_bf = np.sum(env.return_loss(x_train_batch,xx,yy))
					action1, action2, action3, prob = rl.act(x, act_times, cur_x)
					flag = 0
					fflag = 0
					if action1 > -1:
						i = 0
						if state[int(action1/dim_feature), action1%dim_feature] == 1 or state[action1%dim_feature, int(action1/dim_feature)] == 1 or int(action1/dim_feature) == action1%dim_feature:
							flag = 1
						if flag == 0 or (len(xx) == 0 and int(action1/dim_feature) != action1%dim_feature):
							xx.append(int(action1/dim_feature))
							yy.append(action1%dim_feature)

					loss_af1 = np.sum(env.return_loss(x_train_batch,xx,yy))
					reward1 = rate*(loss_bf-loss_af1)

					num_delete = 0

					if action2 > -1:
						if state[int(action2/dim_feature), action2%dim_feature] == 1:
							fflag = 1
							for i in range(len(xx)):
								if xx[i]==int(action2/dim_feature) and yy[i]==action2%dim_feature:
									del xx[i]
									del yy[i]
									break


					loss_af2 = np.sum(env.return_loss(x_train_batch,xx,yy))

					reward2 = rate*(loss_af1-loss_af2)

					print('loss',loss_bf,loss_af1,loss_af2)


					min_ambiguity = 0
					min_index = -1

					if len(xx) > max_connection and flag == 0:
						for i in range(len(xx)-1):
							min_ambiguity = max(min_ambiguity, abs(np.mean(np.mean(prob,axis = 0),axis = 0)[yy[i]]-0.5))
							if min_ambiguity==abs(np.mean(np.mean(prob,axis = 0), axis = 0)[yy[i]]-0.5):
								min_index = i
						action2 = xx[min_index]*dim_feature+yy[min_index]
						del xx[min_index]
						del yy[min_index]

					loss_af2 = np.sum(env.return_loss(x_train_batch,xx,yy))
					reward2 = rate*(loss_bf-loss_af2)

					state = np.zeros((dim_feature, dim_feature))
					if len(xx)>0:
						for i in range(len(xx)):
							state[xx[i],yy[i]] = 1

					if action3 == 1:
						done = 1
					else:
						done = 0

					score += reward1 + reward2
					rl.remember(x, action1, action2, prob, reward1, reward2)
					if done:
						episode += 1
						rl.train()
						rl.remove_thr = 1.0/(dim_feature*dim_feature)
						act_times = 0
						print('Episode: %d - Score: %f.' % (episode, score))

						score = 0
						prev_x = None
						break
			loss_bf = np.sum(env.return_loss(x_train,[],[]))
			loss_af = np.sum(env.return_loss(x_train,xx,yy))
			print('final_reward', loss_bf-loss_af)
			finalrewards.append(loss_bf-loss_af)

	print('finalrewards',finalrewards)
	print('xx',xx,yy,alpha,beta)

	with open('xx.bin','wb') as xx_bin:
		pickle.dump(xx,xx_bin)
	with open('yy.bin','wb') as yy_bin:
		pickle.dump(yy,yy_bin)

	env.model.compile(loss=env.TOLoss(xx,yy),optimizer=opt)

	env.model.fit(x_train, x_train, batch_size=batchsize, epochs=num_epoch_us )

	x_test = np.transpose(np.load('feat16_test.npy'))

	num_post = [1]*dim_feature
	prob = np.zeros((x_train.shape[0], dim_feature))
	w = env.model.predict(x_train)
	for j in range(dim_feature):
		prob[:,j] = w[:,j*dim_feature+j]
		for i in range(len(xx)):
			if j == yy[i]:
				prob[:,yy[i]] += w[:,xx[i]*dim_feature+yy[i]]
				num_post[yy[i]] += 1
		prob[:,j] /= num_post[j]


	prob_train = prob


	num_post = [1]*dim_feature
	prob = np.zeros((x_test.shape[0],dim_feature))
	w = env.model.predict(x_test)

	for j in range(dim_feature):
		prob[:,j] = w[:,j*dim_feature+j]
		for i in range(len(xx)):
			if j == yy[i]:
				prob[:,yy[i]] += w[:,xx[i]*dim_feature+yy[i]]
				num_post[yy[i]] += 1
		prob[:,j] /= num_post[j]


	prob_test = prob

	trainName = '%d_feat_train.mat' % dim_feature
	testName =  '%d_feat_test.mat' % dim_feature

	sio.savemat(trainName, {'prob_train':prob_train})
	sio.savemat(testName, {'prob_test':prob_test})
	print('test',prob_test)
	print('alpha')
