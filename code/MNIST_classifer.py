import numpy as np 
import pandas as pd
import tensorflow as tf
import math
import random
import json
import time

def main():

	'''Management Data'''
	data_path='../input/'
	model_folder='./'
	model_Name='MNIST.ckpt'
	init_flag=None    # if first training
	test_report_name='MNIST-TestReport.txt'
	log_file_name='MNIST_Log.txt'
	loss_log_file_train='MNIST_LossLog_train.txt'
	loss_log_file_val='MNIST_LossLog_val.txt'
	loss_log_file_test='MNIST_LossLog_test.txt'

	train_BS=32 # batch size for train
	val_BS=50 # for validation

	ckpt_offset=0
	dropout_rate=0.5
	one_epoch=int(32000/train_BS)
	total_step=one_epoch*1000 # 5000 epochs
	val_bias=one_epoch # validate performance every 1 epoch
	save_bias=one_epoch*100 # save every 50 epochs 
	trainloss_bias=25 # record loss evey 25 steps
	lrdecay_bias=one_epoch*20 #test performance every 50 epochs by using whole Test set
	lrdecay_rate=0.97
	val_quan=int(10000/val_BS)
	learning_rate=5e-2
	momentum_rate=0.9
	cnn_layerNum=2 #CNN layers
	cnn_layer=[[3,3,1,64],[3,3,64,128],7*7*128]
	layerNum=2 
	layer=[1024,1024]
	'''Management Data'''


	if ckpt_offset<=0:
		init_flag=True
	else:
		init_flag=False

	'''Data prepare, rotate or shuffle'''
	TrainSet,ValSet=data_preparer(data_path)

	print('Data load finished')
	for shuf in range(5):
		random.shuffle(TrainSet)
		random.shuffle(ValSet)
	print('Data shuffle finished')
	print('Data prepare finished')
	'''Data prepare, rotate or shuffle'''
	if init_flag:
		'''Placeholder Data '''
		x=tf.placeholder(tf.float32, [None, 784])
		y_=tf.placeholder(tf.float32, [None,10])
		x_input=tf.reshape(x,[-1,28,28,1])
		ist = tf.placeholder(tf.bool,name='ist')
		lr=tf.placeholder(tf.float32)
		'''Placeholder Data'''

		'''output '''
		y,keep_prob=MLP(x_input,layerNum,layer,cnn_layerNum,cnn_layer,ist)
		result=tf.argmax(y, 1)
		'''output '''

		'''loss & train'''
		final_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_step = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum_rate).minimize(final_loss)
		'''loss & train'''

		'''dash data'''
		#final_prob=tf.reduce_mean(tf.cast(tf.equal(tf.round(y), y_), tf.float32))
		final_prob=tf.reduce_mean(tf.cast(tf.equal(result, tf.argmax(y_,1)), tf.float32))
		'''dash data'''
		tf.add_to_collection('train_step', train_step)
		tf.add_to_collection('final_loss', final_loss)
		tf.add_to_collection('final_prob', final_prob)
		tf.add_to_collection('lr', lr)
		tf.add_to_collection('ist', ist)
		tf.add_to_collection('x', x)
		tf.add_to_collection('y', y)
		tf.add_to_collection('result', result)		
		tf.add_to_collection('y_', y_)
		tf.add_to_collection('keep_prob', keep_prob)
	else:
		saver = tf.train.import_meta_graph( model_folder+model_Name+'-'+str(ckpt_offset)+ '.meta')


	train_file=open(model_folder+loss_log_file_train, 'a')
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if init_flag:
			sess.run(tf.global_variables_initializer())
		else :
			saver.restore(sess, model_folder+model_Name+'-'+str(ckpt_offset))
			final_loss=tf.get_collection('final_loss')[0]
			x=tf.get_collection('x')[0]
			y_=tf.get_collection('y_')[0]
			ist=tf.get_collection('ist')[0]
			lr=tf.get_collection('lr')[0]
			keep_prob=tf.get_collection('keep_prob')[0]
			final_prob=tf.get_collection('final_prob')[0]
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_step=tf.get_collection('train_step')[0]
			print('Model Restored! ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		print('Training Start! ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		for i in range(total_step): # training begin
			timeStart=time.time()
			if (i) % val_bias == 0:
				random.shuffle(TrainSet)
				'''for validation'''
				finalLoss=0
				finalProb=0
				for vi in range(val_quan):
					val_batch=next_feed(val_BS,ValSet) 
					eval_loss,eval_prob=sess.run(fetches=[final_loss,final_prob],feed_dict={x:val_batch[0],y_:val_batch[1],keep_prob:1.0,ist:False})
					finalLoss=finalLoss+eval_loss
					finalProb=finalProb+eval_prob
				report_loss(i+ckpt_offset,model_folder,loss_log_file_val,finalLoss,finalProb,val_quan)
				report_val(i,ckpt_offset,model_folder,log_file_name,finalLoss,finalProb,val_quan,learning_rate)
			
			if(i+1)%lrdecay_bias==0:
				learning_rate=learning_rate*lrdecay_rate
			
			if(i+1)%save_bias== 0:
				'''for save'''
				saver.save(sess, model_folder+model_Name, global_step=i+1+ckpt_offset)
				train_file.flush()

			train_batch=next_feed(train_BS,TrainSet)
			if(i+ckpt_offset)%trainloss_bias==0: 
				train_loss,train_prob=sess.run(fetches=[final_loss,final_prob],feed_dict={x:train_batch[0],y_:train_batch[1],keep_prob:1.0,ist:False})
				report_trainloss(train_file,i+ckpt_offset,train_loss,train_prob)
			train_step.run(feed_dict={x:train_batch[0],y_:train_batch[1],keep_prob:dropout_rate,ist:True,lr:learning_rate})
		train_file.close()



		sess.close()
		print('Done',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def MLP(input_x,layers_num,layers,cnn_layers_num,cnn_layers,ist):

  keep_prob_MLP = tf.placeholder(tf.float32)# dropout rate
  W_MLP = weight_variable(cnn_layers[0],'CNN_layer0')
  h_MLP=max_pool_2x2(af1(batch_norm(conv2d(input_x, W_MLP),ist)))
  for i in range(1,cnn_layers_num):
  	W_MLP = weight_variable(cnn_layers[i],'CNN_layer'+str(i))
  	h_MLP = max_pool_2x2(af1(batch_norm(conv2d(h_MLP, W_MLP),ist)))

  h_MLP=tf.reshape(h_MLP,[-1,cnn_layers[-1]])

  W_MLP = weight_variable([cnn_layers[-1], layers[0]],'MLP_layer0')
  h_MLP = tf.nn.dropout(af2(batch_norm(tf.matmul(h_MLP, W_MLP),ist)),keep_prob_MLP)

  for i in range(1,layers_num):
  	W_MLP = weight_variable([layers[i-1], layers[i]],'MLP_layer'+str(i))
  	h_MLP = tf.nn.dropout(af2(batch_norm(tf.matmul(h_MLP, W_MLP),ist)),keep_prob_MLP)
  
  W_mark = weight_variable([layers[-1], 10],'mark_layer')
  final_mark = batch_norm(tf.matmul(h_MLP, W_mark),ist)
  return final_mark,keep_prob_MLP

def batch_norm(x,ist):
  '''Batch Norm'''
  return tf.contrib.layers.batch_norm(inputs=x,is_training=ist,fused=True,data_format='NHWC',epsilon=1e-3,scale=True,decay=0.9)


def conv2d(x, W,s=[1, 1, 1, 1],p='SAME'):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(input=x,filter=W, strides=s, padding=p,data_format='NHWC')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape,name):
  """weight_variable generates a weight variable of a given shape."""
  return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),regularizer=tf.contrib.layers.l2_regularizer(5e-2))


def af1(x):
  '''change setting for different activation function'''
  return tf.nn.elu(x)
def af2(x):
  '''change setting for different activation function'''
  return tf.nn.tanh(x)

def data_preparer(path):
	'''prepare data before training'''
	'''train set processing'''
	train = pd.read_csv(path+"train.csv")
	label=train.pop('label') #target
	train_x=np.array(train).tolist() 
	train_y=np.array(label).tolist()
	TrainSet=[]
	for i in range(len(train_x)):
		oneHot=[0,0,0,0,0,0,0,0,0,0]
		oneHot[train_y[i]]=1
		TrainSet.append({'x':train_x[i],'y':oneHot})
		#print(TrainSet[i]['y'])
	random.shuffle(TrainSet)
	ValSet=TrainSet[32000:42000]
	TrainSet=TrainSet[0:32000]
	#print(TrainSet)

	'''test set processing'''

	return TrainSet,ValSet

def next_feed(num,dataArr):
  '''pop the next feed in batch size number'''
  iArr=[]
  oArr=[]
  for i in range(num):
    p=dataArr.pop(0)
    dataArr.append(p)
    iArr.append(p['x'])
    oArr.append(p['y'])
  return [iArr,oArr]

def report_trainloss(file,steps,loss,prob):
  """output the log of loss value and accuracy during train"""
  jsonBody={'steps':steps,'loss':loss/1,'prob':prob/1}
  file.write(json.dumps(jsonBody)+'\n')


def report_loss(steps,folder_path,file_name,loss,prob,quan):
  """output the log of loss value and accuracy after test or each validation"""
  log_file=open(folder_path+file_name, 'a')
  jsonBody={'steps':steps,'loss':loss/quan,'prob':prob/quan,'quan':quan}
  log_file.write(json.dumps(jsonBody)+'\n')
  log_file.close()


def report_val(i,ckpt_offset,folder_path,log_file_name,loss,prob,val_quan,lr):
  """output the log report of after each validation"""
  print('---------Validation--------')
  print('Step',i+ckpt_offset,' '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  print('loss %g' % (loss/val_quan))
  print('prob %g' % (prob/val_quan))
  print('lr %g' % (lr))
  print('###########################')
  log_file=open(folder_path+log_file_name, 'a')
  log_file.write('---------Validation--------'+'\n')
  log_file.write('Step'+str(i+ckpt_offset)+' '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
  log_file.write('loss ' + str(loss/val_quan)+'\n')
  log_file.write('prob ' + str(prob/val_quan)+'\n')
  log_file.write('lr ' + str(lr)+'\n')
  log_file.write('###########################'+'\n')
  log_file.close()




if __name__ == '__main__':
	#data_preparer('../input/')
	main()