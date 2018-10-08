import numpy as np 
import pandas as pd
import tensorflow as tf
import math
import random
import json
import time
import csv

def main():

	'''Management Data'''
	data_path='../input/'
	model_folder='../model/exp6/'
	model_Name='MNIST.ckpt'
	ckpt_offset=65600

	'''Data prepare, rotate or shuffle'''
	TestSet=data_preparer(data_path)
	'''Data prepare, rotate or shuffle'''
	saver = tf.train.import_meta_graph( model_folder+model_Name+'-'+str(ckpt_offset)+ '.meta')
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, model_folder+model_Name+'-'+str(ckpt_offset))
		final_loss=tf.get_collection('final_loss')[0]
		x=tf.get_collection('x')[0]
		y=tf.get_collection('y')[0]
		r=tf.get_collection('result')[0]
		ist=tf.get_collection('ist')[0]
		keep_prob=tf.get_collection('keep_prob')[0]
		print('Model Restored! ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		data_Id=[]
		data_Label=[]
		for i in range(len(TestSet)):
			test_batch=next_feed(1,TestSet)
			result=r.eval(feed_dict={x:test_batch[0],keep_prob:1.0,ist:False})
			data_Id.append(test_batch[1][0])
			data_Label.append(int(result[0]))
		data_result={'ImageId':data_Id,'Label':data_Label}
		df = pd.DataFrame(data_result, columns= ['ImageId', 'Label'])
		export_csv = df.to_csv (model_folder+'result.csv', index = None, header=True)
		sess.close()
		print('Done',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))




def data_preparer(path):
	'''prepare data before training'''
	'''test set processing'''
	test = pd.read_csv(path+"test.csv")
	test_x=np.array(test).tolist()
	TestSet=[]
	#print(test_x[0])

	for i in range(len(test_x)):
		TestSet.append({'x':test_x[i],'id':i+1})
	'''test set processing'''
	return TestSet


def next_feed(num,dataArr):
  '''pop the next feed in batch size number'''
  iArr=[]
  idArr=[]
  for i in range(num):
    p=dataArr.pop(0)
    dataArr.append(p)
    iArr.append(p['x'])
    idArr.append(p['id'])
  return [iArr,idArr]


if __name__ == '__main__':
	#data_preparer('../inputs/')
	main()