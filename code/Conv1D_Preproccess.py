import numpy as np
import pandas as pd
from scipy import signal

image=np.array([[1,2,3],[4,5,6],[7,8,9]])
mask=np.array([[1/4,1/4],[1/4,1/4]])
re=signal.convolve2d(image,mask,boundary='symm',mode='valid')
# print(re)


def conv(il,m_s,i_s):
	length=len(il)
	il=il.reshape(length,i_s,i_s).tolist()
	masklist=[]
	for i in range(m_s):
		masklist.append([])
		for j in range(m_s):
			masklist[i].append(1/m_s**2)
	mask=np.array(masklist)
	for i in range(length):
		a=il.pop(0)
		a=signal.convolve2d(a,mask,boundary='symm',mode='valid')
		il.append(a)
	il=np.array(il).reshape(length,(i_s-2)**2)
	return il

def data_preparer(path,train_perc):
	'''prepare data before training'''
	'''train set processing'''

	train = pd.read_csv(path+"train.csv").sample(frac=1)
	label=train.pop('label') #target
	train_x=np.array(train)
	train_y=np.array(label)
	i=28
	while i>2:
		train_x=conv(train_x,3,i)
		print(train_x.shape)
		i=i-2
		print(i)
	#print(train_x.shape)

	# #print(len(train_x))
	# divide=int(len(train_x)*train_perc)
	# TrainSet=[train_x[0:divide],train_y[0:divide]]
	# if train_perc==1.0:
	# 	divide=int(len(train_x)*0.7)
	# ValSet=[train_x[divide:-1],train_y[divide:-1]]

	# test = pd.read_csv(path+"test.csv")
	# test = ss.transform(test)
	# test = pca.transform(test)
	# test_x=np.array(test).tolist()
	# id_list=[]
	# for i in range(len(test_x)):
	# 	id_list.append(i+1)
	# TestSet=[test_x,id_list]
	# '''test set processing'''

	return TrainSet,ValSet,TestSet

if __name__ == '__main__':
	data_preparer('../input/',1.0)
	#main()