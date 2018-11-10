import numpy as np 
import pandas as pd
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
def main():

	'''Management Data'''
	data_path='../input/'
	model_folder='../model/RF/'
	model_Name='titanic.ckpt'
	'''Management Data'''
	'''Data prepare, rotate or shuffle'''
	TrainSet,ValSet,TestSet=data_preparer(data_path,0.5)
	clf = ensemble.RandomForestClassifier(n_estimators=30,min_samples_leaf=1,max_depth=40,max_features='auto',oob_score=True)
	clf.fit(TrainSet[0],TrainSet[1])
	val_x=clf.predict(ValSet[0])
	print(validation_accu(val_x,ValSet[1]))
	data_PassengerId=TestSet[1]
	# print(TestSet[0])
	data_Label=clf.predict(TestSet[0])
	data_result={'ImageId':TestSet[1],'Label':data_Label}
	df = pd.DataFrame(data_result, columns= ['ImageId', 'Label'])
	export_csv = df.to_csv (model_folder+'result.csv', index = None, header=True)

def validation_accu(x,y):
	true=0
	for i in range(len(x)):
		if x[i]==y[i]:
			true=true+1
	return true/len(x)

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
	pca = PCA(n_components=40)
	ss = MinMaxScaler()

	train = pd.read_csv(path+"train.csv").sample(frac=1)
	label=train.pop('label') #target
	train_x=np.array(train)
	train_x = ss.transform(train_x)
	train_x = pca.fit_transform(train_x)	
	trian_x=train_x.tolist()
	train_y=np.array(label).tolist()
	#print(len(train_x))
	divide=int(len(train_x)*train_perc)
	TrainSet=[train_x[0:divide],train_y[0:divide]]
	if train_perc==1.0:
		divide=int(len(train_x)*0.7)
	ValSet=[train_x[divide:-1],train_y[divide:-1]]

	test = pd.read_csv(path+"test.csv")
	test_x=np.array(test_x)
	test_x = ss.transform(test_x)
	test_x = pca.transform(test_x)	
	test_x=np.array(test_x).tolist()
	id_list=[]
	for i in range(len(test_x)):
		id_list.append(i+1)
	TestSet=[test_x,id_list]
	'''test set processing'''

	return TrainSet,ValSet,TestSet

if __name__ == '__main__':
	#data_preparer('../input/',1.0)
	main()