import pylab as pl
import matplotlib.pyplot as plt
import json

def dataReader_train(path):
	cycle=10000#1000
	divisor=400#100
	accu_step=[]
	accu_loss=[]
	accu_prob=[]
	accu_val_l=0
	accu_val_p=0
	for line in open(path):
		jsonBody=json.loads(line)
		l=jsonBody['loss']
		p=jsonBody['prob']	
		s=jsonBody['steps']
		if s!=0:
			accu_val_l=accu_val_l+l
			accu_val_p=accu_val_p+p
		if s %cycle ==0 and s!=0:
			accu_step.append(s)
			accu_loss.append(accu_val_l/divisor)
			accu_prob.append(accu_val_p/divisor)
			accu_val_l=0
			accu_val_p=0

	return accu_step,accu_loss,accu_prob


def dataReader_val(path):
	cycle=10000
	divisor=10
	accu_step=[]
	accu_loss=[]
	accu_prob=[]
	accu_val_l=0
	accu_val_p=0
	for line in open(path):
		jsonBody=json.loads(line)
		l=jsonBody['loss']
		p=jsonBody['prob']	
		s=jsonBody['steps']
		if s!=0:
			accu_val_l=accu_val_l+l
			accu_val_p=accu_val_p+p
		if s %cycle ==0 and s!=0 :
			accu_step.append(s)
			accu_loss.append(accu_val_l/divisor)
			accu_prob.append(accu_val_p/divisor)
			accu_val_l=0
			accu_val_p=0

	return 	accu_step,accu_loss,accu_prob

def dataReader_test(path):
	stepSet=[]
	lossSet=[]
	probSet=[]

	for line in open(path):
		jsonBody=json.loads(line)
		l=jsonBody['loss']
		p=jsonBody['prob']
		s=jsonBody['steps']
		probSet.append(p)
		lossSet.append(l)
		stepSet.append(s)

	return 	stepSet,lossSet,probSet



if __name__ == '__main__':

	# change your model folder here
	forder_name='../model/exp2/MNIST' 

	val_step,val_loss,val_prob=dataReader_val(forder_name+'_LossLog_val.txt')
	train_step,train_loss,train_prob=dataReader_train(forder_name+'_LossLog_train.txt')
	plt.figure(1)
	#loss
	pl.plot( train_step,train_loss,'c')# use pylab to plot x and y
	pl.plot( train_step,train_loss,'c*')# use pylab to plot x and y
	pl.plot( val_step,val_loss,'g')# use pylab to plot x and y
	pl.plot( val_step,val_loss,'g*')# use pylab to plot x and y
	pl.title('Loss (CE)')
	pl.ylabel('Loss (CE)')
	pl.xlabel('steps')

	#prob
	plt.figure(2)
	pl.plot( train_step,train_prob,'c')# use pylab to plot x and y
	pl.plot( train_step,train_prob,'c*')# use pylab to plot x and y
	pl.plot( val_step,val_prob,'g')# use pylab to plot x and y
	pl.plot( val_step,val_prob,'g*')# use pylab to plot x and y
	pl.title('Accuracy')
	pl.ylabel('Accuracy')
	pl.xlabel('steps')
		
	pl.show()# show the plot on the screen
