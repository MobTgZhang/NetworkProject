import numpy as np
import theano
import pickle
import theano.tensor as T
import matplotlib.pyplot as plt
import numpy as np
from model import BpNet,SvmMlp,RbmMlp,RBM,DBNMlp
from utils import DataSet,DataLoader,standard_vector
from utils import Peason,Jaccard,Cosine
def test_BpNet(dataset):
	test_data,target = dataset.get_data()
	f = open("bpnet.pt", 'rb')  
	bpmodel= pickle.load(f)  
	f.close()
	x = T.dmatrix("x")
	prediction = bpmodel.forward(x)
	predict = theano.function(inputs = [x],outputs = prediction)
	pre_data = predict(test_data)
	l1 = Peason(pre_data,target)
	print("peason:%f "%l1)
	l2 = Jaccard(pre_data,target)
	print("Jaccard:%f "%l2)
	l3 = Cosine(pre_data,target)
	print("Cosine:%f "%l3)
def test_SvmMlp(dataset):
	test_data,target = dataset.get_data()
	f = open("SvmMlp.pt", 'rb')  
	bpmodel= pickle.load(f)  
	f.close()
	x = T.dmatrix("x")
	prediction = bpmodel.forward(x)
	predict = theano.function(inputs = [x],outputs = prediction)
	pre_data = predict(test_data)
	l1 = Peason(pre_data,target)
	print("peason:%f "%l1)
	l2 = Jaccard(pre_data,target)
	print("Jaccard:%f "%l2)
	l3 = Cosine(pre_data,target)
	print("Cosine:%f "%l3)
def test_DBNMlp(dataset):
	test_data,target = dataset.get_data()
	f = open("DBNMlp.pt", 'rb')  
	bpmodel= pickle.load(f)  
	f.close()
	prediction = bpmodel.last_layer.output
	predict = theano.function(inputs = [bpmodel.x],outputs = prediction)
	pre_data = predict(test_data)
	l1 = Peason(pre_data,target)
	print("peason:%f "%l1)
	l2 = Jaccard(pre_data,target)
	print("Jaccard:%f "%l2)
	l3 = Cosine(pre_data,target)
	print("Cosine:%f "%l3)
def test_RbmSvm(dataset):
	test_data,target = dataset.get_data()
	f = open("RbmSvm.pt", 'rb')  
	bpmodel= pickle.load(f)
	f.close()
	x = T.dmatrix("x")
	bpmodel.rbm.reconstruct(test_data)
	prediction = bpmodel.svm.forward(x)
	predict = theano.function(inputs = [x],outputs = prediction)
	pre_data = predict(test_data)
	l1 = Peason(pre_data,target)
	print("peason:%f "%l1)
	l2 = Jaccard(pre_data,target)
	print("Jaccard:%f "%l2)
	l3 = Cosine(pre_data,target)
	print("Cosine:%f "%l3)
def trainDBNMlp(dataloader,params):
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")
	# defination of the layers 
	input_size = params["input_size"]
	output_size = params["output_size"]
	hidden_layers_sizes = params["hidden_size"]
	pretrain_times = params["pretrain_times"]
	training_epoches = params["training_epoches"]
	learning_rate = params["learning_rate"]

	numpy_rng = np.random.RandomState(1245)
	dbnmlp = DBNMlp(numpy_rng,theano_rng=None,n_ins=input_size,n_outs=output_size,hidden_layers_sizes=hidden_layers_sizes)
	prediction = dbnmlp.last_layer.output
	# the RBM costs all
	pretrain_func = dbnmlp.pretraining_functions()
	pre_train_loss = []
	for k in range(pretrain_times):
		length = len(dataloader.dataset)//dataloader.batch_size
		d = []
		for k,(train,target) in enumerate(dataloader):
			c = []
			for j in range(len(dbnmlp.rbm_layers)):
				loss = pretrain_func[j](train)
				c.append(loss)
			d.append(np.mean(c))
		dataloader.clear()
		loss = np.mean(d)
		print(loss)
		pre_train_loss.append(loss)
	pre_train_loss = np.array(pre_train_loss)
	x_list = np.linspace(0,len(pre_train_loss),len(pre_train_loss))
	plt.plot(x_list,pre_train_loss)
	plt.show()
	# than train the model
	trainDbnMlpFunc = dbnmlp.build_finetune_function(learning_rate)
	dbn_train_loss = []
	for k in range(training_epoches):
		length = len(dataloader.dataset)//dataloader.batch_size
		d = []
		for k,(train,target) in enumerate(dataloader):
			loss = trainDbnMlpFunc(train,target)
			d.append(loss)
		dataloader.clear()
		loss = np.mean(d)
		print(loss)
		dbn_train_loss.append(loss)
	dbn_train_loss = np.array(dbn_train_loss)
	x_list = np.linspace(0,len(dbn_train_loss),len(dbn_train_loss))
	plt.plot(x_list,dbn_train_loss)
	plt.show()
	plt.show()
	f= open('DBNMlp.pt', 'wb')
	pickle.dump(dbnmlp,f, protocol=pickle.HIGHEST_PROTOCOL)  
	f.close()
def train_BpNet(dataloader,params):
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")
	# defination of the layers 
	input_size = params["input_size"]
	hidden_size = params["hidden_size"]
	output_size = params["output_size"]
	learning_rate = params["learning_rate"]
	train_epoches = params["train_epoches"]
	np.random.RandomState(1234)
	net = BpNet(input_size,hidden_size,output_size)
	prediction = net.forward(x)
	# define the cost
	cost = T.mean(T.square(prediction - y))
	# update the grad 
	updates = net.update_grad(cost,learning_rate)
	# apply gradient descent
	train_func = theano.function(
		inputs = [x,y],
		outputs = [cost],
		updates = updates
	)
	# prediction 
	predict = theano.function(inputs = [x],outputs = prediction)
	# training model
	loss_plt = []
	for k in range(train_epoches):
		loss = 0
		length = len(dataloader.dataset)//dataloader.batch_size
		for k,(train,target) in enumerate(dataloader):
			err = train_func(train,target)
			loss += err[0]
		dataloader.clear()
		loss = loss/length
		loss_plt.append(loss)
	loss_plt = np.array(loss_plt)
	x_list = np.linspace(0,len(loss_plt),len(loss_plt))
	plt.plot(x_list,loss_plt)
	plt.show()
	f= open('bpnet.pt', 'wb')  
	pickle.dump(net,f, protocol=pickle.HIGHEST_PROTOCOL)  
	f.close()
def train_SvmMlp(dataloader,params):
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")
	# defination of the layers 
	input_size = params["input_size"]
	output_size = params["output_size"]
	learning_rate = params["learning_rate"]
	train_epoches = params["train_epoches"]
	net = SvmMlp(input_size,output_size)
	prediction = net.forward(x)
	# define the cost
	cost = T.mean(T.square(prediction - y))
	# update the grad 
	updates = net.updateGrad(cost,learning_rate)
	# apply gradient descent
	train_func = theano.function(
		inputs = [x,y],
		outputs = [cost],
		updates = updates
	)
	# prediction 
	predict = theano.function(inputs = [x],outputs = prediction)
	# training model
	loss_plt = []
	for k in range(train_epoches):
		loss = 0
		length = len(dataloader.dataset)//dataloader.batch_size
		for k,(train,target) in enumerate(dataloader):
			err = train_func(train,target)
			loss += err[0]
		dataloader.clear()
		loss = loss/length
		loss_plt.append(loss)
	loss_plt = np.array(loss_plt)
	x_list = np.linspace(0,len(loss_plt),len(loss_plt))
	plt.plot(x_list,loss_plt)
	plt.show()
	f= open('SvmMlp.pt', 'wb')  
	pickle.dump(net,f, protocol=pickle.HIGHEST_PROTOCOL)  
	f.close()
def train_RbmSvm(dataloader,params):
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")
	# defination of the layers 
	n_visible = params["input_size"]
	output_size = params["output_size"]
	learning_rate = params["learning_rate"]
	train_epoches = params["train_epoches"]
	net = RbmMlp(input=x,n_visible = n_visible,n_out = output_size,vbias=None,numpy_rng=None,theano_rng=None)
	#theano函数可以没有输出量，train_rbm实现RBM参数更新的功能
	train_rbm = net.pretraining_function()
	#SVM
	predict_svm = net.svm.forward(x)
	# define the cost
	cost_svm = T.mean(T.square(predict_svm - y))
	updates_grad = net.svm.updateGrad(cost_svm)
	train_svm = theano.function([x,y],cost_svm,updates = updates_grad,name = 'train_svm')
	# prediction 
	predict = theano.function(inputs = [x],outputs = predict_svm)
	# pretraining model
	loss_rbm= []
	for index in range(20):
		loss = 0
		length = len(dataloader.dataset)//dataloader.batch_size
		for k,(train,target) in enumerate(dataloader):
			err_rbm = train_rbm(train)
			# err_svm = train_svm(train,target)
			loss += err_rbm
			loss_rbm.append(err_rbm)
		dataloader.clear()
		loss = loss/length
		print(index,loss)
		loss_rbm.append(loss)
	loss_rbm = np.array(loss_rbm)
	x_list = np.linspace(0,len(loss_rbm),len(loss_rbm))
	plt.plot(x_list,loss_rbm)
	plt.show()
	# training model
	loss_svm= []
	for index in range(train_epoches):
		loss = 0
		length = len(dataloader.dataset)//dataloader.batch_size
		for k,(train,target) in enumerate(dataloader):
			# err_rbm = train_rbm(train)
			err_svm = train_svm(train,target)
			loss += err_svm
		dataloader.clear()
		loss = loss/length
		print(index,loss)
		loss_svm.append(loss)
	loss_svm = np.array(loss_svm)
	x_list = np.linspace(0,len(loss_svm),len(loss_svm))
	plt.plot(x_list,loss_svm)
	plt.show()
	f= open('RbmSvm.pt', 'wb')  
	pickle.dump(net,f, protocol=pickle.HIGHEST_PROTOCOL)  
	f.close()
if __name__ == '__main__':
	file_path = "/home/asus/py3env/project/project/data/winequality-red.csv"
	dataset = DataSet(file_path,train = True,transform = standard_vector,target_transform = standard_vector)
	dataloader = DataLoader(dataset,10)
	params = {
		"input_size":10,
		"hidden_size":[100,500],
		"output_size":1,
		"learning_rate":0.015,
		"training_epoches":120,
		"pretrain_times":7
	}
	# train_BpNet(dataloader,params)
	# train_SvmMlp(dataloader,params)
	# train_RbmSvm(dataloader,params)
	trainDBNMlp(dataloader,params)
	file_path = "/home/asus/py3env/project/project/data/winequality-red.csv"
	dataset = DataSet(file_path,train = False,transform = standard_vector,target_transform = standard_vector)
	# test_BpNet(dataset)
	# test_SvmMlp(dataset)
	# test_RbmSvm(dataset)
	test_DBNMlp(dataset)