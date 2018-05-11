import pandas as pd
import numpy as np
import math
# peason 
def Peason(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    x = x - x.mean()
    y = y - y.mean()
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
# Jaccard
def Jaccard(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    a = x.dot(y)
    b = np.linalg.norm(x)
    c = np.linalg.norm(y)
    return a/(np.power(b,2)+np.power(c,2)-a)
# cosine
def Cosine(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
# normalize the data
def standard_vector(Array):
	zscore = lambda x: (x-x.mean())/x.std()
	maxmin = lambda x: (x - x.min())/(x.max() - x.min())
	ArrT = np.zeros(Array.shape)
	if len(Array.shape)<2:
		# 0-10 stage
		ArrT = maxmin(Array).reshape(len(Array),1)
	else:
		length = Array.shape[1]
		for k in range(length):
			if k == 8:
				out = Array[:,k]
				ArrT[:,k] = out/14
			else:
				out = Array[:,k]
				ArrT[:,k] = zscore(out)
	return ArrT
# prepare the dataset
class DataSet:
	def __init__(self,root,train = True,transform = None,target_transform = None,percentage = 0.8):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.train = train
		self.percentage = percentage

		if self.train:
			self.train_data,self.train_target = self.load_data()
			if self.transform:
				self.train_data,self.train_target = self.transform(self.train_data),self.target_transform(self.train_target)
		else:
			self.test_data,self.test_target = self.load_data()
			if self.transform:
				self.test_data,self.test_target = self.transform(self.test_data),self.target_transform(self.test_target)
	def __getitem__(self,index):
		if self.train:
			return self.train_data[index],self.train_target[index]
		else:
			return self.test_data[index],self.test_target[index]
	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)
	def __repr__(self):
		str_all = ""
		if self.train:
			str_all += "train data:\n"
			str_all += str(self.train_data)
			str_all += "\n"
			str_all += str(self.train_target)
		else:
			str_all += "test data:\n"
			str_all += str(self.test_data)
			str_all += "\ntest target:\n"
			str_all += str(self.test_target)
		return str_all
	def get_data(self):
		if self.train:
			return self.train_data,self.train_target
		else:
			return self.test_data,self.test_target
	def load_data(self):
		DataFrame = pd.read_csv(self.root,sep = ";")
		Length = len(DataFrame)
		self.keys = list(DataFrame.columns)
		DataFrame = DataFrame.as_matrix()
		train_length = math.floor(self.percentage*Length)
		if self.train:
			train_data,train_target = DataFrame[:train_length,:len(self.keys) - 2],DataFrame[:train_length,-1]
			return train_data,train_target
		else:
			test_data,test_target = DataFrame[train_length:,:len(self.keys) - 2],DataFrame[train_length:,-1]
			return test_data,test_target
# data loader processing
class DataLoader:
	def __init__(self,dataset,batch_size):
		self.dataset = dataset
		self.start = 0
		self.k_numbers = 0
		self.batch_size = batch_size
		self.length = len(self.dataset)
	def __iter__(self):
		return self
	def __next__(self):
		if self.k_numbers >self.batch_size:
			raise StopIteration
		else:
			start = self.start + self.k_numbers * self.batch_size 
			end   = self.start + (self.k_numbers + 1) * self.batch_size
			self.k_numbers += 10
			return self.dataset[start:end]
	def clear(self):
		self.start = 0
		self.k_numbers = 0
def test_dataset():
	file_path = "/home/asus/py3env/project/project/data/winequality-red.csv"
	dataset = DataSet(file_path,train = True,transform = standard_vector,target_transform = standard_vector)
	dataloader = DataLoader(dataset,10)
	print("length:",len(dataset))
	for k,(train,target) in enumerate(dataloader):
		print(k,train,target)
		exit()
	print(dataloader.k_numbers)
if __name__ == '__main__':
	test_dataset()