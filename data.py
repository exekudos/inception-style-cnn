import numpy
import csv

train_val, train_label, test_val, test_label=[],[],[],[]
def load():
	filename='fer2013.csv'
	raw=open(filename)
	data=csv.reader(raw)
	train_label,train_val,test_label,test_val=[],[],[],[]
	i=0
	for row in data:
		if i<28709:
			train_label.append(row[0])
			train_val.append(list(row[1].split()))
		elif i<35887:
			test_label.append(row[0])
			test_val.append(list(row[1].split()))
		i+=1
		if i>=35887:
			break
	train_label=numpy.array(train_label)
	train_val=numpy.array(train_val)
	test_label=numpy.array(test_label)
	test_val=numpy.array(test_val)
