import mxnet as mx
import logging, sys
import numpy as np

def load_data(file_name, b_with_label, shape):
	file = open(file_name, 'r')
	lines = file.readlines()
	file.close()

	lines = lines[1:]
	row_count = len(lines)
	col_count = len(lines[0].split(','))
	offset = 1 if b_with_label else 0	
	dims = [row_count]
	dims += shape
	features = np.zeros(dims)
	labels = np.zeros(row_count)
	for i in range(row_count):
		items = lines[i].strip().split(',')
		features[i] = np.array(map(lambda x : 1 if x > 0 else 0, np.array(items[offset:], dtype=np.float))).reshape(shape)
		if b_with_label:
			labels[i] = int(items[0])

	if b_with_label:
		return features, labels
	else:
		return features

# define mlp
def get_mlp():
	data = mx.symbol.Variable('data')
	fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
	act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
	fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
	act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
	fc3 = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 32)
	act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")	
	fc4 = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=10)
	mlp = mx.symbol.SoftmaxOutput(data = fc4, name = 'softmax')

	return mlp

# def lenet
def get_lenet():
	data = mx.symbol.Variable('data')
	# first conv
	conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
	tanh1 = mx.symbol.Activation(data=conv1, act_type="relu")
	pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
	                          kernel=(2,2), stride=(2,2))
	# second conv
	conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
	tanh2 = mx.symbol.Activation(data=conv2, act_type="relu")
	pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
	                          kernel=(2,2), stride=(2,2))
	# first fullc
	flatten = mx.symbol.Flatten(data=pool2)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
	tanh3 = mx.symbol.Activation(data=fc1, act_type="relu")
	# second fullc
	fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
	# loss
	lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')	

	return lenet

# def alexnet
def get_alexnet():
	input_data = mx.symbol.Variable(name="data")
	# stage 1
	conv1 = mx.symbol.Convolution(
	    data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
	relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
	pool1 = mx.symbol.Pooling(
	    data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
	lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
	# stage 2
	conv2 = mx.symbol.Convolution(
	    data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
	relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
	pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
	lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
	# stage 3
	conv3 = mx.symbol.Convolution(
	    data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
	relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
	conv4 = mx.symbol.Convolution(
	    data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
	relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
	conv5 = mx.symbol.Convolution(
	    data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
	relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
	pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
	# stage 4
	flatten = mx.symbol.Flatten(data=pool3)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
	relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
	dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
	# stage 5
	fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096)
	relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
	dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
	# stage 6
	fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=10)
	softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')	

	return softmax

train_file = "../data/mnist/train.csv"
test_file = "../data/mnist/test.csv"

(train_data, train_label) = load_data(train_file, True, shape = [1, 28, 28])

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx = mx.gpu(), symbol = get_lenet(), num_epoch = 50,
    learning_rate = 0.05, momentum = 0.9, wd = 0.00001)

#data_offset = 35000
#model.fit(X=train_data[:data_offset, :], y=train_label[:data_offset], 
#	eval_data=(train_data[data_offset: , :], train_label[data_offset: ]),
#	batch_end_callback=mx.callback.Speedometer(100))

model.fit(X=train_data, y=train_label, batch_end_callback=mx.callback.Speedometer(100))

test_data = load_data(test_file, False, shape = [1, 28, 28])
probs = model.predict(test_data)

file = open("prdict_labels.csv", 'w')
file.write("ImageId,Label\n")
test_sample_count = probs.shape[0]
for image_id in range(test_sample_count):
	file.write(str(image_id+1) + "," + str(int(probs[image_id].argmax())) + '\n')

file.close()




