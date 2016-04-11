#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import neighbors
from sklearn.decomposition import PCA
import sys, os

def load_data(file_name, b_with_label):
	file = open(file_name, 'r')
	lines = file.readlines()
	file.close()

	lines = lines[1:]
	row_count = len(lines)
	col_count = len(lines[0].split(','))
	offset = 1 if b_with_label else 0	
	features = np.zeros((row_count, col_count - offset))
	labels = np.zeros(row_count)
	for i in range(row_count):
		items = lines[i].strip().split(',')
		features[i] = map(lambda x : 1 if x > 0 else 0, np.array(items[offset:], dtype=np.float))
		if b_with_label:
			labels[i] = int(items[0])

	if b_with_label:
		return features, labels
	else:
		return features

def clf_model(train_data, train_label, test_data, model_type):
	print "Train samples : ", train_data.shape
	print "Test samples : ", test_data.shape
	if model_type == "svm":
		return svm_model(train_data, train_label, test_data)
	elif model_type == "rf":
		return rf_model(train_data, train_label, test_data)
	elif model_type == "gbdt":
		return gbdt_model(train_data, train_label, test_data)
	elif model_type == "knn":
		return knn_model(train_data, train_label, test_data)

def svm_model(train_data, train_label, test_data):
	print "Model : PCA ==> SVM"
	pca = PCA(n_components = 100, copy = True)
	clf = svm.SVC()
	clf.fit(pca.fit_transform(train_data), train_label)
	return clf.predict(pca.transform(test_data))

def rf_model(train_data, train_label, test_data):
	print "Model rf"
	clf = ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = 10, min_samples_split = 5)
	pca = PCA(n_components = 100, copy = True)
	#clf.fit(pca.fit_transform(train_data), train_label)
	#return clf.predict(pca.transform(test_data))
	clf.fit(train_data, train_label)
	return clf.predict(test_data)

def gbdt_model(train_data, train_label, test_data):
	print "Model gbdt"
	clf = ensemble.GradientBoostingClassifier()
	pca = PCA(n_components = 100, copy = True)
	clf.fit(pca.fit_transform(train_data), train_label)
	return clf.predict(pca.transform(test_data))

def knn_model(train_data, train_label, test_data):
	print "Model KNN"
	clf = neighbors.KNeighborsClassifier(algorithm = "kd_tree")
	pca = PCA(n_components = 200, copy = True)
	clf.fit(pca.fit_transform(train_data), train_label)
	return clf.predict(pca.transform(test_data))

train_file = "train.csv"
test_file = "test.csv"

(train_features, labels) = load_data(train_file, True)
if True:
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, labels, test_size=0.3)

	pred_labels = clf_model(X_train, y_train, X_test, "knn")
	count = 0
	test_sample_count = y_test.shape[0]
	for i in range(test_sample_count):
		count += 1 if (y_test[i] == pred_labels[i]) else 0
	print 'Precision : ', 1.0 * count / test_sample_count	
	sys.exit()

test_features = load_data(test_file, False)
pred_labels = clf_model(train_features, labels, test_features, "svm")

file = open("prdict_labels.csv", 'w')
file.write("ImageId,Label\n")
test_sample_count = pred_labels.shape[0]
for image_id in range(test_sample_count):
	file.write(str(image_id+1) + "," + str(int(pred_labels[image_id])) + '\n')

file.close()




