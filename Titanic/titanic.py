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
	profile_list = []
	labels = np.zeros(row_count)
	pid_list = []
	for i in range(row_count):
		items = lines[i].strip().split(',')
		pid_list.append(items[0])
		if b_with_label:
			labels[i] = int(items[1])
		profile_list.append(items[1 + offset:])

	if b_with_label:
		return profile_list, pid_list, labels
	else:
		return profile_list, pid_list

def fg(profiles):
	#Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	sample_count = len(profiles)
	featurs = np.zeros((sample_count, 12), dtype=np.float)
	for i in range(sample_count):
		items = profiles[i]
		featurs[i][0] = 1.0 if items[0] == "1" else 0
		featurs[i][1] = 1.0 if items[0] == "2" else 0
		featurs[i][2] = 1.0 if items[0] == "3" else 0
		featurs[i][3] = 1.0 if items[3] == "female" else 0.0
		featurs[i][4] = 1.0 if items[3] == 'male' else 0.0
		featurs[i][5] = int(float(items[4]) / 10.0) if len(items[4]) > 0 else 0.0
		featurs[i][6] = 1.0 if len(items[5]) > 0 and int(items[5]) > 0 else 0.0
		featurs[i][6] = 1.0 if len(items[6]) > 0 and int(items[6]) > 0 else 0.0
		featurs[i][8] = int(float(items[8]) / 5.0) if len(items[8]) > 0 else 0.0
		featurs[i][9] = 1.0 if items[10] == "C" else 0.0
		featurs[i][10] = 1.0 if items[10] == "Q" else 0.0
		featurs[i][11] = 1.0 if items[10] == "S" else 0.0
		#featurs[i][(int(float(items[4]) / 10.0) if len(items[4]) > 0 else 0.0) + 8 * (1.0 if items[3] == "female" else 0.0)] = 1.0
	return featurs

def vc_model(train_features, train_labels, model):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, labels, test_size=0.2)
	if model == "lr":
		clf = linear_model.LogisticRegression(C=2.0)
	elif model == "gbdt":
		clf = ensemble.GradientBoostingClassifier(n_estimators = 1000, random_state=312, min_samples_leaf=3)
	elif model == "svm":
		clf = svm.SVC()
	elif model == "rf":
		clf = ensemble.RandomForestClassifier(n_estimators = 10000, random_state=312, min_samples_leaf=3)
	else:
		print 'Invalid model name...'
		return
	clf.fit(X_train, y_train)

	res = clf.predict(X_test)
	precision = 0
	test_sample = len(X_test)
	for i in range(test_sample):
		precision += 1 if res[i] == y_test[i] else 0
	print 'Presicion : ', 100.0 * precision / test_sample	

def predict_test(train_features, train_labels, test_features, model):
	if model == "lr":
		clf = linear_model.LogisticRegression(C=2.0)
	elif model == "gbdt":
		clf = ensemble.GradientBoostingClassifier()
	elif model == "svm":
		clf = svm.SVC()
	elif model == "rf":
		clf = ensemble.RandomForestClassifier(n_estimators = 10000, random_state=312, min_samples_leaf=3)
	clf.fit(train_features, train_labels)

	return clf.predict(test_features)	

def print_test_res(pid_list, pred_labels):
	file = open("predict_labels.csv", 'w')
	file.write("PassengerId,Survived\n")
	test_sample_count = len(pid_list)
	for i in range(test_sample_count):
		file.write(pid_list[i] + "," + str(int(pred_labels[i])) + '\n')

train_file = "train.csv"
test_file = "test.csv"

(train_profile, pid_list, labels) = load_data(train_file, True)
train_features = fg(train_profile)
vc_model(train_features, labels, "rf")

if True:
	(test_profile, pid_test) = load_data(test_file, False)
	test_features = fg(test_profile)
	print_test_res(pid_test, predict_test(train_features, labels, test_features, "rf"))








