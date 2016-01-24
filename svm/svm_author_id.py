#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(kernel='rbf', C=10000.0)

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print("training time: {:.3f}s".format(time()-t0))

t0 = time()
pred = clf.predict(features_test)
print("prediction time: {:.3f}s".format(time()-t0))

count_chris = 0
for result in pred:
	if result == 1:
		count_chris += 1

print("prediction for Chris (1) Class: {}".format(count_chris))

accuracy_score = accuracy_score(labels_test, pred)
print(accuracy_score)

#########################################################


