"""William Cukierski's tutorial on SVM.
"""
import numpy as np
from sklearn import svm

train_file = "./data/train.csv"
labels_file = "./data/trainLabels.csv"
test_file = "./data/test.csv"

# Read the data
train = np.loadtxt(open(train_file,"rb"), delimiter=",", skiprows=0)
trainLabels = np.loadtxt(open(labels_file,"rb"), delimiter=",", skiprows=0)
test = np.loadtxt(open(test_file,"rb"), delimiter=",", skiprows=0)

# Train a linear SVM
clf = svm.LinearSVC()
clf.fit(train, trainLabels)
predictions = clf.predict(test)
np.savetxt(
    "linearSVMSubmission.csv",
    predictions.astype(int),
    fmt='%d',
    delimiter=",")

# Train a fancy SVM
clf = svm.SVC()
clf.fit(train, trainLabels)
predictions = clf.predict(test)
np.savetxt(
    "fancySVMSubmission.csv",
    predictions.astype(int),
    fmt='%d',
    delimiter=",")
