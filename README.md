The package includes the following files:
	
	data_extraction.py: The data extraction script extracts the relevant data from the corpus and creates a CSV corpus.csv to store the data in a format that can be used by the system
	SVMClassifier.py
	corpus.txt: This is the corpus that is used for the project, this file can be stored at any desired path. When the data_extraction.py script is run, the system prompts the user to enter the path of the location where this file is stored)
	README.txt

Instructions:

1) Install the required libraries, the system needs the following import statements to import the required libraries:

	import csv
	import time
	import glob
	import numpy as np
	from sklearn.preprocessing import LabelBinarizer
	from sklearn.pipeline import Pipeline
	from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import LinearSVC
	from sklearn.metrics import (confusion_matrix, classification_report)
	import matplotlib.pyplot as plt
	import seaborn as sn

2) Save the corpus.txt file on disk

3) Run the data_extraction.py script and provide the required input (i.e. file path to the corpus.txt)

4) Run the SVMClassifier.py script and provide the required input (i.e. number of examples to be used for training)

5) A heatmap of the confusion matrix (percentage) and classification report are printed


Methods:

1)__init__(self)
The init method is used to initialize various class variables.
It also reads the data from corpus.csv and shuffles the corpus to produce a random training and test data split based on the training sample size specified by the user


2)SVM_LinearSVC(self)
implements the Linear SVC SVM Classifier and returns a list of classified output labels for the test data

Important Variables:

X_train = examples used for training (1 to train_ex from shuffled corpus)
Y1 	= labels corresponding to samples in X_train 
Y_train = labels corresponding to X_train + X_test
X_test 	= examples used for testing (train_ex to size)
y 	= Transformed output from label binarizer (lb)	