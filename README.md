[![DOI](https://zenodo.org/badge/94644625.svg)](https://zenodo.org/badge/latestdoi/94644625)


The package includes the following files:
	
-data_extraction.py: The data extraction script extracts the relevant data from the corpus and creates a CSV corpus.csv to store the data in a format that can be used by the system
-SVMClassifier.py: implements the Linear SVC SVM Classifier and returns a list of classified output labels for the test data
-corpus.txt: This is the corpus that is used for the project, this file can be stored at any desired path. When the data_extraction.py script is run, the system prompts the user to enter the path of the location where this file is stored)
-README.txt

Instructions:

1) Install the required libraries, the system needs the following import statements to import the required libraries:

import glob
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
from sklearn.calibration import (calibration_curve, CalibratedClassifierCV)

2) Save the corpus.txt file on disk

3) Run the data_extraction.py script and provide the required input (i.e. file path to the corpus.txt)

4) Run the SVMClassifier.py script and provide the required input

5) A heatmap of the confusion matrix (percentage), accuracy score and classification report are printed

	