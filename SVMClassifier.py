import glob
import numpy
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("corpus.csv", sep=",", encoding="latin-1")

df = df.set_index('id')
df.columns = ['class', 'text']

data = df.reindex(numpy.random.permutation(df.index))



pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('tfidf',              TfidfTransformer()),
    ('classifier',         OneVsRestClassifier(LinearSVC()))
])

k_fold = KFold(n=len(data), n_folds=6, shuffle=True)

for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    #Enter unseen data here
    #files = glob.glob("corpus/*.txt")
    #lines = []
    #for fle in files:
    #    with open(fle) as f:
    #        lines += f.readlines()        
    #test_text = numpy.array(lines)
    #################################

    lb = LabelBinarizer()
    Z = lb.fit_transform(train_y)

    pipeline.fit(train_text, Z)
    predicted = pipeline.predict(test_text)
    predictions = lb.inverse_transform(predicted)
    

    df2=pd.DataFrame(predictions)
    df2.index+=1
    df2.index.name='Id'
    df2.columns=['Label']
    df2.to_csv('results.csv',header=True)
    for item, labels in zip(test_text, predictions):
        print('Item: {0} => Label: {1}'.format(item, labels))    
   

    cm = confusion_matrix(test_y, predictions)
    accuracy = accuracy_score(test_y, predictions)

print 'The resulting accuracy using Linear SVC is ', (100 * accuracy), '%\n'

percentage_matrix = 100 * cm / cm.sum(axis=1).astype(float)
plt.figure(figsize=(16, 16))
sns.heatmap(percentage_matrix, annot=True,  fmt='.2f', xticklabels=['Java', 'Python', 'Scala'], yticklabels=['Java', 'Python', 'Scala']);
plt.title('Confusion Matrix (Percentage)');
plt.show()
print(classification_report(test_y, predictions,target_names=['Java', 'Python', 'Scala'], digits=2))
