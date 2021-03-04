from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss


train = open('./generate_dataset/train_category_L6.txt', 'r').read().split('\n')
test = open('./generate_dataset/test_category_L6.txt', 'r').read().split('\n')

X_train = []
y_train = []

for i, sample in enumerate(train):
    if i % 1000 == 999:
        break
    query_lables = sample.split('\t')
    query = query_lables[0]
    labels = query_lables[1:]
    X_train.append(query)
    y_train.append(labels)

X_test = []
y_test = []
for i,sample in enumerate(test):
    if i % 1000 == 50:
        break

    query_lables = sample.split('\t')
    query = query_lables[0]
    labels = query_lables[1:]
    X_test.append(query)
    y_test.append(labels)

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2', max_features = 20000 )
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

y_test = np.array(y_test)
y_train = np.array(y_train)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit_transform(y_train)


 # transform target variable
y_test = multilabel_binarizer.transform(y_test)
y_train = multilabel_binarizer.transform(y_train)


classifier = BinaryRelevance(GaussianNB())
classifier.fit(X_train, y_train)
br_predictions = classifier.predict(X_test)
v = (br_predictions.toarray())
g = 0

# print("Accuracy = ",accuracy_score(y_test,br_predictions.toarray()))
# print("F1 score = ",f1_score(y_test,br_predictions, average="micro"))
# print("Hamming loss = ",hamming_loss(y_test,br_predictions))

