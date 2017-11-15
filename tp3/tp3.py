import numpy as np

from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


import re
import string

Xraw = []
yraw = []
with open("SMSSpamCollection") as f:
  for row in f.readlines():
    row = row.strip('\n').lower()
    tokens = row.split('\t')
    tokens[1] = re.sub(r"[" + string.punctuation + "]","" , tokens[1])
    # OU
    # tokens[1] = "".join([l for l in tokens[1] if l not in string.punctuation])
    Xraw.append(tokens[1])
    yraw.append(1 if tokens[0] == "spam" else 0)

tfidf_model = TfidfVectorizer(min_df= 5, max_df =0.5, stop_words = "english")
X= tfidf_model.fit_transform(Xraw)
y= np.array(yraw)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

print("X_train -- Nombre de ligne: %d, colonnes: %d" % X_train.shape)
print("y_train -- Nombre de ligne: %d, colonnes: 1" % y_train.shape)
print("X_test -- Nombre de ligne: %d, colonnes: %d" % X_test.shape)
print("y_test -- Nombre de ligne: %d, colonnes: 1" % y_test.shape)

model = KNeighborsClassifier(n_neighbors = 1)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
prediction_proba = model.predict_proba(X_test)

print(accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

k_fold = StratifiedKFold(n_splits = 10)
k_fold.get_n_splits(X_train, y_train)
for index_train, index_test in k_fold.split(X_train, y_train):
  x_train_fold = X_train[index_train, :]

