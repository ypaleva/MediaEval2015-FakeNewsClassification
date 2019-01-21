import codecs
import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

PATH_TRAINING = "/home/yoanapaleva/Desktop/mediaeval-2015-trainingset.csv"
PATH_TEST = "/home/yoanapaleva/Desktop/mediaeval-2015-testset.csv"

docTrain = codecs.open(PATH_TRAINING, 'rU', 'UTF-8')
docTest = codecs.open(PATH_TEST, 'rU', 'UTF-8')
df_train = pd.read_csv(docTrain, sep='\t')
df_test = pd.read_csv(docTest, sep='\t')


def label2target(label):
    return {
        'fake': 1,
        'real': 0,
        'humor': 1
    }[label]


y_train = df_train.label.apply(label2target)
y_test = df_test.label.apply(label2target)

tweets_train = df_train.tweetText
tweets_test = df_test.tweetText

vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(tweets_train)

X_train = vectorizer.transform(tweets_train)
X_test = vectorizer.transform(tweets_test)

target_names = ['real', 'fake']

log = LogisticRegression(C=3593.81, penalty='l1')

# From : https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/
# penalty = ['l1', 'l2']
# C = np.logspace(0, 4, 10)
# hyperparameters = dict(C=C, penalty=penalty)
# clf = GridSearchCV(log, hyperparameters, cv=5, verbose=0)
#
# best_model = clf.fit(X_train, y_train)
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])

# svm = SVC()
# params = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001], 'kernel': ['linear', 'rbf']}
# grid = GridSearchCV(svm, params, refit=True, verbose=2)
# grid.fit(X_train, y_train)
#
# best_model = grid.fit(X_train, y_train)
# print('Best Kernel:', best_model.best_estimator_.get_params()['kernel'])
# print('Best Gamma:', best_model.best_estimator_.get_params()['gamma'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])


# log.fit(X_train, y_train)
# log_predicted = log.predict(X_test)
# print('F1-score log: ', f1_score(y_test, log_predicted))
# print(classification_report(y_test, log_predicted, target_names=target_names))
#
# print('Confusion matrix: ', confusion_matrix(y_test, log_predicted))

# probabilities = log.predict_proba(X_test)
# predictions = probabilities[:, 1]
# fpr, tpr, threshold = roc_curve(y_test, predictions)
# roc_auc = metrics.auc(fpr, tpr)
#
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

scores_10 = cross_val_score(log, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_10.mean(), scores_10.std() * 2))
print(scores_10.mean())
print(scores_10.std())
print(math.pow(scores_10.std(), 2))
