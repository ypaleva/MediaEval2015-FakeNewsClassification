import codecs
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

print('------------------LOGISTIC REGRESSION-------------------------')

log = LogisticRegression()
log.fit(X_train, y_train)
log_predicted = log.predict(X_test)
print('F1-score log: ', f1_score(y_test, log_predicted))
print(classification_report(y_test, log_predicted, target_names=target_names))

print('------------------SVM-------------------------')
svm = SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear')
svm.fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
print('F1-score svm: ', f1_score(y_test, svm_predicted))
print(classification_report(y_test, svm_predicted, target_names=target_names))

print('------------------DECISION TREE-------------------------')

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_predicted = tree.predict(X_test)
print('F1-score tree: ', f1_score(y_test, tree_predicted))
print(classification_report(y_test, tree_predicted, target_names=target_names))
