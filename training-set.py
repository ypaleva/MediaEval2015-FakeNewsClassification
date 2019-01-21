import codecs

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PATH_TRAINING = "/home/yoanapaleva/Desktop/mediaeval-2015-trainingset.csv"
PATH_TEST = "/home/yoanapaleva/Desktop/mediaeval-2015-testset.csv"

docTrain = codecs.open(PATH_TRAINING, 'rU', 'UTF-8')
docTest = codecs.open(PATH_TEST, 'rU', 'UTF-8')
df_train = pd.read_csv(docTrain, sep='\t')
df_test = pd.read_csv(docTest, sep='\t')

print(df_train.groupby('label').size())


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

train, test, y_train, y_test = train_test_split(tweets_train, y_train, test_size=0.3,
                                                random_state=1000)

print('Train size: ', train.size)
print('Test size: ', test.size)
vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(train)

X_train = vectorizer.transform(train)
X_test = vectorizer.transform(test)

target_names = ['real', 'fake']

print('------------------LOGISTIC REGRESSION-------------------------')

log = LogisticRegression()
log.fit(X_train, y_train)
log_predicted = log.predict(X_test)
print('F1-score log: ', f1_score(y_test, log_predicted))
print(classification_report(y_test, log_predicted, target_names=target_names))

print('------------------NAIVE BAYES-------------------------')
bayes = GaussianNB()
bayes.fit(X_train.toarray(), y_train)
bayes_predicted = bayes.predict(X_test.toarray())
print('F1-score bayes: ', f1_score(y_test, bayes_predicted))
print(classification_report(y_test, bayes_predicted, target_names=target_names))

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

print('------------------RANDOM FOREST-------------------------')

forest = RandomForestClassifier(n_estimators=100, max_depth=10)
forest.fit(X_train, y_train)
forest_predicted = forest.predict(X_test)
print('F1-score forest: ', f1_score(y_test, forest_predicted))
print(classification_report(y_test, forest_predicted, target_names=target_names))

print('------------------ADA BOOST-------------------------')

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=100)
ada.fit(X_train, y_train)
ada_predicted = ada.predict(X_test)
print('F1-score ada: ', f1_score(y_test, ada_predicted))
print(classification_report(y_test, ada_predicted, target_names=target_names))

print('------------------PERCEPTRON-------------------------')

mlp = MLPClassifier(activation='relu', early_stopping=False, learning_rate='constant', hidden_layer_sizes=(5, 2),
                    learning_rate_init=0.001)
mlp.fit(X_train, y_train)
mlp_predicted = mlp.predict(X_test)
print('F1-score mlp: ', f1_score(y_test, mlp_predicted))
print(classification_report(y_test, mlp_predicted, target_names=target_names))
