import pandas as pd
import numpy as np
import codecs
import re
import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from scipy.sparse import hstack
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

PATH_TRAINING = "/home/yoanapaleva/Desktop/mediaeval-2015-trainingset.csv"
PATH_TEST = "/home/yoanapaleva/Desktop/mediaeval-2015-testset.csv"

docTrain = codecs.open(PATH_TRAINING, 'rU', 'UTF-8')
docTest = codecs.open(PATH_TEST, 'rU', 'UTF-8')
df = pd.read_csv(docTrain, sep='\t')
dfTest = pd.read_csv(docTest, sep='\t')

print(df.head())
print(df.dtypes)
print('Size train: ', df['tweetText'].size)
print('Size test: ', dfTest['tweetText'].size)

stop_words = set(stopwords.words('english', 'portuguese'))
wordnet_lemmatizer = WordNetLemmatizer()

### Adapted from: https://www.kaggle.com/langkilde/linear-svm-classification-of-sentiment-in-tweets/code
def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

###

print("Normalizing training tweets...")
df['normalized_tweet'] = df.tweetText.apply(normalizer)
print("Normalizing test tweets...")
dfTest['normalized_tweet'] = dfTest.tweetText.apply(normalizer)

print(df[['tweetText', 'normalized_tweet']].head())
print('Value counts: ', df['label'].value_counts())

### Adapted from: https://www.kaggle.com/langkilde/linear-svm-classification-of-sentiment-in-tweets/code
def ngrams(input_list):
    # onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams + trigrams
###

print("Making n-grams from training tweets...")
df['grams'] = df.normalized_tweet.apply(ngrams)
print("Making n-grams from test tweets...")
dfTest['grams'] = dfTest.normalized_tweet.apply(ngrams)
print(df[['grams']].head())

### Adapted from: https://www.kaggle.com/langkilde/linear-svm-classification-of-sentiment-in-tweets/code
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorized_data_train = vectorizer.fit_transform(df.tweetText)
vectorized_data_test = vectorizer.transform(dfTest.tweetText)

indexed_data_train = hstack((np.array(range(0, vectorized_data_train.shape[0]))[:, None], vectorized_data_train))
indexed_data_test = hstack((np.array(range(0, vectorized_data_test.shape[0]))[:, None], vectorized_data_test))


def label2target(label):
    return {
        'fake': 1,
        'real': 0,
        'humor': 1
    }[label]
##############################################################################

targets_tr = df.label.apply(label2target)
targets_ts = dfTest.label.apply(label2target)

# gamma = 0.01 -> 0.901% (C = 100)
# gamma = 0.001 ->  0.904%  (C = 100)
# C = 50 (gamma = 0.001) -> 0.904%
# C = 20 -> 0.904% , c=5
clf = OneVsRestClassifier(svm.SVC(gamma=0.001, C=5, probability=True, class_weight='balanced', kernel='linear'))
# clf.fit(data_train, targets_train)
# predicted = clf.predict(data_test)

forest = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='auto')

# forest.fit(data_train, targets_train)
# predicted = forest.predict(data_test)

# kmeans = KMeans(n_clusters=4)

mlp = MLPClassifier(activation='relu', early_stopping=False, learning_rate='constant', hidden_layer_sizes=(5, 2),
                    learning_rate_init=0.001)

# gamma = auto -> 0.64
# gamma = scale -> 0.72
svm = SVC(gamma='scale', kernel='linear')

# max_depth = 5-> 0.69
# max_depth = 32 -> 0.73
# max_depth -> auto -> 0.87
# max_leaf = 30 -> 0.72
# max_leaf = 60 -> 0.74
tree = DecisionTreeClassifier(criterion='gini', max_features='auto')
# tree.fit(data_train, targets_train)
# predicted = tree.predict(data_test)


# no parameters 0.905
# solver='lbfgs' -> 0.903

log = LogisticRegression()

gnb = GaussianNB()

# n=200 -> 0.6781274795027771
# n=5 -> 0.6781274795027771
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         n_estimators=5)

ada.fit(indexed_data_train, targets_tr)

predicted = ada.predict(indexed_data_test)
print(f1_score(targets_ts, predicted, average='micro'))

