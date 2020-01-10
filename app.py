import tensorflow as tf
import keras_applications, keras_preprocessing
import fasttext
import pandas as pf  # reading csv files
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder

# nltk.download('stopwords')
# nltk.download('punkt')

truthRating = {
    0: 'True',
    1: 'Mostly true',
    2: 'Half true',
    3: 'Mostly fake',
    4: 'Fake',
    5: 'Pants on Fire'
}

stop_words = set(stopwords.words('english'))


def sentiment_tokenize(data):
    data['Statement'] = data['Statement'].apply(lambda x: sent_tokenize(x))
    return data


def words_tokenize(data):
    data['Statement'] = data['Statement'].apply(lambda x: word_tokenize(x))
    return data


def preprocess_data(data, _regular_expressions=False, _remove_capitals=False, _remove_punctuation=False,
                    _remove_stopwords=False, _stemming=False):
    if _regular_expressions:
        # replace numbers with 'numbr'
        data = data.str.replace(r'\d+(\.\d+)?', 'numbr ')
        # replace money symbols with 'moneysymb'
        data = data.str.replace(r'£|\$', 'moneysymb ')

    if _remove_punctuation:
        # remove punctuation
        data = data.str.replace(r'[^\w\d\s]', ' ')
        # replace whitespace with a single space
        data = data.str.replace(r'\s+', ' ')
        # remove leading / trailoring space
        data = data.str.replace(r'%\s+|\s+?$', '')

    if _remove_capitals:
        # remove capital letters
        data = data.str.lower()

    if _remove_stopwords:
        # remove common words
        data = data.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    if _stemming:
        # set words to their basic form
        stm = nltk.PorterStemmer()
        data = data.apply(lambda x: ' '.join(stm.stem(word) for word in x.split()))

    return data


# import data
data = pf.read_csv('train.csv', sep='	')

# split data
ratings = data['Rating']
statements = data['Statement']

# swap truth ratings to numbers
encoder = LabelEncoder()
Y = encoder.fit_transform(ratings)

# preprocess data
statements = preprocess_data(statements, _regular_expressions=True, _remove_capitals=True, _remove_punctuation=True,
                             _remove_stopwords=True, _stemming=True)

# create most used words dictionary
all_words = []
for statement in statements:
    words = word_tokenize(statement)
    for w in words:
        all_words.append(w)


all_words = nltk.FreqDist(all_words)
# print(f'words nr after freqdist: {len(all_words)}')
# print(f'most common: {all_words.most_common()}')

word_features = list(all_words.keys())[:2000]


def find_features(statement):
    words = word_tokenize(statement)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


feature = find_features(statements[0])
for key, value in feature.items():
    if value:
        print(key)



# print(data.head())
#
# data = preprocess_data(data)
#
# print(data.head())
