import tensorflow as tf
import keras_applications,keras_preprocessing
import fasttext
import pandas as pf  # reading csv files
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

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


def remove_punctuation(data):

    def remove(text):
        blank_text = [c for c in text if c not in string.punctuation]
        return blank_text

    data['Statement'] = data['Statement'].apply(lambda x: remove(x))
    return data


def remove_capital_letters(data):
    data['Statement'] = data['Statement'].apply(lambda x: x.lower())
    return data


def sentiment_tokenize(data):
    data['Statement'] = data['Statement'].apply(lambda x: sent_tokenize(x))
    return data


def words_tokenize(data):
    data['Statement'] = data['Statement'].apply(lambda x: word_tokenize(x))
    return data


# works on tokens
def remove_stopwords(data):

    def remove(text):
        clean_list = [word for word in text if word not in set(stopwords.words('english'))]
        return clean_list

    data['Statement'] = data['Statement'].apply(lambda x: remove(x))
    return data


def prepare_data(data):

    s1 = remove_capital_letters(data)
    s2 = words_tokenize(s1)
    s3 = remove_punctuation(s2)
    s4 = remove_stopwords(s3)

    return s4


data = pf.read_csv('train.csv', sep='	')

print(data.head())

data = prepare_data(data)

print(data.head())

# print(data.head())
# print(data['Statement'])



# 1. prepare data
#  1.1 delete common words
#  1.2 delete punctuation
