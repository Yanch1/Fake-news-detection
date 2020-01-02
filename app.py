import tensorflow as tf
import keras_applications,keras_preprocessing
import fasttext
import pandas as pf  # reading csv files
import string

truthRating = {
    0: 'True',
    1: 'Mostly true',
    2: 'Half true',
    3: 'Mostly fake',
    4: 'Fake',
    5: 'Pants on Fire'
}

data = pf.read_csv('train.csv', sep='	')
# data2 = pf.read_csv('test.csv', sep='	')
# print(data.shape)
# print(data2.shape)
# print(data['Statement'])

print(data.head())


# USUWA PRZECINKI

# def remove_punctiation(text):
#     blank_text = "".join([c for c in text if c not in string.punctuation])
#     return blank_text
#
# data['Statement'] = data['Statement'].apply(lambda x: remove_punctiation(x))

# WSZYSTKO Z MALYCH LITER

# data['Statement'] = data['Statement'].apply(lambda x: x.lower())

print(data.head())

# 1. prepare data
#  1.1 delete common words
#  1.2 delete punctuation
