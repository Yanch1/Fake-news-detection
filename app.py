import pandas as pf  # reading csv files
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
from nltk.stem import WordNetLemmatizer

WORDS_NUMBER = 2000

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

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
                    _remove_stopwords=False, _stemming=False, _lemmatizing=False):
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

    if _lemmatizing:
        lemmatizer = WordNetLemmatizer()
        data = data.apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))

    return data


def find_features(statement):
    words = word_tokenize(statement)
    features = []
    for word in word_features:
        features.append(word in words)

    return np.array(features)

######################################################################################################
#   Load and preprocess data
######################################################################################################

full_data = pf.read_csv('full.csv', sep='	', skiprows=[4021, 6315, 9380], index_col=False)

full_data = full_data.sample(frac=1, random_state=1)



ratings = full_data['Truth-Rating']
statements = full_data['Statement']
speakers = full_data['Speaker']

ratings = ratings.replace(0, 1)
ratings = ratings.replace(2, 1)

ratings = ratings.replace(5, 0)
ratings = ratings.replace(4, 0)
ratings = ratings.replace(3, 0)

encoder = LabelEncoder()
labels = encoder.fit_transform(ratings)
speakers = encoder.fit_transform(speakers)

speakers = np.array(speakers)
speakers = np.true_divide(speakers, speakers.argmax())

statements = preprocess_data(statements, _regular_expressions=True, _remove_capitals=True,
                                   _remove_punctuation=True, _remove_stopwords=True, _stemming=True)

all_words = []
for statement in statements:
    words = word_tokenize(statement)
    for w in words:
        all_words.append(w)


all_words = nltk.FreqDist(all_words)
most_common = all_words.most_common(WORDS_NUMBER)

######################################################################################################
#   V2
######################################################################################################


num = 3
dictionary = {}

dictionary['<PAD>'] = 0
dictionary['<START>'] = 1
dictionary['<UNKNOWN>'] = 2

for key, value in most_common:
    dictionary[key] = num
    num = num + 1


data = []
for statement in statements:
    frame = []
    frame.append(dictionary['<START>'])
    for word in word_tokenize(statement):
        if word in dictionary.keys():
            frame.append(dictionary[word])
        else:
            frame.append(dictionary['<UNKNOWN>'])

    while len(frame) < 100:
        frame.append(dictionary['<PAD>'])

    data.append(frame)

data = np.array(data)


######################################################################################################
#   model 2
######################################################################################################

model = keras.Sequential()
model.add(keras.layers.Embedding(WORDS_NUMBER+3, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


######################################################################################################
#  split data
######################################################################################################

split_index = int(len(data) * 0.9)

split_list = np.array_split(data, [split_index])
train_data = split_list[0]
test_data = split_list[1]

split_list = np.array_split(labels, [split_index])
train_labels = split_list[0]
test_labels = split_list[1]

######################################################################################################
#   test 2nd model
######################################################################################################

x_val = train_data[:3000]
x_train = train_data[3000:]

y_val = train_labels[:3000]
y_train = train_labels[3000:]

fitModel = model.fit(x_train, y_train, epochs=50, batch_size=300, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)


######################################################################################################
#   preprocessing
######################################################################################################
word_features = []
for pair in most_common:
    word_features.append(pair[0])


whole_set = []
for statement in statements:
    whole_set.append(find_features(statement))


whole_set = np.array(whole_set)
whole_set = np.column_stack((whole_set, speakers))


split_index = int(len(whole_set) * 0.9)

split_list = np.array_split(whole_set, [split_index])
training = split_list[0]
testing = split_list[1]

split_list = np.array_split(labels, [split_index])
training_labels = split_list[0]
testing_labels = split_list[1]

######################################################################################################
#   Create model 1
######################################################################################################
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(WORDS_NUMBER+1,)),
#     keras.layers.Dense(600, activation="softmax"),
#     keras.layers.Dense(2, activation="softmax")
# ])
#
# # relu - 0.21 acc / 3.97 loss
# # elu - 0.20 acc / 2,69 loss
# # softmax - 0.23 acc / 1.73 loss
# # selu - 0.20 acc / 2.9 loss
# # softplus - 0.21 acc / 2.45 loss
# # softsign - 0.20 acc / 2.76 loss
# # tanh - 0.21 acc / 2.66 loss
#
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(training, training_labels, epochs=20)
#
# ######################################################################################################
# #   Test model 1
# ######################################################################################################
#
# test_loss, test_acc = model.evaluate(testing, testing_labels)
# print(f"test acc = {test_acc}")
# print(f"test loss = {test_loss}")

