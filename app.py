import pandas as pf  # reading csv files
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np

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


def find_features(statement):
    words = word_tokenize(statement)
    features = []
    for word in word_features:
        features.append(word in words)

    return np.array(features)

#############################################################
################# Load and preprocess data ##################
#############################################################

full_data = pf.read_csv('full.csv', sep='	', skiprows=[4021, 6315, 9380])
ratings = full_data['Speaker']
statements = full_data['Truth-Rating']

encoder = LabelEncoder()
labels = encoder.fit_transform(ratings)
statements = preprocess_data(statements, _regular_expressions=True, _remove_capitals=True,
                                   _remove_punctuation=True, _remove_stopwords=True, _stemming=True)

all_words = []
for statement in statements:
    words = word_tokenize(statement)
    for w in words:
        all_words.append(w)


all_words = nltk.FreqDist(all_words)
most_common = all_words.most_common(2000)


word_features = []
for pair in most_common:
    word_features.append(pair[0])


whole_set = []
for statement in statements:
    whole_set.append(find_features(statement))


whole_set = np.array(whole_set)

split_index = int(len(whole_set) * 0.8)

split_list = np.array_split(whole_set, [split_index])
training = split_list[0]
testing = split_list[1]

split_list = np.array_split(labels, [split_index])
training_labels = split_list[0]
testing_labels = split_list[1]

######################################################
################## Create model ######################
######################################################
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2000,)),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(6, activation="softmax")
])

# relu - 0.21 acc / 3.97 loss
# elu - 0.20 acc / 2,69 loss
# softmax - 0.23 acc / 1.73 loss
# selu - 0.20 acc / 2.9 loss
# softplus - 0.21 acc / 2.45 loss
# softsign - 0.20 acc / 2.76 loss
# tanh - 0.21 acc / 2.66 loss


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training, training_labels, epochs=10)

######################################################
################ Test model ##########################
######################################################

test_loss, test_acc = model.evaluate(testing, testing_labels)
print(f"test acc = {test_acc}")
print(f"test loss = {test_loss}")
